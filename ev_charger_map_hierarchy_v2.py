"""전국 전기차 충전소 행정구역 단계형 지도.

원본 CSV는 읽기만 합니다. 시·도 → 시/군/구 → (일반구) → 읍/면/동 → 충전소.
실행: python -m streamlit run ev_charger_map_hierarchy_fixed.py --server.port 8766 --server.address 127.0.0.1
"""

import json
import math
import re
from pathlib import Path

import admdongkor as adk
import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from shapely.geometry import box, mapping
from shapely.ops import unary_union

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / 'ev_charger_navigation_clean.csv'
BOUNDARY_DATE = '20261001'
MAX_POINTS = 450
DAYS = [('Mon', '월'), ('Tue', '화'), ('Wed', '수'), ('Thu', '목'),
        ('Fri', '금'), ('Sat', '토'), ('Sun', '일')]

st.set_page_config(page_title='전국 전기차 충전소 안내', page_icon='⚡', layout='wide')


def parse_groups(value):
    try:
        obj = json.loads(value) if isinstance(value, str) else []
        return [v for v in obj if isinstance(v, dict)] if isinstance(obj, list) else []
    except (ValueError, TypeError):
        return []


def unique_coord(value):
    if isinstance(value, str) and value.strip().startswith('['):
        try:
            numbers = [float(x) for x in json.loads(value) if str(x).strip()]
            return numbers[0] if numbers and max(numbers) - min(numbers) < 1e-6 else np.nan
        except (ValueError, TypeError):
            return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


@st.cache_data(show_spinner='전처리 CSV 불러오는 중...')
def load_csv(path, mtime):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    required = {'statId', 'statNm', 'addr', 'lat', 'lng', 'zcode', 'zscode',
                'total_chargers', 'available_chargers', 'charging_chargers', 'type_output_counts'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'필수 컬럼 누락: {sorted(missing)}')

    for col in ('total_chargers', 'available_chargers', 'charging_chargers'):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
    df['_lat'] = df['lat'].map(unique_coord)
    df['_lng'] = df['lng'].map(unique_coord)
    df['_valid'] = df['_lat'].between(32.5, 39.5) & df['_lng'].between(124, 132)
    df['_groups'] = df['type_output_counts'].map(parse_groups)
    df['_types'] = df['_groups'].map(lambda g: {str(x.get('chgerType', '')).strip() for x in g})
    df['_powers'] = df['_groups'].map(lambda g: {str(x.get('output', '')).strip() for x in g})
    # ※ addr의 첫 단어는 행정구역이 아니다. 지자체 코드/경계를 사용한다.
    df['_province_code'] = df['zcode'].str.strip().str.zfill(2)
    return df


def clean_sig_name(value, province_name):
    """'경기도수원시영통구' → '수원시영통구' (원본은 변경하지 않음)."""
    name = re.sub(r'\s+', '', str(value).strip())
    prefix = re.sub(r'\s+', '', str(province_name))
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    return name


def municipality_name(sgg_name):
    """수원시권선구→수원시, 성남시분당구→성남시, 종로구→종로구."""
    match = re.match(r'^(.+?시)(.+구)$', sgg_name)
    return match.group(1) if match else sgg_name


@st.cache_resource(show_spinner='최신 행정구역 경계 불러오는 중...')
def load_boundaries():
    # 행안부 코드 컬럼은 sidocd / sggcd. 2026-10-01 시점으로 고정한다.
    sido = adk.get(BOUNDARY_DATE, 'sido', crs='EPSG:4326').copy().reset_index(drop=True)
    sgg = adk.get(BOUNDARY_DATE, 'sgg', crs='EPSG:4326').copy().reset_index(drop=True)
    for frame, code, name in ((sido, 'sidocd', 'sidonm'), (sgg, 'sggcd', 'sggnm')):
        if not {code, name}.issubset(frame.columns):
            raise ValueError(f'행정구역 컬럼 확인 필요: {list(frame.columns)}')
        frame['key'] = frame[code].astype(str).str.strip().str[:2 if code == 'sidocd' else 5]
        frame['name'] = frame[name].astype(str).str.strip()

    # sgg 이름이 '수원시권선구' 같은 통합 문자열이어도 계층 분리한다.
    provinces = dict(zip(sido['key'], sido['name']))
    sgg['province'] = sgg['key'].str[:2]
    sgg['sig_name'] = [clean_sig_name(n, provinces.get(p, ''))
                       for n, p in zip(sgg['name'], sgg['province'])]
    sgg['city_name'] = sgg['sig_name'].map(municipality_name)
    sgg['city_key'] = sgg['province'] + '::' + sgg['city_name']
    # 시 아래에 일반구가 있으면 병합하여 시(수원시, 성남시 등) 경계를 생성한다.
    cities = []
    for (p, city_key), grp in sgg.groupby(['province', 'city_key'], sort=False):
        cities.append({'key': city_key, 'province': p, 'name': grp['city_name'].iat[0],
                       'geometry': unary_union(grp.geometry.tolist())})
    city = gpd.GeoDataFrame(cities, geometry='geometry', crs='EPSG:4326')
    return sido, sgg, city


@st.cache_resource(show_spinner='읍·면·동 경계를 불러오는 중...')
def load_dong_boundaries():
    dong = adk.get(BOUNDARY_DATE, 'emd', crs='EPSG:4326').copy().reset_index(drop=True)
    needed = {'emdcd', 'emdnm', 'sggcd', 'geometry'}
    if not needed.issubset(dong.columns):
        raise ValueError(f'읍면동 경계 컬럼 확인 필요: {list(dong.columns)}')
    dong['key'] = dong['emdcd'].astype(str).str.strip()
    dong['name'] = dong['emdnm'].astype(str).str.strip()
    dong['sig_code'] = dong['sggcd'].astype(str).str.strip().str[:5]
    return dong


def assign_polygons(df, polygons, code_name, distance_m=300):
    """좌표로 실제 행정경계 지정. 인접 폴리곤 경계 오차는 300m 이내만 보정."""
    valid = df.index[df['_valid']]
    out = pd.Series('', index=df.index, dtype='object')
    if not len(valid) or polygons.empty:
        return out

    pts = gpd.GeoDataFrame({'id': valid}, index=valid,
                           geometry=gpd.points_from_xy(df.loc[valid, '_lng'], df.loc[valid, '_lat']),
                           crs='EPSG:4326')
    hit = gpd.sjoin(pts, polygons[[code_name, 'geometry']], how='left', predicate='intersects')
    result = hit[code_name].groupby(level=0).first().dropna()
    out.loc[result.index] = result.astype(str)

    # 단순화된 경계에 매우 가까워 매칭되지 않은 점만 근접 폴리곤으로 보정한다.
    unmatched = valid.difference(result.index)
    if len(unmatched):
        near = gpd.sjoin_nearest(pts.loc[unmatched].to_crs(5179),
                                polygons[[code_name, 'geometry']].to_crs(5179),
                                how='left', max_distance=distance_m)
        near_result = near[code_name].groupby(level=0).first().dropna()
        out.loc[near_result.index] = near_result.astype(str)
    return out


@st.cache_data(show_spinner='충전소 위치를 행정구역과 연결하는 중...')
def load_data_with_regions(path, mtime):
    df = load_csv(path, mtime).copy()
    sido, sig, city = load_boundaries()
    valid_provinces = set(sido['key'])
    df['_province_code'] = df['_province_code'].where(df['_province_code'].isin(valid_provinces), '')

    # zscode는 일부 일반구를 대표 시 코드로 기록한다. 실제 구 판단은 좌표가 우선이다.
    df['_sig_code'] = assign_polygons(df, sig, 'key')
    miss = df['_sig_code'].eq('')
    known_sigs = set(sig['key'])
    from_file = df.loc[miss, 'zscode'].str.strip()
    df.loc[miss, '_sig_code'] = from_file.where(from_file.isin(known_sigs), '')

    sig_lookup = sig.drop_duplicates('key').set_index('key')
    spatial_province = df['_sig_code'].map(sig_lookup['province']).fillna('')
    df['_province_code'] = spatial_province.where(spatial_province.ne(''), df['_province_code'])
    df['_city_key'] = df['_sig_code'].map(sig_lookup['city_key']).fillna('')
    return df


def dong_for_region(current, sig_code):
    """선택한 시군구에 대해서만 읍면동 공간 조인을 실행한다."""
    dong = load_dong_boundaries()
    sub = dong[dong['sig_code'].eq(sig_code)].copy()
    result = current.copy()
    result['_dong_code'] = assign_polygons(result, sub, 'key')
    return sub, result


def stats_by(df, key):
    if df.empty:
        return {}
    return df.groupby(key).agg(stations=('statId', 'size'),
                               chargers=('total_chargers', 'sum'),
                               available=('available_chargers', 'sum')).to_dict('index')


def polygons_and_bubbles(polygons, data, group_column):
    counts = stats_by(data, group_column)
    features, bubbles = [], []
    for shape in polygons.itertuples(index=False):
        key, geom = str(shape.key), shape.geometry
        if key not in counts or geom is None or geom.is_empty:
            continue  # 선택 지역의 충전소가 없는 경계는 렌더링하지 않음
        n = int(counts[key]['stations'])
        ch = int(counts[key]['chargers'])
        avail = int(counts[key]['available'])
        name = str(shape.name)
        features.append({'type': 'Feature', 'geometry': mapping(geom),
                         'properties': {'key': key, 'name': name, 'stations': n,
                                        'chargers': ch, 'available': avail}})
        pos = geom.representative_point()
        bubbles.append({'key': key, 'name': name, 'lng': pos.x, 'lat': pos.y,
                        'number': f'{n:,}', 'radius': max(7.0, min(17.0, (13 + 3 * n ** .25) / 2))})
    return {'type': 'FeatureCollection', 'features': features}, pd.DataFrame(bubbles)


def grid_for(current, region, cells=6):
    """좁혀진 동에서만 임시 격자 사용. 전국 단위 고정 0.1도 격자는 사용하지 않는다."""
    lon0, lat0 = current['_lng'].min(), current['_lat'].min()
    lon1, lat1 = current['_lng'].max(), current['_lat'].max()
    dx, dy = max((lon1 - lon0) / cells, 1e-6), max((lat1 - lat0) / cells, 1e-6)
    marked = current.copy()
    x = np.floor((marked['_lng'] - lon0) / dx).astype(int).clip(0, cells - 1)
    y = np.floor((marked['_lat'] - lat0) / dy).astype(int).clip(0, cells - 1)
    marked['_grid'] = x.astype(str) + ':' + y.astype(str)
    shapes = []
    for key in marked['_grid'].unique():
        i, j = map(int, key.split(':'))
        polygon = box(lon0 + i * dx, lat0 + j * dy,
                      lon0 + (i + 1) * dx, lat0 + (j + 1) * dy)
        if region is not None:
            clipped = polygon.intersection(region)
            if not clipped.is_empty:
                polygon = clipped
        shapes.append({'key': key, 'name': f'소구역 {key}', 'geometry': polygon})
    areas = gpd.GeoDataFrame(shapes, geometry='geometry', crs='EPSG:4326')
    geojson, bubbles = polygons_and_bubbles(areas, marked, '_grid')
    return marked, geojson, bubbles, dict(zip(areas['key'], areas.geometry))


def center_view(geom):
    if geom is None:
        return pdk.ViewState(latitude=36.15, longitude=127.85, zoom=6)
    x0, y0, x1, y1 = geom.bounds
    span = max(x1 - x0, y1 - y0, 0.005)
    zoom = min(15.5, max(6, 8.6 - math.log2(span)))
    return pdk.ViewState(latitude=(y0 + y1) / 2, longitude=(x0 + x1) / 2, zoom=zoom)


def draw_map(geojson, bubbles, current, individual, geom):
    layers = []
    if not individual and geojson and geojson['features']:
        layers.append(pdk.Layer('GeoJsonLayer', id='areas', data=geojson,
                                pickable=True, filled=True, stroked=True, auto_highlight=True,
                                get_fill_color=[64, 130, 196, 48],
                                get_line_color=[25, 75, 135, 230],
                                line_width_units='pixels', get_line_width=1.8,
                                highlight_color=[40, 120, 215, 105]))
        if bubbles is not None and not bubbles.empty:
            layers.extend([
                pdk.Layer('ScatterplotLayer', id='circles', data=bubbles,
                          get_position='[lng,lat]', get_radius='radius',
                          radius_units='pixels', radius_min_pixels=7, radius_max_pixels=17,
                          get_fill_color=[29, 103, 181, 225], pickable=False),
                pdk.Layer('TextLayer', id='counts', data=bubbles,
                          get_position='[lng,lat]', get_text='number', get_size=10,
                          get_color=[255, 255, 255, 255],
                          get_text_anchor='"middle"', get_alignment_baseline='"center"',
                          pickable=False),
                pdk.Layer('TextLayer', id='area_labels', data=bubbles,
                          get_position='[lng,lat]', get_text='name', get_size=12,
                          get_color=[23, 48, 76, 255], get_pixel_offset=[0, 22],
                          get_text_anchor='"middle"', get_alignment_baseline='"center"',
                          pickable=False),
            ])
    else:
        pins = current[['statId', 'statNm', 'addr', 'total_chargers',
                        'available_chargers', '_lat', '_lng']].copy()
        pins['color'] = pins['available_chargers'].apply(
            lambda n: [26, 149, 103, 220] if n else [198, 78, 78, 220])
        layers.append(pdk.Layer('ScatterplotLayer', id='stations', data=pins,
                                get_position='[_lng,_lat]', get_radius=4,
                                radius_units='pixels', radius_min_pixels=3, radius_max_pixels=6,
                                get_fill_color='color', pickable=True, auto_highlight=True))

    tooltip = ({'html': '<b>{properties.name}</b><br/>충전소 {properties.stations}곳 / '
                        '충전기 {properties.chargers}대<br/>사용 가능 {properties.available}대',
                'style': {'backgroundColor': '#1b334f', 'color': 'white'}}
               if not individual else
               {'text': '{statNm}\n충전기 {total_chargers}대 / 사용 가능 {available_chargers}대\n{addr}'})
    deck = pdk.Deck(layers=layers, initial_view_state=center_view(geom),
                    map_provider='carto', map_style='light', tooltip=tooltip)
    output = st.pydeck_chart(deck, height=620, on_select='rerun', selection_mode='single-object',
                             key=f"ev_hierarchy_{st.session_state['map_revision']}")
    selection = output.selection.objects
    if selection.get('areas'):
        item = selection['areas'][0]
        return 'area', item.get('properties', item)
    if selection.get('stations'):
        return 'station', selection['stations'][0]
    return None, None


def show_schedule(row):
    schedule = []
    for en, ko in DAYS:
        begin = str(row.get(f'{en}_start', '')).strip()
        end = str(row.get(f'{en}_end', '')).strip()
        if begin == '00:00:00' and end == '00:00:00':
            desc = '휴무'
        elif begin == '00:00:00' and end == '24:00:00':
            desc = '24시간'
        elif begin and end:
            desc = begin[:5] + ' ~ ' + end[:5]
            if begin > end:
                desc += ' (익일 종료)'
        else:
            desc = '미확인'
        schedule.append({'요일': ko, '운영시간': desc})
    st.dataframe(pd.DataFrame(schedule), hide_index=True, use_container_width=True)


def show_station(row):
    st.subheader(row.get('statNm', '') or row['statId'])
    st.caption('충전소 ID: ' + row['statId'])
    st.write('📍', row.get('addr', '') or '주소 미확인')
    detail = ' · '.join(str(row.get(k, '')).strip() for k in ('addrDetail', 'location')
                        if str(row.get(k, '')).strip() not in ('', '확인불가'))
    if detail:
        st.caption('상세위치: ' + detail)
    a, b, c = st.columns(3)
    a.metric('전체 충전기', f"{int(row['total_chargers']):,}대")
    b.metric('사용 가능', f"{int(row['available_chargers']):,}대")
    c.metric('충전 중', f"{int(row['charging_chargers']):,}대")
    groups = [{'충전 규격 코드': v.get('chgerType', ''), '출력(kW)': v.get('output', ''),
               '대수': v.get('count', 0)} for v in row['_groups']]
    st.markdown('**충전 규격 · 출력별 대수**')
    if groups:
        st.dataframe(pd.DataFrame(groups), hide_index=True, use_container_width=True)
    else:
        st.caption('규격별 데이터 없음')
    with st.expander('운영시간 · 이용 조건', expanded=True):
        show_schedule(row)
        for title, col in [('주차 무료 여부', 'parkingFree'), ('이용 제한', 'limitYn'),
                           ('이용 제한 상세', 'limitDetail'), ('비고', 'note'),
                           ('층수', 'floorNum'), ('운영기관', 'busiNm'), ('연락처', 'busiCall')]:
            value = str(row.get(col, '')).strip()
            if value and value != '확인불가':
                st.write(f'**{title}:** {value}')
    if bool(row['_valid']):
        st.link_button('Google 지도에서 길찾기',
                       f"https://www.google.com/maps/dir/?api=1&destination={row['_lat']},{row['_lng']}&travelmode=driving",
                       use_container_width=True)


def reset_deeper(level):
    # sidebar의 위쪽 행정단계를 변경하면 아래 단계만 초기화
    if level == 'province':
        st.session_state['nav_city'] = '전체'
        st.session_state['nav_sig'] = '전체'
    elif level == 'city':
        st.session_state['nav_sig'] = '전체'
    st.session_state['selected_dong'] = None
    st.session_state['grid_path'] = []
    st.session_state['map_revision'] += 1


def navigate(**updates):
    # Streamlit widget 생성 이후 직접 값을 수정하지 않도록 다음 rerun 시작 시 적용
    st.session_state['pending_navigation'] = updates
    st.rerun()


# ───────────────────────────── 메인 화면 ─────────────────────────────
st.title('⚡ 전국 전기차 충전소 지도')
st.caption('시·도 → 시·군·구 → (일반구) → 읍·면·동 → 충전소. '
           '파란 원이 아닌 경계 **영역**을 클릭해 이동합니다. 상태는 CSV 수집 시점 기준입니다.')

if not CSV_PATH.exists():
    st.error('ev_charger_navigation_clean.csv를 이 .py 파일과 같은 폴더에 넣어 주세요.')
    st.stop()

try:
    sido_map, sgg_map, city_map = load_boundaries()
    df = load_data_with_regions(str(CSV_PATH), CSV_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f'경계 자료 또는 CSV 처리 오류: {exc}')
    st.stop()

for key, default in [('nav_province', '전국'), ('nav_city', '전체'), ('nav_sig', '전체'),
                     ('selected_dong', None), ('grid_path', []), ('map_revision', 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# 이전 지도 클릭으로 예약된 값은 selectbox 생성 전 변경해야 한다.
pending = st.session_state.pop('pending_navigation', None)
if pending:
    for key, value in pending.items():
        st.session_state[key] = value
    st.session_state['map_revision'] += 1

prov_names = dict(zip(sido_map['key'], sido_map['name']))
city_names = dict(zip(city_map['key'], city_map['name']))
sig_names = dict(zip(sgg_map['key'], sgg_map['sig_name']))

with st.sidebar:
    st.header('지역 탐색 · 충전소 검색')
    p_choices = ['전국'] + sorted(sido_map['key'].tolist(), key=lambda k: prov_names[k])
    if st.session_state['nav_province'] not in p_choices:
        st.session_state['nav_province'] = '전국'
    p = st.selectbox('광역자치단체 (시·도)', p_choices,
                     format_func=lambda k: '전국' if k == '전국' else prov_names[k],
                     key='nav_province', on_change=reset_deeper, args=('province',))

    city = '전체'
    if p != '전국':
        sub_city = city_map[city_map['province'].eq(p)]
        city_choices = ['전체'] + sorted(sub_city['key'].tolist(), key=lambda k: city_names[k])
        if st.session_state['nav_city'] not in city_choices:
            st.session_state['nav_city'] = '전체'
        city = st.selectbox('시·군 / 구', city_choices,
                            format_func=lambda k: '전체' if k == '전체' else city_names[k],
                            key='nav_city', on_change=reset_deeper, args=('city',))

    sig_code = '전체'
    if city != '전체':
        sub_sgg = sgg_map[sgg_map['city_key'].eq(city)]
        if len(sub_sgg) > 1:  # 수원시 → 영통구/권선구/팔달구/장안구
            sig_choices = ['전체'] + sorted(sub_sgg['key'].tolist(), key=lambda k: sig_names[k])
            if st.session_state['nav_sig'] not in sig_choices:
                st.session_state['nav_sig'] = '전체'
            sig_code = st.selectbox('세부 구', sig_choices,
                                    format_func=lambda k: '전체' if k == '전체' else sig_names[k],
                                    key='nav_sig', on_change=reset_deeper, args=('sig',))

    keyword = st.text_input('충전소 이름·주소 검색', placeholder='예: 서울역, 주차장')
    min_count = st.number_input('최소 충전기 대수', min_value=0, value=0)
    available_only = st.checkbox('사용 가능 충전기 1대 이상')
    types = sorted({v for g in df['_types'] for v in g if v})
    outputs = sorted({v for g in df['_powers'] for v in g if v})
    wanted_type = st.selectbox('충전 규격 코드', ['전체'] + types)
    wanted_power = st.selectbox('출력(kW)', ['전체'] + outputs)
    st.caption('주소 표기가 달라도 행정구역 필터는 동일한 기준으로 작동합니다.')

filtered = df[df['_valid'] & df['_sig_code'].ne('')].copy()
if keyword.strip():
    q = keyword.strip()
    filtered = filtered[filtered['statNm'].str.contains(q, case=False, regex=False)
                        | filtered['addr'].str.contains(q, case=False, regex=False)]
if min_count:
    filtered = filtered[filtered['total_chargers'].ge(min_count)]
if available_only:
    filtered = filtered[filtered['available_chargers'].gt(0)]
if wanted_type != '전체' or wanted_power != '전체':
    filtered = filtered[filtered['_groups'].map(lambda groups: any(
        (wanted_type == '전체' or str(g.get('chgerType', '')).strip() == wanted_type)
        and (wanted_power == '전체' or str(g.get('output', '')).strip() == wanted_power)
        for g in groups))]

# 선택한 위계 안에서만 필터링. 임의 addr 문자열은 이 과정에 사용하지 않음.
current = filtered
view_geom = None
if p != '전국':
    current = current[current['_province_code'].eq(p)]
    shapes = sido_map[sido_map['key'].eq(p)]
    if not shapes.empty:
        view_geom = shapes.geometry.iloc[0]
if city != '전체':
    current = current[current['_city_key'].eq(city)]
    shapes = city_map[city_map['key'].eq(city)]
    if not shapes.empty:
        view_geom = shapes.geometry.iloc[0]
if sig_code != '전체':
    current = current[current['_sig_code'].eq(sig_code)]
    shapes = sgg_map[sgg_map['key'].eq(sig_code)]
    if not shapes.empty:
        view_geom = shapes.geometry.iloc[0]

if current.empty:
    st.warning('조건에 맞는 충전소가 없어요. 검색어나 지역 필터를 변경해 주세요.')
    st.stop()

metrics = st.columns(3)
metrics[0].metric('현재 범위 충전소', f'{len(current):,}곳')
metrics[1].metric('전체 충전기', f"{int(current['total_chargers'].sum()):,}대")
metrics[2].metric('사용 가능', f"{int(current['available_chargers'].sum()):,}대")

# 한 단계 아래 후보가 1개뿐이면 자동으로 그 아래 단계까지 내려간다.
# 이렇게 해야 세종시 등에서 '통째로 하나인 원'만 반복해서 표시되지 않는다.
level, geojson, bubbles, shapes_lookup = None, None, None, {}
selected_prov, selected_city, selected_sig = (None if p == '전국' else p,
                                                None if city == '전체' else city,
                                                None if sig_code == '전체' else sig_code)
if not selected_prov:
    level = 'province'
    geojson, bubbles = polygons_and_bubbles(sido_map, current, '_province_code')
    heading = '전국 — 광역자치단체'
else:
    city_candidates = current['_city_key'].loc[current['_city_key'].ne('')].unique().tolist()
    if not selected_city and len(city_candidates) == 1:
        selected_city = city_candidates[0]
        current = current[current['_city_key'].eq(selected_city)]
        shapes = city_map[city_map['key'].eq(selected_city)]
        if not shapes.empty:
            view_geom = shapes.geometry.iloc[0]
    if not selected_city:
        level = 'city'
        chosen = city_map[city_map['province'].eq(selected_prov)]
        geojson, bubbles = polygons_and_bubbles(chosen, current, '_city_key')
        heading = f"{prov_names.get(selected_prov, '')} — 시·군·구"
    else:
        sig_candidates = current['_sig_code'].loc[current['_sig_code'].ne('')].unique().tolist()
        if not selected_sig and len(sig_candidates) == 1:
            selected_sig = sig_candidates[0]
            current = current[current['_sig_code'].eq(selected_sig)]
            shapes = sgg_map[sgg_map['key'].eq(selected_sig)]
            if not shapes.empty:
                view_geom = shapes.geometry.iloc[0]
        if not selected_sig:
            level = 'sig'
            chosen = sgg_map[sgg_map['city_key'].eq(selected_city)]
            geojson, bubbles = polygons_and_bubbles(chosen, current, '_sig_code')
            heading = f"{city_names.get(selected_city, '')} — 세부 구"

if level is None:
    # 시군구 다음에는 원 수가 많을 때 읍면동 행정경계를 먼저 제공한다.
    if len(current) > MAX_POINTS or st.session_state['selected_dong']:
        sub_dong, enriched = dong_for_region(current, selected_sig)
        candidate_codes = enriched['_dong_code'].loc[enriched['_dong_code'].ne('')].unique().tolist()
        selected_dong = st.session_state['selected_dong']
        if selected_dong and selected_dong not in candidate_codes:
            selected_dong = None
            st.session_state['selected_dong'] = None
            st.session_state['grid_path'] = []
        if not selected_dong and len(candidate_codes) == 1:
            selected_dong = candidate_codes[0]
        if not selected_dong and len(candidate_codes) > 1:
            level = 'dong'
            geojson, bubbles = polygons_and_bubbles(sub_dong, enriched, '_dong_code')
            current = enriched
            heading = f"{sig_names.get(selected_sig, '')} — 읍·면·동"
        elif selected_dong:
            current = enriched[enriched['_dong_code'].eq(selected_dong)]
            row = sub_dong[sub_dong['key'].eq(selected_dong)]
            if not row.empty:
                view_geom = row.geometry.iloc[0]

    # 선택한 읍면동에 충전소가 450곳 초과일 때만 더 작은 영역으로 분할한다.
    if level is None:
        for cell in st.session_state['grid_path']:
            marked, _, _, polys = grid_for(current, view_geom)
            current = marked[marked['_grid'].eq(cell)].drop(columns='_grid')
            if cell in polys:
                view_geom = polys[cell]
        if len(current) > MAX_POINTS and len(st.session_state['grid_path']) < 6:
            marked, geojson, bubbles, shapes_lookup = grid_for(current, view_geom)
            if bubbles is not None and len(bubbles) > 1:
                level = 'grid'
                heading = f'좁혀진 구역 — 소구역별 집계 ({len(current):,}곳)'
            else:
                level = 'station'
                heading = '같은 위치의 충전소 — 목록으로 선택'
        else:
            level = 'station'
            heading = f'개별 충전소 ({len(current):,}곳)'

st.subheader(heading)
trail = ['전국']
if selected_prov:
    trail.append(prov_names.get(selected_prov, selected_prov))
if selected_city:
    trail.append(city_names.get(selected_city, selected_city))
if selected_sig and selected_sig != selected_city:
    trail.append(sig_names.get(selected_sig, selected_sig))
if st.session_state['selected_dong']:
    trail.append('읍·면·동')
st.caption('  ›  '.join(trail))

nav_left, nav_right = st.columns([1, 3])
with nav_left:
    if st.button('← 이전 단계', use_container_width=True, disabled=(p == '전국')):
        if st.session_state['grid_path']:
            st.session_state['grid_path'].pop()
            st.session_state['map_revision'] += 1
            st.rerun()
        if st.session_state['selected_dong']:
            st.session_state['selected_dong'] = None
            st.session_state['map_revision'] += 1
            st.rerun()
        if sig_code != '전체':
            navigate(nav_sig='전체', selected_dong=None, grid_path=[])
        elif city != '전체':
            navigate(nav_city='전체', nav_sig='전체', selected_dong=None, grid_path=[])
        else:
            navigate(nav_province='전국', nav_city='전체', nav_sig='전체',
                     selected_dong=None, grid_path=[])
with nav_right:
    if p != '전국' and st.button('전국 지도'):
        navigate(nav_province='전국', nav_city='전체', nav_sig='전체',
                 selected_dong=None, grid_path=[])

if level == 'station' and len(current) > MAX_POINTS:
    st.info('동일 좌표에 충전소가 밀집되어 있어 아래 목록으로 선택할 수 있습니다.')
else:
    clicked_type, clicked = draw_map(geojson, bubbles, current, level == 'station', view_geom)
    if clicked_type == 'area':
        key = str(clicked.get('key', ''))
        if level == 'province':
            navigate(nav_province=key, nav_city='전체', nav_sig='전체',
                     selected_dong=None, grid_path=[])
        if level == 'city':
            navigate(nav_city=key, nav_sig='전체', selected_dong=None, grid_path=[])
        if level == 'sig':
            navigate(nav_sig=key, selected_dong=None, grid_path=[])
        if level == 'dong':
            st.session_state['selected_dong'] = key
            st.session_state['grid_path'] = []
            st.session_state['map_revision'] += 1
            st.rerun()
        if level == 'grid' and key in shapes_lookup:
            st.session_state['grid_path'].append(key)
            st.session_state['map_revision'] += 1
            st.rerun()
    elif clicked_type == 'station':
        st.session_state['selected_station'] = str(clicked.get('statId', ''))

st.caption('파란 원은 개수 표시용입니다. 지도에 색칠된 행정구역 안을 클릭하면 아래 단계로 이동합니다.')
st.divider()
left, right = st.columns([1.1, 1], gap='large')
with left:
    st.subheader('현재 영역 충전소 목록')
    display_cols = ['statNm', 'addr', 'total_chargers', 'available_chargers']
    st.dataframe(current[display_cols].head(300), hide_index=True, use_container_width=True)
    choices = current[['statId', 'statNm']].head(300)
    labels = dict(zip(choices['statId'], choices['statNm']))
    picked = st.selectbox('상세정보 조회', ['선택 안 함'] + choices['statId'].tolist(),
                          format_func=lambda v: ('선택 안 함' if v == '선택 안 함'
                                                 else f"{labels.get(v, '')} ({v})"))
    if picked != '선택 안 함':
        st.session_state['selected_station'] = picked
    st.caption(f'현재 영역 {len(current):,}곳 중 최대 300곳을 목록에 표시합니다.')
with right:
    st.subheader('충전소 상세정보')
    chosen_id = st.session_state.get('selected_station', '')
    found = current[current['statId'].eq(chosen_id)] if chosen_id else current.iloc[:0]
    if not found.empty:
        show_station(found.iloc[0])
    else:
        st.write('개별 충전소 마커나 목록을 선택하면 상세정보가 표시됩니다.')

st.caption('행정경계: 2026-10-01 기준 admdongkor / 통계청·행정안전부 기반. 원본 CSV는 읽기 전용입니다.')
