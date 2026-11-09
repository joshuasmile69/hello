"""Streamlit EV charging station map | input: ev_charger_navigation_clean.csv.

Run: python -m streamlit run ev_charger_map_admin.py --server.port 8765 --server.address 127.0.0.1
"""

import json
import math
import re
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
import admdongkor as adk
from shapely.geometry import mapping, box
from shapely.ops import unary_union
import pydeck as pdk
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'ev_charger_navigation_clean.csv'
# 개발 중에는 구버전 26컬럼 파일도 시험할 수 있다.
FALLBACK_FILE = BASE_DIR / 'ev_charger_navigation.csv'
DAYS = [('Mon', '월'), ('Tue', '화'), ('Wed', '수'), ('Thu', '목'),
        ('Fri', '금'), ('Sat', '토'), ('Sun', '일')]
MAX_DETAIL_POINTS = 450  # 지정 영역 내 450곳 이하에서만 개별 충전소 표시
BOUNDARY_DATE = "20261001"  # 2026년 10월 1일 기준 행정구역 경계

st.set_page_config(page_title='전국 전기차 충전소 안내', page_icon='⚡', layout='wide')


def json_array(value):
    """충전기 종류별 수량 JSON을 안전하게 리스트로 복원한다."""
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return []


def unique_coord(value):
    """좌표가 JSON 배열이면 같은 숫자일 때만 대표값 사용; 다르면 지도에서 제외."""
    if isinstance(value, str) and value.lstrip().startswith('['):
        try:
            values = [float(v) for v in json.loads(value) if str(v).strip()]
            if values and max(values) - min(values) < 0.000001:
                return values[0]
            return float('nan')
        except (ValueError, TypeError):
            return float('nan')
    try:
        return float(value)
    except (ValueError, TypeError):
        return float('nan')


def parse_types(s):
    """[{'chgerType':'04','output':'100','count':2},...] 읽기."""
    return [x for x in json_array(s) if isinstance(x, dict)]


@st.cache_data(show_spinner='충전소 데이터를 불러오는 중입니다.')
def load_data(csv_path, modified_time):
    # 모든 값을 문자열로 읽어 충전기 ID, 지역 코드의 앞자리 0을 보존한다.
    data = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    needed = {'statId', 'statNm', 'addr', 'lat', 'lng', 'total_chargers',
              'available_chargers', 'charging_chargers', 'type_output_counts'}
    missing = needed - set(data.columns)
    if missing:
        raise ValueError('CSV에 필요한 컬럼이 없습니다: ' + ', '.join(sorted(missing)))

    for col in ('total_chargers', 'available_chargers', 'charging_chargers'):
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype('int64')

    data['_lat'] = data['lat'].map(unique_coord)
    data['_lng'] = data['lng'].map(unique_coord)
    data['_valid_location'] = data['_lat'].between(32.5, 39.5) & data['_lng'].between(124, 132)
    data['_region'] = data['addr'].str.strip().str.split().str[0].fillna('지역 미상')
    data.loc[data['_region'].str.startswith('[') | data['_region'].eq(''), '_region'] = '지역 미상'

    # JSON 파싱을 한 번만 수행한 뒤 사이드바 필터에 재사용한다.
    data['_groups'] = data['type_output_counts'].map(parse_types)
    data['_types'] = data['_groups'].map(
        lambda groups: set(str(x.get('chgerType', '')).strip() for x in groups)
    )
    data['_outputs'] = data['_groups'].map(
        lambda groups: set(str(x.get('output', '')).strip() for x in groups)
    )
    return data


def render_schedule(row):
    """최종 39개 컬럼에 있는 주간 운영시간 표를 출력한다."""
    records = []
    for en, ko in DAYS:
        start, end = str(row.get(f'{en}_start', '')).strip(), str(row.get(f'{en}_end', '')).strip()
        if start == '00:00:00' and end == '00:00:00':
            text = '휴무'
        elif start == '00:00:00' and end == '24:00:00':
            text = '24시간'
        elif start and end:
            text = f'{start[:5]} ~ {end[:5]}'
            if start > end:  # 18:00 ~ 익일 07:00 등
                text += ' (익일 종료)'
        else:
            text = '시간 미확인'
        records.append({'요일': ko, '운영시간': text})
    if any(f'{en}_start' in row.index for en, _ in DAYS):
        st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)
    elif 'useTime' in row.index:
        st.write(row['useTime'] or '운영시간 미확인')
    else:
        st.caption('운영시간 미확인')


def show_station(row):
    name = row.get('statNm', '') or row['statId']
    st.subheader(name)
    st.caption(f"충전소 ID: {row['statId']}")
    st.write('📍', row.get('addr', '') or '주소 미확인')
    pos = ' · '.join(str(row.get(k, '')).strip() for k in ('addrDetail', 'location')
                     if str(row.get(k, '')).strip() not in ('', '확인불가'))
    if pos:
        st.caption('상세위치: ' + pos)

    m1, m2, m3 = st.columns(3)
    m1.metric('전체 충전기', f"{int(row['total_chargers']):,}대")
    m2.metric('사용 가능', f"{int(row['available_chargers']):,}대")
    m3.metric('충전 중', f"{int(row['charging_chargers']):,}대")

    items = []
    for x in row['_groups']:
        items.append({
            '충전 규격 코드': x.get('chgerType', '확인불가'),
            '출력 (kW)': x.get('output', '확인불가'),
            '대수': x.get('count', 0),
        })
    st.markdown('**충전기 규격 · 출력별 설치 대수**')
    if items:
        st.dataframe(pd.DataFrame(items), hide_index=True, use_container_width=True)
    else:
        st.caption('종류별 정보 없음')

    with st.expander('운영시간 · 이용 조건', expanded=True):
        render_schedule(row)
        for label, key in [('주차 무료 여부', 'parkingFree'), ('이용 제한 여부', 'limitYn'),
                           ('이용 제한 상세', 'limitDetail'), ('안내사항', 'note'),
                           ('설치 층수', 'floorNum'), ('운영기관', 'busiNm'),
                           ('운영기관 연락처', 'busiCall')]:
            value = str(row.get(key, '')).strip()
            if value and value != '확인불가':
                st.write(f'**{label}:** {value}')

    if bool(row['_valid_location']):
        url = f"https://www.google.com/maps/dir/?api=1&destination={row['_lat']},{row['_lng']}&travelmode=driving"
        st.link_button('Google 지도에서 길찾기', url, use_container_width=True)



# ──────────────── 2026년 행정구역 경계·충전소 위치 연결 ────────────────
@st.cache_resource(show_spinner='2026년 시·도 / 시·군·구 경계를 불러오는 중...')
def load_boundaries():
    # admdongkor: 통계청 SGIS 행정구역 경계 / EPSG:4326 좌표로 가져온다.
    # 최초 한 번은 GitHub에서 경계 파일을 내려받고 이후 로컬 캐시를 재사용한다.
    sido = adk.get(BOUNDARY_DATE, 'sido', crs='EPSG:4326').reset_index(drop=True)
    sgg = adk.get(BOUNDARY_DATE, 'sgg', crs='EPSG:4326').reset_index(drop=True)

    def code_name(gdf, length):
        # admdongkor의 실제 컬럼명: 시·도=sidocd/sidonm, 시·군·구=sggcd/sggnm
        code_col = 'sidocd' if length == 2 else 'sggcd'
        name_col = 'sidonm' if length == 2 else 'sggnm'
        if code_col not in gdf.columns or name_col not in gdf.columns:
            raise ValueError(f'행정경계 코드·이름 컬럼 확인 필요: {list(gdf.columns)}')
        gdf = gdf.copy()
        gdf['key'] = gdf[code_col].astype(str).str[:length]
        gdf['name'] = gdf[name_col].astype(str).str.strip()
        return gdf

    sido = code_name(sido, 2)
    sgg = code_name(sgg, 5)
    sgg['province'] = sgg['key'].str[:2]
    province_names = dict(zip(sido['key'], sido['name']))

    def city_from_sig(row):
        # 수원시 영통구 → 수원시 / 서울특별시 종로구 → 종로구
        value = row['name'].strip()
        prefix = province_names.get(row['province'], '')
        if value.startswith(prefix) and prefix:
            value = value[len(prefix):].strip()
        # '경기도 수원시영통구' 같이 붙어 있는 원본도 처리.
        match = re.match(r'^(.+?시)\s*(.+구)$', value)
        return match.group(1) if match else value

    sgg['city_name'] = sgg.apply(city_from_sig, axis=1)
    sgg['city_key'] = sgg['province'] + '::' + sgg['city_name']
    sgg['sig_name'] = sgg['name']

    # 시 단위에 하위 구가 있으면 해당 구들을 병합하여 시의 경계를 만든다.
    cities = []
    for city_key, group in sgg.groupby('city_key', sort=False):
        merged = unary_union(group.geometry.tolist())
        cities.append({
            'key': city_key,
            'province': group['province'].iat[0],
            'name': group['city_name'].iat[0],
            'sig_codes': group['key'].tolist(),
            'geometry': merged,
        })

    city = gpd.GeoDataFrame(cities, geometry='geometry', crs='EPSG:4326')
    return sido, sgg, city


@st.cache_data(show_spinner='충전소 좌표와 행정구역을 연결하는 중...')
def load_with_regions(csv_path, modified_time):
    data = load_data(csv_path, modified_time).copy()
    _, sig, _ = load_boundaries()
    data['_sig_code'] = ''

    # 좌표가 경계 폴리곤 내부에 있는지 실제로 계산한다.
    # 텍스트 주소나 과거 시군구 코드에만 의존하지 않아 2026년 변경도 대응.
    valid = data.index[data['_valid_location']]
    if len(valid):
        points = gpd.GeoDataFrame(
            {'row_id': valid.to_numpy()},
            geometry=gpd.points_from_xy(data.loc[valid, '_lng'], data.loc[valid, '_lat']),
            crs='EPSG:4326', index=valid,
        )
        joined = gpd.sjoin(points, sig[['key', 'geometry']], how='left', predicate='within')
        matched = joined['key'].groupby(level=0).first().fillna('')
        data.loc[matched.index, '_sig_code'] = matched.to_numpy()

    # 경계 바로 위에 있는 좌표 등 공간 매칭에서 빠진 경우, 5자리 코드가 같으면 보완.
    missing = data['_sig_code'].eq('') & data['_valid_location']
    codes = set(sig['key'])
    guessed = data.loc[missing, 'zscode'].astype(str).str.strip()
    data.loc[guessed.index, '_sig_code'] = guessed.where(guessed.isin(codes), '')

    lookup = sig.set_index('key')[['province', 'city_key']]
    data['_province_code'] = data['_sig_code'].map(lookup['province']).fillna('')
    data['_city_key'] = data['_sig_code'].map(lookup['city_key']).fillna('')
    return data


def summary_for(data, key):
    if data.empty:
        return {}
    counts = data.groupby(key, sort=False).agg(
        stations=('statId', 'size'),
        chargers=('total_chargers', 'sum'),
        available=('available_chargers', 'sum'),
    )
    return counts.to_dict('index')


def shape_features(gdf, data, key_col, name_col):
    """클릭 가능한 행정구역 면을 만들고, 원은 시각화만 담당하도록 분리."""
    counts = summary_for(data, key_col)
    features, circles = [], []

    for row in gdf.itertuples(index=False):
        key = str(getattr(row, 'key'))
        name = str(getattr(row, 'name'))
        geom = getattr(row, 'geometry')
        if geom is None or geom.is_empty:
            continue
        metrics = counts.get(key, {})
        stations = int(metrics.get('stations', 0))
        chargers = int(metrics.get('chargers', 0))
        available = int(metrics.get('available', 0))
        features.append({
            'type': 'Feature', 'geometry': mapping(geom),
            'properties': {'key': key, 'name': name, 'stations': stations,
                           'chargers': chargers, 'available': available},
        })
        if stations:
            point = geom.representative_point()  # 다각형 내부의 실제 위치를 선택
            circles.append({'key': key, 'name': name, 'lng': point.x, 'lat': point.y,
                            'stations': stations, 'chargers': chargers,
                            'available': available, 'number': f'{stations:,}',
                            # 기존 반지름 공식의 정확히 절반(픽셀).
                            'radius': max(7.5, min(18, (13 + 3 * stations ** 0.25) / 2))})

    return {'type': 'FeatureCollection', 'features': features}, pd.DataFrame(circles)


def grid_level(data, clip_geom, side=7):
    """시군구에 충전소가 450곳을 넘으면 실제 경계 안쪽만 격자로 분할."""
    x0, y0, x1, y1 = (float(data['_lng'].min()), float(data['_lat'].min()),
                      float(data['_lng'].max()), float(data['_lat'].max()))
    dx, dy = max((x1 - x0) / side, 1e-6), max((y1 - y0) / side, 1e-6)
    current = data.copy()
    ix = np.floor((current['_lng'] - x0) / dx).astype(int).clip(0, side - 1)
    iy = np.floor((current['_lat'] - y0) / dy).astype(int).clip(0, side - 1)
    current['_grid_key'] = ix.astype(str) + ':' + iy.astype(str)

    counts = summary_for(current, '_grid_key')
    records = []
    for key, stats in counts.items():
        i, j = (int(v) for v in key.split(':'))
        shape = box(x0 + i * dx, y0 + j * dy, x0 + (i + 1) * dx, y0 + (j + 1) * dy)
        shape = shape.intersection(clip_geom)
        if shape.is_empty:
            shape = box(x0 + i * dx, y0 + j * dy, x0 + (i + 1) * dx, y0 + (j + 1) * dy)
        records.append({'key': key, 'name': f'구역 {key}', 'geometry': shape})

    poly = gpd.GeoDataFrame(records, geometry='geometry', crs='EPSG:4326')
    # 집계 원 자체는 클릭하지 않으므로 구역 내 원하는 위치를 눌러 내려갈 수 있다.
    geojson, circles = shape_features(poly, current, '_grid_key', 'name')
    return current, geojson, circles, {r['key']: r['geometry'] for r in records}


def view_for(geometry=None, province=False):
    if geometry is None:
        return pdk.ViewState(latitude=36.2, longitude=127.8, zoom=6)
    xmin, ymin, xmax, ymax = geometry.bounds
    span = max(xmax - xmin, ymax - ymin, 0.004)
    zoom = min(15, max(7 if province else 8, 8.6 - math.log2(span)))
    return pdk.ViewState(latitude=(ymin + ymax) / 2,
                         longitude=(xmin + xmax) / 2, zoom=zoom)


def show_map(geojson, bubbles, geometry=None, individual=None, revision=0):
    """polygon click: 지도의 행정구역 면적 선택. 원과 지역명은 pickable=False."""
    layers = []
    if geojson and geojson['features']:
        layers.append(pdk.Layer(
            'GeoJsonLayer', id='areas', data=geojson,
            stroked=True, filled=True, pickable=True, auto_highlight=True,
            get_fill_color=[67, 129, 202, 45],
            get_line_color=[27, 76, 142, 230],
            line_width_units='pixels', get_line_width=1.8,
            highlight_color=[36, 133, 212, 100],
        ))
    if bubbles is not None and not bubbles.empty:
        layers.extend([
            pdk.Layer(
                'ScatterplotLayer', id='bubbles', data=bubbles,
                get_position='[lng,lat]', get_radius='radius', radius_units='pixels',
                radius_min_pixels=7.5, radius_max_pixels=18,
                get_fill_color=[30, 105, 185, 215], pickable=False,
            ),
            pdk.Layer(
                'TextLayer', id='numbers', data=bubbles,
                get_position='[lng,lat]', get_text='number', get_size=10,
                get_color=[255, 255, 255, 255], get_text_anchor='"middle"',
                get_alignment_baseline='"center"', pickable=False,
            ),
            pdk.Layer(
                'TextLayer', id='names', data=bubbles,
                get_position='[lng,lat]', get_text='name', get_size=12,
                get_color=[28, 47, 67, 255], get_pixel_offset=[0, 24],
                get_text_anchor='"middle"', get_alignment_baseline='"center"',
                pickable=False,
            ),
        ])
    if individual is not None and not individual.empty:
        scatter = individual[['statId', 'statNm', 'addr', '_lat', '_lng',
                              'total_chargers', 'available_chargers']].copy()
        scatter['color'] = scatter['available_chargers'].apply(
            lambda n: [27, 153, 107, 215] if n else [207, 88, 88, 215])
        layers.append(pdk.Layer(
            'ScatterplotLayer', id='stations', data=scatter,
            get_position='[_lng,_lat]', get_radius=4, radius_units='pixels',
            radius_min_pixels=3, radius_max_pixels=6,
            get_fill_color='color', pickable=True, auto_highlight=True,
        ))
    tooltip = {
        'html': '<b>{properties.name}</b><br/>충전소 {properties.stations}곳 / '
                '충전기 {properties.chargers}대<br/>사용 가능 {properties.available}대<br/>'
                '<small>해당 지역을 클릭하면 확대됩니다.</small>',
        'style': {'backgroundColor': '#20364d', 'color': 'white'},
    } if individual is None else {
        'text': '{statNm}\n충전기 {total_chargers}대 / 사용 가능 {available_chargers}대\n{addr}'
    }
    deck = pdk.Deck(
        layers=layers, initial_view_state=view_for(geometry),
        map_provider='carto', map_style='light', tooltip=tooltip,
    )
    result = st.pydeck_chart(deck, on_select='rerun', selection_mode='single-object',
                            key=f'ev_admin_map_{revision}', height=620)
    selected = result.selection.objects
    stations = selected.get('stations', [])
    if stations:
        return 'station', stations[0]
    area = selected.get('areas', [])
    if area:
        picked = area[0]
        props = picked.get('properties', picked)
        return 'area', props
    return None, None


def reset_to(level):
    # stage hierarchy: province → city → district(optional) → grid → station
    if level == 'all':
        for name in ('province', 'city', 'district'):
            st.session_state[name] = None
        st.session_state['grid_path'] = []
    elif level == 'province':
        st.session_state['city'] = None
        st.session_state['district'] = None
        st.session_state['grid_path'] = []
    elif level == 'city':
        st.session_state['district'] = None
        st.session_state['grid_path'] = []
    elif level == 'district':
        st.session_state['grid_path'] = []
    st.session_state['map_rev'] += 1


# ──────────────────────────── 화면 구성 ────────────────────────────
st.title('⚡ 전국 전기차 충전소 지도')
st.caption('시·도 → 시/군 → 구(있는 지역) → 세부 구역 → 개별 충전소. '
           '구역 내 지도를 클릭하면 확대됩니다. 충전기 상태는 CSV 수집 시점 기준입니다.')

csv_path = DATA_FILE if DATA_FILE.exists() else FALLBACK_FILE
if not csv_path.exists():
    st.error('ev_charger_navigation_clean.csv를 이 .py 파일과 같은 폴더에 넣어 주세요.')
    st.stop()
if csv_path == FALLBACK_FILE:
    st.info('정제된 CSV가 없어 기존 ev_charger_navigation.csv를 사용하고 있습니다.')

try:
    sido_map, sgg_map, city_map = load_boundaries()
    df = load_with_regions(str(csv_path), csv_path.stat().st_mtime)
except Exception as exc:
    st.error(f'행정구역 경계 또는 CSV 처리 오류: {exc}')
    st.caption('최초 실행 시 GitHub 지도 자료를 받아야 하므로 인터넷 연결이 필요합니다. '
               'pip install -r requirements_admin_map.txt 설치 여부도 확인해 주세요.')
    st.stop()

with st.sidebar:
    st.header('충전소 검색 · 필터')
    regions = ['전국'] + sorted(x for x in df['_region'].unique() if x != '지역 미상')
    region = st.selectbox('시·도', regions)
    keyword = st.text_input('이름·주소 검색', placeholder='예: 서울역, 공영주차장, 강남구')
    charger_min = st.number_input('최소 충전기 대수', min_value=0, value=0, step=1)
    available_only = st.checkbox('사용 가능 충전기 1대 이상')
    all_types = sorted({v for vals in df['_types'] for v in vals if v})
    typ = st.selectbox('충전 규격 코드', ['전체'] + all_types)
    all_outputs = sorted({v for vals in df['_outputs'] for v in vals if v})
    power = st.selectbox('출력(kW)', ['전체'] + all_outputs)
    st.caption('사용 가능 대수는 실시간 정보가 아닙니다.')

filtered = df
if region != '전국':
    filtered = filtered[filtered['_region'].eq(region)]
if keyword.strip():
    q = keyword.strip()
    filtered = filtered[
        filtered['statNm'].str.contains(q, regex=False, case=False, na=False)
        | filtered['addr'].str.contains(q, regex=False, case=False, na=False)
    ]
if charger_min:
    filtered = filtered[filtered['total_chargers'].ge(charger_min)]
if available_only:
    filtered = filtered[filtered['available_chargers'].gt(0)]
if typ != '전체' or power != '전체':
    filtered = filtered[filtered['_groups'].map(
        lambda groups: any(
            (typ == '전체' or str(g.get('chgerType', '')).strip() == typ)
            and (power == '전체' or str(g.get('output', '')).strip() == power)
            for g in groups
        ))]

metric_cols = st.columns(4)
metric_cols[0].metric('검색된 충전소', f'{len(filtered):,}곳')
metric_cols[1].metric('전체 충전기', f"{int(filtered['total_chargers'].sum()):,}대")
metric_cols[2].metric('사용 가능', f"{int(filtered['available_chargers'].sum()):,}대")
metric_cols[3].metric('행정구역 매칭', f"{int(filtered['_sig_code'].ne('').sum()):,}곳")

if filtered.empty:
    st.warning('조건에 맞는 충전소가 없습니다.')
    st.stop()

mapped = filtered[filtered['_sig_code'].ne('') & filtered['_valid_location']].copy()
if mapped.empty:
    st.warning('행정구역 경계 내에서 표시할 충전소가 없습니다.')
    st.stop()

for key, default in [('province', None), ('city', None), ('district', None),
                     ('grid_path', []), ('map_rev', 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# 필터 변경으로 현재 선택한 구역에 검색 결과가 없으면 전국 단계로 복귀.
if st.session_state['province'] and not mapped['_province_code'].eq(st.session_state['province']).any():
    reset_to('all')
if st.session_state['city'] and not mapped['_city_key'].eq(st.session_state['city']).any():
    reset_to('province')
if st.session_state['district'] and not mapped['_sig_code'].eq(st.session_state['district']).any():
    reset_to('city')

province = st.session_state['province']
city = st.session_state['city']
district = st.session_state['district']
current = mapped
view_geom = None

if province:
    current = current[current['_province_code'].eq(province)]
    matching = sido_map[sido_map['key'].eq(province)]
    if not matching.empty:
        view_geom = matching.geometry.iloc[0]
if city:
    current = current[current['_city_key'].eq(city)]
    matching = city_map[city_map['key'].eq(city)]
    if not matching.empty:
        view_geom = matching.geometry.iloc[0]
if district:
    current = current[current['_sig_code'].eq(district)]
    matching = sgg_map[sgg_map['key'].eq(district)]
    if not matching.empty:
        view_geom = matching.geometry.iloc[0]

# 광역단체 → 시·군 → 구 경계. 하나의 시·군·구만 있는 경우 그 단계를 건너뛴다.
if not province:
    level = 'province'
    areas, bubbles = shape_features(sido_map, current, '_province_code', 'name')
    title = '전국 · 시·도 지도'
elif not city:
    level = 'city'
    subset = city_map[city_map['province'].eq(province)]
    areas, bubbles = shape_features(subset, current, '_city_key', 'name')
    title = '선택한 광역단체 · 시/군/구 경계'
elif not district and sgg_map[sgg_map['city_key'].eq(city)].shape[0] > 1:
    level = 'district'
    subset = sgg_map[sgg_map['city_key'].eq(city)]
    areas, bubbles = shape_features(subset, current, '_sig_code', 'name')
    title = '선택한 시 · 구별 경계'
else:
    # 선택한 시군구에 450곳을 초과하면 맵 전체에 마커를 뿌리지 않는다.
    for cell_id in st.session_state['grid_path']:
        expanded, _, _, shapes = grid_level(current, view_geom)
        current = expanded[expanded['_grid_key'].eq(cell_id)].drop(columns=['_grid_key'])
        if not current.empty and cell_id in shapes:
            view_geom = shapes[cell_id]
    if len(current) > MAX_DETAIL_POINTS:
        level = 'grid'
        expanded, areas, bubbles, shapes = grid_level(current, view_geom)
        title = f'선택 지역 · 구역별 집계 ({len(current):,}곳)'
    else:
        level = 'station'
        areas, bubbles = None, None
        title = f'개별 충전소 ({len(current):,}곳)'

st.subheader(title)
left, right = st.columns([1, 4])
with left:
    if province and st.button('← 이전 단계', use_container_width=True):
        if st.session_state['grid_path']:
            st.session_state['grid_path'].pop()
            st.session_state['map_rev'] += 1
        elif district:
            reset_to('city')
        elif city:
            reset_to('province')
        else:
            reset_to('all')
        st.rerun()
with right:
    if province and st.button('전국으로 돌아가기'):
        reset_to('all')
        st.rerun()

if level == 'grid' and len(bubbles) == 1:
    st.info('같은 위치에 많은 충전소가 겹쳐 있습니다. 아래 목록에서 선택해 주세요.')
else:
    selected_kind, selected = show_map(
        areas, bubbles, geometry=view_geom,
        individual=current if level == 'station' else None,
        revision=st.session_state['map_rev'],
    )
    if selected_kind == 'area':
        chosen = str(selected.get('key', ''))
        if level == 'province' and mapped['_province_code'].eq(chosen).any():
            st.session_state['province'] = chosen
            reset_to('province')
            st.rerun()
        elif level == 'city' and current['_city_key'].eq(chosen).any():
            st.session_state['city'] = chosen
            reset_to('city')
            st.rerun()
        elif level == 'district' and current['_sig_code'].eq(chosen).any():
            st.session_state['district'] = chosen
            reset_to('district')
            st.rerun()
        elif level == 'grid' and chosen in shapes:
            st.session_state['grid_path'].append(chosen)
            st.session_state['map_rev'] += 1
            st.rerun()
    elif selected_kind == 'station':
        st.session_state['selected_station'] = str(selected.get('statId'))

st.caption('경계 영역 안을 클릭하면 해당 시·도 / 시·군·구로 이동합니다. 파란 숫자 원은 통계 표시 전용이며 클릭할 필요가 없습니다.')

st.divider()
a, b = st.columns([1.1, 1], gap='large')
with a:
    st.subheader('현재 범위 충전소 목록')
    st.dataframe(current[['statNm', 'addr', 'total_chargers', 'available_chargers']].head(300),
                 hide_index=True, use_container_width=True)
    choices = current[['statId', 'statNm']].head(300)
    labels = dict(zip(choices['statId'], choices['statNm']))
    picked = st.selectbox('상세정보 조회', ['선택 안 함'] + choices['statId'].tolist(),
                          format_func=lambda x: f"{labels.get(x, '')} ({x})" if x != '선택 안 함' else x)
    if picked != '선택 안 함':
        st.session_state['selected_station'] = picked
    st.caption(f'현재 선택된 범위 {len(current):,}곳 중 최대 300곳을 목록에 표시합니다.')
with b:
    st.subheader('충전소 상세정보')
    selected_id = st.session_state.get('selected_station')
    found = current[current['statId'].eq(selected_id)] if selected_id else current.iloc[:0]
    if not found.empty:
        show_station(found.iloc[0])
    else:
        st.write('개별 충전소 마커를 클릭하거나 목록에서 충전소를 선택해 주세요.')

st.caption('경계 데이터: 통계청 SGIS · vuski/admdongkor (CC BY 4.0), 2026-10-01 기준. '
           '배경지도: CARTO. 지도에서 영역 밖의 임의 좌표 클릭은 지원하지 않습니다.')
