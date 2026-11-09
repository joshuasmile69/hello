"""Streamlit EV charging station map | input: ev_charger_navigation_clean.csv.

Run from the folder containing the CSV: python -m streamlit run ev_charger_map_app.py
"""

import json
import math
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'ev_charger_navigation_clean.csv'
# 개발 중에는 구버전 26컬럼 파일도 시험할 수 있다.
FALLBACK_FILE = BASE_DIR / 'ev_charger_navigation.csv'
DAYS = [('Mon', '월'), ('Tue', '화'), ('Wed', '수'), ('Thu', '목'),
        ('Fri', '금'), ('Sat', '토'), ('Sun', '일')]
MAX_DETAIL_POINTS = 450  # 이 수 이하일 때만 충전소 실제 점을 표시

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



# ──────────────── 행정구역·공간별 집계 유틸리티 ────────────────
def admin_area(address, zscode):
    """주소에서 시/군/구 이름을 추출한다. 명칭이 없으면 지역코드로 대체."""
    parts = str(address).strip().split()
    if len(parts) < 2:
        return '지역코드 ' + (str(zscode).strip() or '미상')
    # 예: 경기도 수원시 영통구 → 수원시 영통구
    if len(parts) >= 3 and parts[1].endswith('시') and parts[2].endswith('구'):
        return parts[1] + ' ' + parts[2]
    return parts[1]


def prepare_admin(data):
    data = data.copy()
    data['_sido_name'] = data['_region']
    # 지역코드를 우선 이용해 같은 시/도의 이름 표기 차이를 묶는다.
    data['_sido_key'] = data.get('zcode', pd.Series('', index=data.index)).astype(str).str.strip()
    data.loc[data['_sido_key'].eq(''), '_sido_key'] = data['_sido_name']
    data['_sgg_name'] = [admin_area(a, z) for a, z in zip(
        data['addr'], data.get('zscode', pd.Series('', index=data.index)))]
    data['_sgg_key'] = data['_sido_key'] + '::' + data['_sgg_name']
    return data


def group_counts(data, key, label_col):
    """실제 충전소 약 10만 건 → 시·도/시·군·구별 집계 원 수십 개."""
    grouped = data.groupby(key, as_index=False).agg(
        lat=('_lat', 'mean'), lng=('_lng', 'mean'),
        stations=('statId', 'size'),
        chargers=('total_chargers', 'sum'),
        available=('available_chargers', 'sum'),
        label=(label_col, lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
    )
    grouped = grouped.rename(columns={key: 'group_key'})
    return decorate_groups(grouped)


def decorate_groups(groups):
    groups = groups.copy()
    # 원 반지름은 실제 지리 범위가 아니라 표시용 픽셀 크기다.
    groups['radius'] = (13 + 3 * groups['stations'].pow(0.25)).clip(15, 36)
    groups['count_text'] = groups['stations'].map(lambda x: f'{x:,}')
    return groups


def grid_split(data, side=7):
    """선택한 시·군·구 내부를 작은 구역들로 나누되 모든 개별 충전소를 지도에 보내지 않는다.

    구역의 지리적 범위를 균등하게 분할한다. 이 함수의 cell_key는 현재 data 안에서만 유효.
    """
    data = data.copy()
    lat_span = float(data['_lat'].max() - data['_lat'].min())
    lng_span = float(data['_lng'].max() - data['_lng'].min())
    lat_step = max(lat_span / side, 1e-7)
    lng_step = max(lng_span / side, 1e-7)
    cell_lat = ((data['_lat'] - data['_lat'].min()) / lat_step).astype(int).clip(0, side - 1)
    cell_lng = ((data['_lng'] - data['_lng'].min()) / lng_step).astype(int).clip(0, side - 1)
    data['_cell_key'] = cell_lat.astype(str) + ':' + cell_lng.astype(str)
    return data


def spatial_groups(data):
    cells = grid_split(data)
    grouped = cells.groupby('_cell_key', as_index=False).agg(
        lat=('_lat', 'mean'), lng=('_lng', 'mean'),
        stations=('statId', 'size'), chargers=('total_chargers', 'sum'),
        available=('available_chargers', 'sum')
    ).rename(columns={'_cell_key': 'group_key'})
    grouped['label'] = '선택 구역'
    return decorate_groups(grouped)


def center_zoom(data, level):
    if data.empty:
        return 36.3, 127.7, 6
    lat, lng = float(data['_lat'].mean()), float(data['_lng'].mean())
    width = max(float(data['_lat'].max() - data['_lat'].min()),
                float(data['_lng'].max() - data['_lng'].min()))
    # 지리적 범위에 따라 화면을 확대. 전국 모드는 의도적으로 넓게.
    if level == 'sido':
        return lat, lng, 6
    zoom = max(6, min(15, 7.5 - math.log2(max(width, 0.001))))
    return lat, lng, zoom


def draw_map(data, level, serial):
    """집계 단계는 원과 숫자; 상세 단계는 실제 충전소 마커만 전송."""
    if level == 'station':
        plot = data[['statId', 'statNm', 'addr', '_lat', '_lng',
                     'total_chargers', 'available_chargers']].rename(
                         columns={'_lat': 'lat', '_lng': 'lng'}).copy()
        plot['color'] = plot['available_chargers'].map(
            lambda n: [28, 148, 110, 215] if n > 0 else [208, 88, 88, 215])
        layers = [pdk.Layer(
            'ScatterplotLayer', data=plot, id='stations',
            get_position='[lng,lat]', get_radius=8, radius_units='pixels',
            get_fill_color='color', pickable=True, auto_highlight=True,
            radius_min_pixels=5, radius_max_pixels=12
        )]
        tooltip = {'text': '{statNm}\n충전기 {total_chargers}대 / 사용 가능 {available_chargers}대\n{addr}'}
        center = plot.rename(columns={'lat': '_lat', 'lng': '_lng'})
    else:
        plot = data[['group_key', 'label', 'lat', 'lng', 'stations', 'chargers',
                     'available', 'radius', 'count_text']].copy()
        layers = [
            pdk.Layer('ScatterplotLayer', data=plot, id='clusters',
                      get_position='[lng,lat]', get_radius='radius',
                      radius_units='pixels', get_fill_color='[36,111,203,195]',
                      pickable=True, auto_highlight=True,
                      radius_min_pixels=15, radius_max_pixels=36),
            pdk.Layer('TextLayer', data=plot, id='cluster_labels',
                      get_position='[lng,lat]', get_text='count_text',
                      get_size=13, get_color='[255,255,255,255]',
                      get_text_anchor='"middle"', get_alignment_baseline='"center"',
                      pickable=False)
        ]
        tooltip = {'text': '{label}\n충전소 {stations}곳 | 충전기 {chargers}대 | 사용 가능 {available}대\n클릭하면 상세 구역으로 이동'}
        center = plot.rename(columns={'lat': '_lat', 'lng': '_lng'})

    lat, lng, zoom = center_zoom(center, level)
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lng, zoom=zoom),
        map_provider='carto', map_style='light', tooltip=tooltip,
    )
    # 단계/선택이 바뀌면 key도 바뀌므로 이전 클릭 이벤트가 재실행되지 않는다.
    event = st.pydeck_chart(deck, on_select='rerun', selection_mode='single-object',
                            key=f'ev_drill_map_{serial}', height=610)
    layer_id = 'stations' if level == 'station' else 'clusters'
    selected = event.selection.objects.get(layer_id, [])
    return selected[0] if selected else None


def go_back():
    if st.session_state['cells_path']:
        st.session_state['cells_path'] = st.session_state['cells_path'][:-1]
    elif st.session_state['picked_sgg'] is not None:
        st.session_state['picked_sgg'] = None
    elif st.session_state['picked_sido'] is not None:
        st.session_state['picked_sido'] = None
    st.session_state['map_revision'] += 1


# ────────────────────────────── 페이지 UI ──────────────────────────────
st.title('⚡ 전국 전기차 충전소 지도')
st.caption('시·도 → 시·군·구 → 구역 → 개별 충전소 순으로 클릭해 확대합니다. 사용 가능 대수는 CSV 수집 시점 기준입니다.')

csv_path = DATA_FILE if DATA_FILE.exists() else FALLBACK_FILE
if not csv_path.exists():
    st.error(f'{DATA_FILE.name}을(를) 이 Python 파일과 같은 폴더에 넣어 주세요.')
    st.stop()
if csv_path == FALLBACK_FILE:
    st.info('39개 컬럼 정제 CSV가 없어 기존 26개 컬럼 CSV를 사용하고 있습니다.')

try:
    df = prepare_admin(load_data(str(csv_path), csv_path.stat().st_mtime))
except (ValueError, OSError) as exc:
    st.error(f'CSV 읽기 오류: {exc}')
    st.stop()

with st.sidebar:
    st.header('충전소 검색 · 필터')
    regions = ['전국'] + sorted(x for x in df['_region'].unique() if x != '지역 미상')
    if '지역 미상' in set(df['_region']):
        regions.append('지역 미상')
    region = st.selectbox('시·도', regions)
    keyword = st.text_input('이름·주소 검색', placeholder='예: 서울역, 공영주차장, 강남구')
    charger_min = st.number_input('최소 충전기 대수', min_value=0, max_value=5000, value=0, step=1)
    available_only = st.checkbox('사용 가능 충전기 1대 이상')
    all_types = sorted({s for values in df['_types'] for s in values if s})
    selected_type = st.selectbox('충전 규격 코드', ['전체'] + all_types)
    all_outputs = sorted({s for values in df['_outputs'] for s in values if s},
                         key=lambda x: (not x.replace('.', '', 1).isdigit(),
                                        float(x) if x.replace('.', '', 1).isdigit() else 0))
    selected_output = st.selectbox('충전 출력(kW)', ['전체'] + all_outputs)
    st.caption('상태 수치는 수집 시점 기준이며 실시간 정보가 아닙니다.')

filtered = df
if region != '전국':
    filtered = filtered[filtered['_region'].eq(region)]
if keyword.strip():
    mask = (filtered['statNm'].str.contains(keyword.strip(), case=False, regex=False, na=False)
            | filtered['addr'].str.contains(keyword.strip(), case=False, regex=False, na=False))
    filtered = filtered[mask]
if charger_min:
    filtered = filtered[filtered['total_chargers'].ge(charger_min)]
if available_only:
    filtered = filtered[filtered['available_chargers'].gt(0)]
if selected_type != '전체' or selected_output != '전체':
    filtered = filtered[filtered['_groups'].map(
        lambda groups: any(
            (selected_type == '전체' or str(x.get('chgerType', '')).strip() == selected_type)
            and (selected_output == '전체' or str(x.get('output', '')).strip() == selected_output)
            for x in groups
        )
    )]

c1, c2, c3, c4 = st.columns(4)
c1.metric('검색된 충전소', f'{len(filtered):,}곳')
c2.metric('설치된 충전기', f"{int(filtered['total_chargers'].sum()):,}대")
c3.metric('사용 가능', f"{int(filtered['available_chargers'].sum()):,}대")
c4.metric('좌표 확인', f"{int(filtered['_valid_location'].sum()):,}곳")

if filtered.empty:
    st.warning('조건에 맞는 충전소가 없습니다.')
    st.stop()

mapped = filtered[filtered['_valid_location']].copy()
if mapped.empty:
    st.warning('위치를 표시할 수 있는 충전소가 없습니다.')
    st.stop()

for key, default in [('picked_sido', None), ('picked_sgg', None),
                     ('cells_path', []), ('map_revision', 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# 다른 필터가 적용되어 선택해 둔 시·도나 시·군·구가 사라진 경우 경로 초기화.
if st.session_state['picked_sido'] is not None and not mapped['_sido_key'].eq(st.session_state['picked_sido']).any():
    st.session_state['picked_sido'] = None
    st.session_state['picked_sgg'] = None
    st.session_state['cells_path'] = []
if st.session_state['picked_sgg'] is not None and not mapped['_sgg_key'].eq(st.session_state['picked_sgg']).any():
    st.session_state['picked_sgg'] = None
    st.session_state['cells_path'] = []

# 현재 지도 범위 선택. 원본 데이터는 캐시되어 있고 화면에는 해당 단계만 보낸다.
current = mapped
sido = st.session_state['picked_sido']
sgg = st.session_state['picked_sgg']
if sido is not None:
    current = current[current['_sido_key'].eq(sido)]
if sgg is not None:
    current = current[current['_sgg_key'].eq(sgg)]

# 같은 시·군·구 안에서 선택한 작은 구역을 순서대로 좁힌다.
for cell in st.session_state['cells_path']:
    current = grid_split(current)
    current = current[current['_cell_key'].eq(cell)].drop(columns=['_cell_key'])

if sido is None:
    level = 'sido'
    groups = group_counts(current, '_sido_key', '_sido_name')
    title = '전국 · 시·도별 충전소'
elif sgg is None:
    level = 'sgg'
    groups = group_counts(current, '_sgg_key', '_sgg_name')
    title = '선택한 시·도 · 시·군·구별 충전소'
elif len(current) > MAX_DETAIL_POINTS:
    level = 'cells'
    groups = spatial_groups(current)
    title = f'선택한 시·군·구 · 구역별 집계 ({len(current):,}곳)'
else:
    level = 'station'
    groups = None
    title = f'개별 충전소 ({len(current):,}곳)'

st.subheader(title)
nav_left, nav_right = st.columns([1, 3])
with nav_left:
    if sido is not None:
        if st.button('← 이전 단계로', use_container_width=True):
            go_back()
            st.rerun()
with nav_right:
    if sido is not None:
        if st.button('전국 지도로 돌아가기'):
            st.session_state['picked_sido'] = None
            st.session_state['picked_sgg'] = None
            st.session_state['cells_path'] = []
            st.session_state['map_revision'] += 1
            st.rerun()

if level == 'cells' and len(groups) == 1 and len(current) > MAX_DETAIL_POINTS:
    # 같은 좌표에 매우 많은 충전소가 있는 경우는 무한 반복으로 확대하지 않는다.
    st.info('이 구역은 좌표가 지나치게 겹칩니다. 아래 목록에서 충전소를 선택해 주세요.')
    # 개별 450개를 넘겨 브라우저에 마커를 보내지 않는다.
    selected = None
else:
    plot_data = current if level == 'station' else groups
    selected = draw_map(plot_data, level, st.session_state['map_revision'])

if selected:
    if level == 'sido':
        st.session_state['picked_sido'] = selected['group_key']
        st.session_state['picked_sgg'] = None
        st.session_state['cells_path'] = []
        st.session_state['map_revision'] += 1
        st.rerun()
    elif level == 'sgg':
        st.session_state['picked_sgg'] = selected['group_key']
        st.session_state['cells_path'] = []
        st.session_state['map_revision'] += 1
        st.rerun()
    elif level == 'cells':
        st.session_state['cells_path'] = st.session_state['cells_path'] + [selected['group_key']]
        st.session_state['map_revision'] += 1
        st.rerun()
    else:
        st.session_state['selected_station'] = selected.get('statId')

st.caption('큰 원은 집계값이며, 개별 충전소가 표시되는 단계에서만 실제 설치 좌표를 보여줍니다.')

st.divider()
left, right = st.columns([1.1, 1], gap='large')
with left:
    st.subheader('현재 지도 범위의 충전소')
    st.dataframe(current[['statNm', 'addr', 'total_chargers', 'available_chargers']].head(300),
                 hide_index=True, use_container_width=True)
    st.caption(f'이 지역의 충전소 {len(current):,}곳 중 최대 300곳만 목록에 표시합니다.')
    choices = current[['statId', 'statNm']].head(300)
    labels = dict(zip(choices['statId'], choices['statNm']))
    choice = st.selectbox('상세정보 조회', ['선택 안 함'] + choices['statId'].tolist(),
                          format_func=lambda x: f"{labels.get(x, '')} ({x})" if x != '선택 안 함' else x)
    if choice != '선택 안 함':
        st.session_state['selected_station'] = choice

with right:
    st.subheader('충전소 상세정보')
    selected_id = st.session_state.get('selected_station')
    if selected_id and current['statId'].eq(selected_id).any():
        show_station(current.loc[current['statId'].eq(selected_id)].iloc[0])
    else:
        st.write('지도의 개별 충전소 점을 클릭하거나 목록에서 충전소를 선택해 주세요.')

st.caption('지도는 CARTO 타일을 사용합니다. 포트 문제 발생 시 --server.port 8765로 실행할 수 있습니다.')
