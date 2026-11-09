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
MAX_POINTS = 12000  # 브라우저 성능을 위해 한 번에 그릴 개별 점의 상한

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


def points_map(data, is_grid=False):
    """실제 충전소 또는 작은 격자별 집계점의 클릭 가능한 지도."""
    if is_grid:
        layer_id = 'grid'
        layer = pdk.Layer(
            'ScatterplotLayer', id=layer_id, data=data,
            get_position='[lng, lat]', get_radius='radius',
            get_fill_color='[40, 112, 203, 155]',
            pickable=True, auto_highlight=True,
            radius_min_pixels=4, radius_max_pixels=28,
        )
        tooltip = {'text': '구역별 충전소 {stations}곳 | 충전기 {chargers}대 | 사용 가능 {available}대'}
    else:
        layer_id = 'station'
        layer = pdk.Layer(
            'ScatterplotLayer', id=layer_id, data=data,
            get_position='[lng, lat]', get_radius=60,
            get_fill_color='color',
            pickable=True, auto_highlight=True,
            radius_min_pixels=4, radius_max_pixels=11,
        )
        tooltip = {'text': '{statNm}\n전체 {total_chargers}대 · 사용 가능 {available_chargers}대\n{addr}'}

    if not data.empty:
        latitude, longitude = float(data['lat'].mean()), float(data['lng'].mean())
    else:
        latitude, longitude = 36.25, 127.75
    zoom = 6 if is_grid else (9 if len(data) > 200 else 12)
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=latitude, longitude=longitude, zoom=zoom),
        map_provider='carto', map_style='light', tooltip=tooltip,
    )
    event = st.pydeck_chart(deck, on_select='rerun', selection_mode='single-object',
                            height=590, key='ev_grid' if is_grid else 'ev_stations')
    selected = event.selection.objects.get(layer_id, [])
    return selected[0] if selected else None


# ──────────────────── 페이지 UI ────────────────────
st.title('⚡ 전국 전기차 충전소 지도')
st.caption('충전소 위치 · 충전기 개수 · 규격/출력 · 이용 조건 | 상태 수치는 CSV 수집 시점 기준이며 실시간 보장은 아닙니다.')

csv_path = DATA_FILE if DATA_FILE.exists() else FALLBACK_FILE
if not csv_path.exists():
    st.error(f'같은 폴더에 {DATA_FILE.name} 파일을 넣어 주세요.')
    st.code('python -m streamlit run ev_charger_map_app.py')
    st.stop()
if csv_path == FALLBACK_FILE:
    st.info('정제된 39개 컬럼 CSV를 찾지 못해 기존 26개 컬럼 CSV로 표시 중입니다.')

try:
    df = load_data(str(csv_path), csv_path.stat().st_mtime)
except (ValueError, OSError) as exc:
    st.error(f'CSV 읽기 오류: {exc}')
    st.stop()

with st.sidebar:
    st.header('충전소 검색 · 필터')
    regions = ['전국'] + sorted(x for x in df['_region'].unique() if x != '지역 미상')
    if '지역 미상' in set(df['_region']):
        regions.append('지역 미상')
    region = st.selectbox('시·도 / 주소 첫 항목', regions)
    keyword = st.text_input('이름·주소 검색', placeholder='예: 서울역, 공영주차장, 강남구')
    charger_min = st.number_input('최소 충전기 대수', min_value=0, max_value=5000, value=0, step=1)
    available_only = st.checkbox('사용 가능 충전기 1대 이상')

    all_types = sorted({s for values in df['_types'] for s in values if s})
    selected_type = st.selectbox('충전 규격 코드', ['전체'] + all_types)
    all_outputs = sorted({s for values in df['_outputs'] for s in values if s},
                         key=lambda x: (not x.replace('.', '', 1).isdigit(), float(x) if x.replace('.', '', 1).isdigit() else 0))
    selected_output = st.selectbox('충전 출력(kW)', ['전체'] + all_outputs)
    st.caption('규격 코드는 원본 API의 chgerType 값입니다. 차량 호환성은 별도 확인이 필요합니다.')

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
    st.warning('조건에 맞는 충전소가 없습니다. 필터를 변경해 주세요.')
    st.stop()

mapped = filtered[filtered['_valid_location']].copy()
if len(mapped) < len(filtered):
    st.caption(f"좌표 미확인 또는 충전기별 좌표가 서로 달라 지도에서 제외된 곳: {len(filtered)-len(mapped):,}곳 (목록에는 유지)")

if 'focus_cell' not in st.session_state:
    st.session_state.focus_cell = None

mode = st.radio('지도 표시', ['전국 분포 (구역 합계)', '개별 충전소 위치'], horizontal=True, key='map_mode')

if mode.startswith('전국'):
    # 0.1도 격자 안의 여러 충전소를 합산: 전국 10만 개 마커로 인한 렌더링 부하를 줄인다.
    cells = mapped.assign(cell_lat=mapped['_lat'].round(1), cell_lng=mapped['_lng'].round(1))
    cells = cells.groupby(['cell_lat', 'cell_lng'], as_index=False).agg(
        lat=('_lat', 'mean'), lng=('_lng', 'mean'),
        stations=('statId', 'size'), chargers=('total_chargers', 'sum'),
        available=('available_chargers', 'sum'))
    cells['cell_id'] = cells['cell_lat'].astype(str) + '|' + cells['cell_lng'].astype(str)
    cells['radius'] = (900 + 75 * cells['chargers'].pow(0.5)).clip(upper=3500)
    selection = points_map(cells[['lat', 'lng', 'stations', 'chargers', 'available', 'radius', 'cell_id']], True)
    st.caption('원은 구역별 집계 위치이며 개별 충전소의 정확한 위치가 아닙니다. 구역을 클릭한 뒤 해당 구역의 개별 충전소를 볼 수 있습니다.')
    if selection:
        key = selection.get('cell_id', '')
        cell_stations = mapped[
            (mapped['_lat'].round(1).astype(str) + '|' + mapped['_lng'].round(1).astype(str)).eq(key)
        ]
        st.write(f"선택한 구역: 충전소 **{len(cell_stations):,}곳**, 충전기 **{int(cell_stations['total_chargers'].sum()):,}대**")
        if st.button('이 구역의 개별 충전소 확인',
                     on_click=lambda: st.session_state.update(
                         focus_cell=key, map_mode='개별 충전소 위치')):
            pass
        st.dataframe(cell_stations[['statNm', 'addr', 'total_chargers', 'available_chargers']].head(100),
                     use_container_width=True, hide_index=True)
else:
    if st.session_state.focus_cell:
        focus = st.session_state.focus_cell
        mapped = mapped[(mapped['_lat'].round(1).astype(str) + '|' + mapped['_lng'].round(1).astype(str)).eq(focus)]
        st.info(f'선택한 구역의 충전소 {len(mapped):,}곳을 표시합니다.')
        if st.button('구역 선택 해제'):
            st.session_state.focus_cell = None
            st.rerun()
    if len(mapped) > MAX_POINTS:
        st.warning(f'해당 조건에 충전소가 {len(mapped):,}곳 있습니다. 브라우저 부하를 막기 위해 {MAX_POINTS:,}곳을 넘으면 개별 점을 일부만 표시하지 않습니다. 지역·검색 조건을 좁혀 주세요.')
    elif mapped.empty:
        st.info('표시할 좌표가 없습니다.')
    else:
        # 지도에 불필요한 긴 JSON 컬럼은 전송하지 않음.
        plot = mapped[['statId', 'statNm', 'addr', '_lat', '_lng',
                       'total_chargers', 'available_chargers']].rename(
            columns={'_lat': 'lat', '_lng': 'lng'})
        plot = plot.copy()
        plot['color'] = plot['available_chargers'].map(
            lambda n: [28, 148, 110, 200] if n > 0 else [208, 88, 88, 195])
        selection = points_map(plot, False)
        if selection and selection.get('statId') in set(df['statId']):
            st.session_state['selected_station'] = selection['statId']
    st.caption('초록색은 수집 시점 사용 가능한 충전기가 1대 이상, 빨간색은 0대입니다. 최신 상태를 보장하지 않습니다.')

st.divider()
left, right = st.columns([1.2, 1], gap='large')
with left:
    st.subheader('충전소 목록')
    listing = filtered[['statId', 'statNm', 'addr', 'total_chargers',
                        'available_chargers', 'charging_chargers']].copy()
    st.dataframe(listing.head(300), hide_index=True, use_container_width=True)
    st.caption(f'목록은 첫 {min(len(listing),300):,}곳만 표시합니다. 전체 {len(listing):,}곳은 위 필터로 좁힐 수 있습니다.')
    choices = filtered[['statId', 'statNm']].head(300)
    choice_ids = choices['statId'].tolist()
    labels = dict(zip(choices['statId'], choices['statNm']))
    choice = st.selectbox('상세정보 조회', ['선택 안 함'] + choice_ids,
                          format_func=lambda x: f"{labels.get(x,'')} ({x})" if x != '선택 안 함' else x)
    if choice != '선택 안 함':
        st.session_state['selected_station'] = choice

with right:
    selected_id = st.session_state.get('selected_station')
    if selected_id and selected_id in set(filtered['statId']):
        station = filtered.loc[filtered['statId'].eq(selected_id)].iloc[0]
        show_station(station)
    else:
        st.subheader('충전소 상세정보')
        st.write('개별 충전소 지도에서 점을 클릭하거나 왼쪽 목록에서 충전소를 선택해 주세요.')

st.caption('지도 배경은 CARTO 제공입니다. 사용 환경의 네트워크에서 지도 타일이 차단되면 배경이 표시되지 않을 수 있습니다.')
