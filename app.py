import streamlit as st
import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="내 자산 시뮬레이터",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# 2. 데이터베이스 로딩
# ---------------------------------------------------------
@st.cache_data
def get_krx_dict():
    try:
        df_krx = fdr.StockListing('KRX')
        stock_dict = {}
        for index, row in df_krx.iterrows():
            name = row['Name']
            code = str(row['Code'])
            market = row['Market']
            
            if market == 'KOSPI': yf_code = code + ".KS"
            elif market == 'KOSDAQ': yf_code = code + ".KQ"
            else: yf_code = code + ".KS"
            
            # 검색 정확도를 위해 띄어쓰기 제거 버전도 저장
            stock_dict[name] = yf_code
            stock_dict[name.replace(" ", "").upper()] = yf_code
        return stock_dict
    except:
        return {}

krx_full_dict = get_krx_dict()

# ---------------------------------------------------------
# 3. 사이드바 (설정 메뉴)
# ---------------------------------------------------------
st.sidebar.header("⚙️ 시뮬레이션 설정")

def search_ticker(user_input):
    key = user_input.strip()
    key_upper = key.upper().replace(" ", "") # 대문자, 공백제거
    
    # [1] 가상 모델
    if "498400" in key_upper or "CC" == key_upper: return "CC", "KODEX 위클리CC(가상)"
    
    # [2] ★ 수동 매핑 추가 (검색 안 되는 것들 강제 연결) ★
    manual_map = {
        "S&P500": "SPY", "나스닥": "QQQ", "달러": "KRW=X",
        "애플": "AAPL", "테슬라": "TSLA", "엔비디아": "NVDA", "비트코인": "BTC-USD",
        # 골드선물 강제 추가
        "골드선물": "132030.KS", "KODEX골드선물": "132030.KS", "KODEX골드선물(H)": "132030.KS",
        "금": "132030.KS", "골드": "132030.KS"
    }
    
    # 입력값이 수동 맵에 있으면 바로 반환
    if key_upper in manual_map: return manual_map[key_upper], key
    # '골드선물'이 포함되어 있으면 강제 연결
    if "골드선물" in key_upper: return "132030.KS", "KODEX 골드선물(H)"

    # [3] 종목코드 6자리 직접 입력 시 (예: 132030)
    if key.isdigit() and len(key) == 6:
        return f"{key}.KS", f"종목코드 {key}"

    # [4] 한국 주식 찾기 (스마트 검색)
    # DB에서 정확히 일치하는 키 찾기
    if key_upper in krx_full_dict: return krx_full_dict[key_upper], key
    
    # 포함 검색 (입력한 단어가 종목명에 들어있는지)
    for name_key, code_val in krx_full_dict.items():
        if key_upper in name_key: 
            return code_val, name_key # 찾았다!
            
    return key_upper, key_upper

# 입력창
input_a_raw = st.sidebar.text_input("🔴 A팀 (예: TIGER 미국나스닥)", value="S&P500")
input_b_raw = st.sidebar.text_input("🔵 B팀 (예: 삼성전자)", value="골드선물")

code_a, name_a = search_ticker(input_a_raw)
code_b, name_b = search_ticker(input_b_raw)

st.sidebar.info(f"🔴 A: {name_a}\n\n🔵 B: {name_b}")
st.sidebar.markdown("---")
init_val = st.sidebar.number_input("💰 초기 투자금 (만원)", value=1000, step=100)
monthly_val = st.sidebar.number_input("📅 월 적립금 (만원)", value=200, step=50)
years = st.sidebar.slider("⏳ 조회 기간 (년)", 1, 30, 20)

# ---------------------------------------------------------
# 4. 데이터 수집
# ---------------------------------------------------------
@st.cache_data
def get_data_safe(t_a, t_b, yrs):
    end = datetime.now()
    start = end - timedelta(days=yrs*365 + 365)
    
    data = pd.DataFrame()
    try:
        # yfinance 다운로드 옵션 강화 (threads=False)
        spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True, threads=False)
        vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True, threads=False)
        krw = yf.download("KRW=X", start=start, end=end, progress=False, auto_adjust=True, threads=False)
        
        if spy.empty: return None, "시장 데이터(S&P500) 로드 실패. 잠시 후 다시 시도해주세요."
        
        # MultiIndex 처리
        if isinstance(spy.columns, pd.MultiIndex): spy = spy.xs('SPY', axis=1, level=1)
        if isinstance(vix.columns, pd.MultiIndex): vix = vix.xs('^VIX', axis=1, level=1)
        if isinstance(krw.columns, pd.MultiIndex): krw = krw.xs('KRW=X', axis=1, level=1)

        data['SP500'] = spy['Close']
        data['VIX'] = vix['Close'].reindex(data.index).ffill()
        data['USD_KRW'] = krw['Close'].reindex(data.index).ffill()
    except Exception as e: return None, f"시장 지표 오류: {e}"

    raw_kospi = None
    if t_a == "CC" or t_b == "CC":
        k = yf.download("^KS11", start=start, end=end, progress=False, auto_adjust=True, threads=False)
        if isinstance(k.columns, pd.MultiIndex): k = k.xs('^KS11', axis=1, level=1)
        raw_kospi = k['Close'].reindex(data.index).ffill()

    def get_asset(code, k_ref):
        if code == "CC":
            daily_prem = (1 + 0.12) ** (1/252) - 1
            ret = k_ref.pct_change().fillna(0)
            val = 10000 * (1 + ret.apply(lambda r: (0.005+daily_prem) if r > 0.005 else (r+daily_prem))).cumprod()
            return val
        else:
            # 야후 파이낸스 다운로드
            df = yf.download(code, start=start, end=end, progress=False, auto_adjust=True, threads=False)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(code, axis=1, level=1)
                except: df = df.droplevel(1, axis=1)
            return df['Close']

    s_a = get_asset(t_a, raw_kospi)
    s_b = get_asset(t_b, raw_kospi)

    if s_a is None: return None, f"'{t_a}' 데이터 없음 (종목코드를 확인하세요)"
    if s_b is None: return None, f"'{t_b}' 데이터 없음 (종목코드를 확인하세요)"

    start_a = s_a.first_valid_index()
    start_b = s_b.first_valid_index()
    if start_a is None or start_b is None: return None, "데이터 기간 오류 (데이터가 너무 짧습니다)"
    
    real_start = max(start_a, start_b)
    
    # 인덱스 정렬 및 병합
    data = data.loc[real_start:]
    data['ASSET_A'] = s_a.loc[real_start:].reindex(data.index).ffill()
    data['ASSET_B'] = s_b.loc[real_start:].reindex(data.index).ffill()
    
    return data.dropna(), "OK"

# ---------------------------------------------------------
# 5. 시뮬레이션 로직
# ---------------------------------------------------------
def run_simulation(df, asset_col, asset_name, init_krw, monthly_krw):
    is_krw = (".KS" in asset_name or ".KQ" in asset_name or "CC" in asset_name)
    start_rate = df['USD_KRW'].iloc[0]
    
    # 초기값 세팅
    if is_krw:
        dca_shares = init_krw / df[asset_col].iloc[0]
        bot_cash = init_krw
    else:
        dca_shares = (init_krw / start_rate) / df[asset_col].iloc[0]
        bot_cash = init_krw / start_rate
        
    bot_shares = 0
    total_invested = init_krw
    hist_dca = []
    hist_bot = []
    
    # 지표 계산
    df['MA125'] = df['SP500'].rolling(125).mean()
    df['Score_Mom'] = np.where(df['SP500'] > df['MA125'], 100, 0)
    df['MA50_VIX'] = df['VIX'].rolling(50).mean()
    df['Score_Vol'] = np.where(df['VIX'] < df['MA50_VIX'], 100, 0)
    
    delta = df['SP500'].diff(1)
    gain = delta.where(delta > 0, 0).ewm(com=13).mean()
    loss = -delta.where(delta < 0, 0).ewm(com=13).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['FNG'] = ((df['Score_Mom']*0.3) + (df['Score_Vol']*0.3) + (df['RSI']*0.4)).rolling(5).mean().clip(0, 100)
    
    prev_month = df.index[0].month
    
    for date, row in df.iterrows():
        price = row[asset_col]
        rate = row['USD_KRW']
        fng = row['FNG']
        
        # 월 적립 (매월 바뀌는 시점)
        if date.month != prev_month:
            total_invested += monthly_krw
            if is_krw:
                dca_shares += monthly_krw / price
                bot_cash += monthly_krw
            else:
                monthly_usd = monthly_krw / rate
                dca_shares += monthly_usd / price
                bot_cash += monthly_usd
            prev_month = date.month
            
        # 매매 로직 (공포탐욕지수 기반)
        if fng <= 20 and bot_cash > 0:
            shares = bot_cash / price
            bot_shares += shares
            bot_cash = 0
        elif fng >= 80 and bot_shares > 0:
            cash = bot_shares * price
            bot_cash += cash
            bot_shares = 0
            
        # 자산 가치 기록
        if is_krw:
            hist_dca.append(dca_shares * price)
            hist_bot.append((bot_shares * price) + bot_cash)
        else:
            hist_dca.append(dca_shares * price * rate)
            hist_bot.append(((bot_shares * price) + bot_cash) * rate)
            
    return total_invested, hist_dca, hist_bot

# ---------------------------------------------------------
# 6. 메인 화면
# ---------------------------------------------------------
st.markdown("### 📱 내 손안의 자산 시뮬레이터")
st.markdown("""
<style>
    .mobile-tip {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        font-size: 14px;
        margin-bottom: 20px;
    }
</style>
<div class='mobile-tip'>
    👈 <b>종목을 바꾸고 싶다면?</b><br>
    왼쪽 상단 <b>화살표(>)</b>를 눌러 메뉴를 열어보세요.
</div>
""", unsafe_allow_html=True)

with st.spinner('데이터 불러오는 중...'):
    data, status = get_data_safe(code_a, code_b, years)
    
    if data is None:
        st.error(f"⚠️ {status}")
        st.info(f"검색어 '{input_b_raw}'(이)가 정확하지 않을 수 있습니다. 종목코드 6자리(예: 132030)를 입력해보세요.")
    else:
        real_start = data.index[0]
        real_years = round((datetime.now() - real_start).days / 365, 1)
        start_str = real_start.strftime("%Y.%m.%d")
        
        if real_years < (years - 1):
            st.warning(f"⚠️ **기간 알림:** 상장일({start_str})이 늦어 **{real_years}년**치만 분석했습니다.")
        else:
            st.success(f"📅 분석 기간: {start_str} ~ 현재 ({real_years}년)")
            
        ik = init_val * 10000
        mk = monthly_val * 10000
        
        inv_a, dca_a, bot_a = run_simulation(data, 'ASSET_A', name_a, ik, mk)
        inv_b, dca_b, bot_b = run_simulation(data, 'ASSET_B', name_b, ik, mk)
        
        st.markdown(f"#### 📊 최종 평가 금액")
        col1, col2 = st.columns(2)
        
        def show(label, final, base):
            p = final - base
            r = (p/base)*100
            return f"**{label}**", f"{int(final/10000):,}만원", f"{r:.1f}%"

        with col1:
            st.markdown(f"##### 🔴 {name_a}")
            l, v, d = show("존버", dca_a[-1], inv_a)
            st.metric(l, v, d)
            l, v, d = show("AI매매", bot_a[-1], inv_a)
            st.metric(l, v, d)
            
        with col2:
            st.markdown(f"##### 🔵 {name_b}")
            l, v, d = show("존버", dca_b[-1], inv_b)
            st.metric(l, v, d)
            l, v, d = show("AI매매", bot_b[-1], inv_b)
            st.metric(l, v, d)
            
        st.markdown("---")
        st.markdown("#### 📈 자산 성장 그래프 (터치 가능)")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=dca_a, mode='lines', name=f'{name_a} (존버)', line=dict(color='#FF4B4B', width=2)))
        fig.add_trace(go.Scatter(x=data.index, y=bot_a, mode='lines', name=f'{name_a} (AI)', line=dict(color='#FF4B4B', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=data.index, y=dca_b, mode='lines', name=f'{name_b} (존버)', line=dict(color='#1C83E1', width=2)))
        fig.add_trace(go.Scatter(x=data.index, y=bot_b, mode='lines', name=f'{name_b} (AI)', line=dict(color='#1C83E1', width=2, dash='dot')))
        
        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            yaxis_tickformat=',',
        )
        st.plotly_chart(fig, use_container_width=True)
