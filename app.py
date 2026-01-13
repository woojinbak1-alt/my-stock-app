import streamlit as st
import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr
import koreanize_matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. 페이지 설정 (사이드바 열림 고정)
# ---------------------------------------------------------
st.set_page_config(
    page_title="전지적 시점 자산 시뮬레이터",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. 한국 주식 전종목 가져오기 (캐싱으로 속도 UP)
# ---------------------------------------------------------
@st.cache_data
def get_krx_list():
    try:
        # 코스피, 코스닥, 코넥스 전종목 불러오기
        df_krx = fdr.StockListing('KRX')
        
        # 이름과 코드를 딕셔너리로 변환 (예: "삼성전자": "005930")
        # 검색 편의를 위해 공백 제거한 이름도 매핑
        stock_dict = {}
        for index, row in df_krx.iterrows():
            name = row['Name']
            code = row['Code']
            market = row['Market']
            
            # 야후 파이낸스용 접미사 붙이기
            if market == 'KOSPI':
                yf_code = code + ".KS"
            elif market == 'KOSDAQ':
                yf_code = code + ".KQ"
            else:
                yf_code = code + ".KS" # 기본값
            
            stock_dict[name] = yf_code
            stock_dict[name.replace(" ", "")] = yf_code # 띄어쓰기 없는 버전도 추가
            
        return stock_dict
    except:
        return {}

# 전종목 리스트 로딩
krx_dict = get_krx_list()

# ---------------------------------------------------------
# 3. 사이드바 UI
# ---------------------------------------------------------
st.sidebar.header("📊 시뮬레이션 설정")
st.sidebar.markdown("---")

# 검색 도움말
with st.sidebar.expander("💡 종목 검색 팁 (필독)"):
    st.markdown("""
    - **한국 주식:** 한글 이름 입력 (예: 삼성전자, 에코프로)
    - **미국 주식:** 티커 입력 권장 (예: AAPL, TSLA, SPY)
    - **가상 모델:** '498400' 또는 '커버드콜' 입력
    """)

# 입력창
input_a = st.sidebar.text_input("🔴 A팀 (빨강) 종목명/티커", value="S&P500")
input_b = st.sidebar.text_input("🔵 B팀 (파랑) 종목명/티커", value="삼성전자")

st.sidebar.markdown("---")
init_val = st.sidebar.number_input("💰 초기 투자금 (만원)", value=1000, step=100)
monthly_val = st.sidebar.number_input("📅 월 적립금 (만원)", value=200, step=50)
years = st.sidebar.slider("⏳ 투자 기간 (년)", 1, 30, 10)

run_btn = st.sidebar.button("🚀 시뮬레이션 시작", type="primary")

# ---------------------------------------------------------
# 4. 티커 변환 엔진 (핵심)
# ---------------------------------------------------------
def find_ticker(user_input):
    key = user_input.strip()
    key_no_space = key.replace(" ", "").upper()
    
    # [1] 가상 모델 (498400)
    if "498400" in key_no_space or "커버드콜" in key_no_space or "CC" == key_no_space:
        return "CC"
    
    # [2] 미국/지수 등 자주 쓰는 티커 수동 매핑
    manual_map = {
        "S&P500": "SPY", "나스닥": "QQQ", "나스닥100": "QQQ",
        "비트코인": "BTC-USD", "이더리움": "ETH-USD",
        "달러": "KRW=X", "애플": "AAPL", "테슬라": "TSLA",
        "엔비디아": "NVDA", "마소": "MSFT", "구글": "GOOGL"
    }
    if key_no_space in manual_map:
        return manual_map[key_no_space]
        
    # [3] 한국 주식 자동 검색 (전종목 DB 조회)
    # 입력한 이름이 DB에 있는지 확인
    if key in krx_dict:
        return krx_dict[key]
    if key_no_space in krx_dict: # 띄어쓰기 무시 검색
        return krx_dict[key_no_space]
        
    # [4] 그래도 없으면 입력한 그대로 반환 (미국 티커라고 가정)
    return key_no_space

# ---------------------------------------------------------
# 5. 데이터 처리 및 시각화
# ---------------------------------------------------------
@st.cache_data
def get_data(ticker_a, ticker_b, years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365 + 365)
    
    # 기본 다운로드 목록
    tickers = ["^GSPC", "^VIX", "KRW=X"]
    
    # A 종목
    if ticker_a == "CC": tickers.append("^KS11")
    elif ticker_a not in tickers: tickers.append(ticker_a)
    
    # B 종목
    if ticker_b == "CC": 
        if "^KS11" not in tickers: tickers.append("^KS11")
    elif ticker_b not in tickers: tickers.append(ticker_b)
    
    try:
        df = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False, auto_adjust=True)
    except Exception as e:
        return None, str(e)
    
    if df.empty: return None, "데이터 없음"
    
    data = pd.DataFrame()
    try:
        # 공통 지표 처리 (멀티인덱스 예외처리)
        if isinstance(df.columns, pd.MultiIndex):
            data['SP500'] = df['^GSPC']['Close'].ffill()
            data['VIX'] = df['^VIX']['Close'].ffill()
            data['USD_KRW'] = df['KRW=X']['Close'].ffill()
            raw_kospi = df['^KS11']['Close'].ffill() if "^KS11" in tickers else None
            
            # A 데이터 추출
            if ticker_a == "CC":
                daily_prem = (1 + 0.12) ** (1/252) - 1
                ret = raw_kospi.pct_change().fillna(0)
                data['ASSET_A'] = 10000 * (1 + ret.apply(lambda r: (0.005 + daily_prem) if r > 0.005 else (r + daily_prem))).cumprod()
            else:
                if ticker_a in df: data['ASSET_A'] = df[ticker_a]['Close'].ffill()
                else: return None, f"'{ticker_a}' 데이터를 찾을 수 없습니다."

            # B 데이터 추출
            if ticker_b == "CC":
                daily_prem = (1 + 0.12) ** (1/252) - 1
                ret = raw_kospi.pct_change().fillna(0)
                data['ASSET_B'] = 10000 * (1 + ret.apply(lambda r: (0.005 + daily_prem) if r > 0.005 else (r + daily_prem))).cumprod()
            else:
                if ticker_b in df: data['ASSET_B'] = df[ticker_b]['Close'].ffill()
                else: return None, f"'{ticker_b}' 데이터를 찾을 수 없습니다."
        else:
            return None, "데이터 구조 오류 (잠시 후 다시 시도)"
            
    except Exception as e:
        return None, f"데이터 처리 오류: {str(e)}"
        
    return data.dropna(), "OK"

# 시뮬레이션 계산 함수
def run_simulation(df, asset_col, asset_name, init_krw, monthly_krw):
    # 한국 주식 판별 (환율 적용 X)
    is_krw_asset = False
    # 티커에 .KS, .KQ가 있거나 한글 이름이면 원화 자산으로 간주
    if ".KS" in asset_name or ".KQ" in asset_name or "CC" in asset_name:
        is_krw_asset = True
    
    start_rate = df['USD_KRW'].iloc[0]
    
    if is_krw_asset:
        dca_shares = init_krw / df[asset_col].iloc[0]
        bot_cash = init_krw
    else:
        dca_shares = (init_krw / start_rate) / df[asset_col].iloc[0]
        bot_cash = init_krw / start_rate
        
    bot_shares = 0
    total_invested = init_krw
    
    hist_dca = []
    hist_bot = []
    
    prev_month = df.index[0].month
    
    # 봇 전략 (FNG)
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
    
    for date, row in df.iterrows():
        price = row[asset_col]
        rate = row['USD_KRW']
        fng = row['FNG']
        
        # 월 적립
        if date.month != prev_month:
            total_invested += monthly_krw
            if is_krw_asset:
                dca_shares += monthly_krw / price
                bot_cash += monthly_krw
            else:
                monthly_usd = monthly_krw / rate
                dca_shares += monthly_usd / price
                bot_cash += monthly_usd
            prev_month = date.month
            
        # 봇 매매
        if fng <= 20 and bot_cash > 0: # 공포 매수
            shares = bot_cash / price
            bot_shares += shares
            bot_cash = 0
        elif fng >= 80 and bot_shares > 0: # 탐욕 매도
            cash = bot_shares * price
            bot_cash += cash
            bot_shares = 0
            
        # 평가금 합산
        if is_krw_asset:
            val_dca = dca_shares * price
            val_bot = (bot_shares * price) + bot_cash
        else:
            val_dca = dca_shares * price * rate
            val_bot = ((bot_shares * price) + bot_cash) * rate
            
        hist_dca.append(val_dca)
        hist_bot.append(val_bot)
        
    return total_invested, hist_dca, hist_bot, is_krw_asset

# ---------------------------------------------------------
# 메인 실행 화면
# ---------------------------------------------------------
st.title("🥊 [세기의 대결] 자산 시뮬레이터")
st.markdown("##### S&P500 vs 내 종목, 적립식 vs AI매매 승자는?")

if run_btn:
    with st.spinner('데이터를 분석 중입니다... (최초 실행 시 10초 소요)'):
        t_a = find_ticker(input_a)
        t_b = find_ticker(input_b)
        
        data, status = get_data(t_a, t_b, years)
        
        if data is None:
            st.error(f"⚠️ 오류 발생: {status}")
            st.info("💡 팁: 정확한 한글 종목명(예: 현대차) 또는 티커(예: TSLA)를 확인해주세요.")
        else:
            # 금액 단위 변환
            init_k = init_val * 10000
            month_k = monthly_val * 10000
            
            # 시뮬레이션 실행
            inv_a, dca_a, bot_a, is_krw_a = run_simulation(data, 'ASSET_A', t_a, init_k, month_k)
            inv_b, dca_b, bot_b, is_krw_b = run_simulation(data, 'ASSET_B', t_b, init_k, month_k)
            
            # 결과 표시 UI
            st.success(f"✅ 분석 완료! ({input_a} vs {input_b})")
            
            # 최종 수익률 카드
            col1, col2 = st.columns(2)
            
            def make_metric(label, final_val, invested):
                profit = final_val - invested
                rate = (profit / invested) * 100
                return label, f"{int(final_val/10000):,}만원", f"{rate:.1f}%"
            
            with col1:
                st.subheader(f"🔴 {input_a}")
                l, v, d = make_metric("존버(Buy&Hold)", dca_a[-1], inv_a)
                st.metric(l, v, d)
                l, v, d = make_metric("AI 봇 매매", bot_a[-1], inv_a)
                st.metric(l, v, d)
                
            with col2:
                st.subheader(f"🔵 {input_b}")
                l, v, d = make_metric("존버(Buy&Hold)", dca_b[-1], inv_b)
                st.metric(l, v, d)
                l, v, d = make_metric("AI 봇 매매", bot_b[-1], inv_b)
                st.metric(l, v, d)

            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # A팀
            ax.plot(data.index, dca_a, color='#FF4B4B', linestyle='-', linewidth=2, label=f'{input_a} (존버)')
            ax.plot(data.index, bot_a, color='#FF4B4B', linestyle='--', linewidth=1, alpha=0.7, label=f'{input_a} (AI)')
            
            # B팀
            ax.plot(data.index, dca_b, color='#1C83E1', linestyle='-', linewidth=2, label=f'{input_b} (존버)')
            ax.plot(data.index, bot_b, color='#1C83E1', linestyle='--', linewidth=1, alpha=0.7, label=f'{input_b} (AI)')
            
            ax.set_title(f"자산 성장 추이 (총 투자원금: {int(inv_a/10000):,}만원)", fontsize=12)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
            
            st.pyplot(fig)

else:
    st.info("👈 왼쪽 메뉴에서 종목과 금액을 입력하고 '시뮬레이션 시작'을 눌러주세요!")
