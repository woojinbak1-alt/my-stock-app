import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import koreanize_matplotlib # 한글 폰트 깨짐 방지

# 페이지 설정
st.set_page_config(page_title="자산 시뮬레이터", layout="wide")

st.title("🥊 [세기의 대결] 자산 시뮬레이터")
st.markdown("티스토리 방문자를 위한 **적립식 vs AI매매** 수익률 비교 계산기입니다.")

# =========================================================
# 사이드바: 사용자 입력
# =========================================================
st.sidebar.header("설정 입력")

# 종목명 입력
input_a = st.sidebar.text_input("🔴 A팀 (빨강) 종목명", value="TIGER미국나스닥100")
input_b = st.sidebar.text_input("🔵 B팀 (파랑) 종목명", value="현대자동차")

# 금액 및 기간 설정
init_val = st.sidebar.number_input("초기 투자금 (만원)", value=0, step=100)
monthly_val = st.sidebar.number_input("월 적립금 (만원)", value=300, step=50)
years = st.sidebar.slider("투자 기간 (년)", min_value=1, max_value=30, value=5)

run_btn = st.sidebar.button("시뮬레이션 시작")

# =========================================================
# 함수 정의
# =========================================================
def find_ticker(user_input):
    key = user_input.strip().upper().replace(" ", "")
    if "498400" in key or "위클리" in key or "커버드콜" in key or "CC" in key: return "CC"
    stock_map = {
        "TIGER미국나스닥100": "133690.KS", "삼성전자": "005930.KS", "현대차": "005380.KS", 
        "현대자동차": "005380.KS", "SK하이닉스": "000660.KS", "S&P500": "SPY", 
        "나스닥": "QQQ", "애플": "AAPL", "테슬라": "TSLA", "비트코인": "BTC-USD",
        "POSCO홀딩스": "005490.KS", "카카오": "035720.KS", "네이버": "035420.KS"
    }
    if key in stock_map: return stock_map[key]
    if key.isdigit() and len(key) == 6: return f"{key}.KS"
    return key

@st.cache_data # 데이터 캐싱 (속도 향상)
def get_data(ticker_a, ticker_b, years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365 + 365)
    
    tickers = ["^GSPC", "^VIX", "KRW=X"]
    if ticker_a == "CC": tickers.append("^KS11")
    elif ticker_a not in tickers: tickers.append(ticker_a)
    
    if ticker_b == "CC": 
        if "^KS11" not in tickers: tickers.append("^KS11")
    elif ticker_b not in tickers: tickers.append(ticker_b)
    
    try:
        df = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False, auto_adjust=True)
    except: return None
    
    if df.empty: return None
    
    data = pd.DataFrame()
    try:
        if isinstance(df.columns, pd.MultiIndex):
            data['SP500'] = df['^GSPC']['Close'].ffill()
            data['VIX'] = df['^VIX']['Close'].ffill()
            data['USD_KRW'] = df['KRW=X']['Close'].ffill()
            raw_kospi = df['^KS11']['Close'].ffill() if "^KS11" in tickers else None
            
            # A 데이터
            if ticker_a == "CC":
                daily_premium = (1 + 0.12) ** (1/252) - 1
                ret = raw_kospi.pct_change().fillna(0)
                data['ASSET_A'] = 10000 * (1 + ret.apply(lambda r: (0.005 + daily_premium) if r > 0.005 else (r + daily_premium))).cumprod()
            else:
                data['ASSET_A'] = df[ticker_a]['Close'].ffill()
                
            # B 데이터
            if ticker_b == "CC":
                daily_premium = (1 + 0.12) ** (1/252) - 1
                ret = raw_kospi.pct_change().fillna(0)
                data['ASSET_B'] = 10000 * (1 + ret.apply(lambda r: (0.005 + daily_premium) if r > 0.005 else (r + daily_premium))).cumprod()
            else:
                data['ASSET_B'] = df[ticker_b]['Close'].ffill()
                
    except: return None
    
    return data.dropna()

def run_sim(df, asset_col, asset_name, init_krw, monthly_krw):
    is_krw = any(x in asset_name for x in ["TIGER", "KODEX", "삼성", "현대", "CC", ".KS"]) or ".KS" in asset_name
    start_rate = df['USD_KRW'].iloc[0]
    
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
    prev_month = df.index[0].month
    
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
    df['FNG'] = ((df['Score_Mom']*0.3) + (df['Score_Vol']*0.3) + (df['RSI']*0.4)).rolling(5).mean()

    for date, row in df.iterrows():
        price = row[asset_col]
        rate = row['USD_KRW']
        fng = row['FNG']
        
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
            
        if fng <= 20 and bot_cash > 0:
            shares = bot_cash / price
            bot_shares += shares
            bot_cash = 0
        elif fng >= 80 and bot_shares > 0:
            cash = bot_shares * price
            bot_cash += cash
            bot_shares = 0
            
        if is_krw:
            hist_dca.append(dca_shares * price)
            hist_bot.append((bot_shares * price) + bot_cash)
        else:
            hist_dca.append(dca_shares * price * rate)
            hist_bot.append(((bot_shares * price) + bot_cash) * rate)
            
    return total_invested, hist_dca, hist_bot

# =========================================================
# 메인 로직
# =========================================================
if run_btn:
    with st.spinner('데이터를 분석 중입니다...'):
        t_a = find_ticker(input_a)
        t_b = find_ticker(input_b)
        
        data = get_data(t_a, t_b, years)
        
        if data is None:
            st.error("데이터를 불러오지 못했습니다. 종목명을 확인해주세요.")
        else:
            # 시뮬레이션
            money_init = init_val * 10000
            money_month = monthly_val * 10000
            
            inv_a, dca_a, bot_a = run_sim(data, 'ASSET_A', t_a, money_init, money_month)
            inv_b, dca_b, bot_b = run_sim(data, 'ASSET_B', t_b, money_init, money_month)
            
            # 결과 표시
            st.success("분석 완료!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"🔴 {input_a} (존버)", value=f"{int(dca_a[-1]/10000):,} 만원", delta=f"{((dca_a[-1]-inv_a)/inv_a*100):.1f}%")
            with col2:
                st.metric(label=f"🔵 {input_b} (존버)", value=f"{int(dca_b[-1]/10000):,} 만원", delta=f"{((dca_b[-1]-inv_b)/inv_b*100):.1f}%")

            # 그래프
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data.index, dca_a, 'r-', label=f'{input_a} (존버)', linewidth=2)
            ax.plot(data.index, bot_a, 'r--', label=f'{input_a} (AI매매)', alpha=0.5)
            ax.plot(data.index, dca_b, 'b-', label=f'{input_b} (존버)', linewidth=2)
            ax.plot(data.index, bot_b, 'skyblue', label=f'{input_b} (AI매매)', linestyle='--')
            
            ax.set_title(f"자산 성장 그래프 (투자원금: {int(inv_a/10000):,}만원)")
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)