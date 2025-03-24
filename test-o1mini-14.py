import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import json
import openai  # 공식 라이브러리 사용
import ta
from ta.utils import dropna
import time
import requests
import logging
import sqlite3
from datetime import datetime, timedelta
import re
import schedule
import numpy as np

# .env 파일에 저장된 환경 변수를 불러오기 (API 키 등)
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Upbit 객체 생성
access = os.getenv("UPBIT_ACCESS_KEY")
secret = os.getenv("UPBIT_SECRET_KEY")
if not access or not secret:
    logger.error("API keys not found. Please check your .env file.")
    raise ValueError("Missing API keys. Please check your .env file.")
upbit = pyupbit.Upbit(access, secret)

# SQLite 데이터베이스 초기화 (컬럼 ai_model 사용)
def init_db():
    conn = sqlite3.connect('bitcoin_trades.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         timestamp TEXT,
         decision TEXT,
         percentage INTEGER,
         reason TEXT,
         btc_balance REAL,
         krw_balance REAL,
         btc_avg_buy_price REAL,
         btc_krw_price REAL,
         reflection TEXT,
         ai_model TEXT,
         token_usage INTEGER,
         duration REAL)
    ''')
    conn.commit()
    return conn

# 거래 기록 저장 함수
def log_trade(conn, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection, ai_model, token_usage, duration):
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("""INSERT INTO trades 
                 (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection, ai_model, token_usage, duration) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection, ai_model, token_usage, duration))
    conn.commit()

# 최근 투자 기록 조회 함수 (최근 7일치)
def get_recent_trades(conn, days=7):
    c = conn.cursor()
    seven_days_ago = (datetime.now() - timedelta(days=days)).isoformat()
    c.execute("SELECT * FROM trades WHERE timestamp > ? ORDER BY timestamp DESC", (seven_days_ago,))
    columns = [column[0] for column in c.description]
    return pd.DataFrame.from_records(data=c.fetchall(), columns=columns)

def calculate_performance(trades_df):
    if trades_df.empty:
        return 0
    initial_balance = trades_df.iloc[-1]['krw_balance'] + trades_df.iloc[-1]['btc_balance'] * trades_df.iloc[-1]['btc_krw_price']
    final_balance = trades_df.iloc[0]['krw_balance'] + trades_df.iloc[0]['btc_balance'] * trades_df.iloc[0]['btc_krw_price']
    return (final_balance - initial_balance) / initial_balance * 100

# ----- 데이터 요약 함수들 ----- #
def summarize_recent_trades(trades_df, max_trades=10):
    if trades_df.empty:
        return "No recent trades."
    count = len(trades_df)
    performance = calculate_performance(trades_df)
    recent = trades_df.tail(max_trades)
    trades_summary = []
    for idx, row in recent.iterrows():
        decision = row.get("decision", "")
        percentage = row.get("percentage", "")
        reason = row.get("reason", "")
        trades_summary.append(f"[{decision}, {percentage}, {reason}]")
    summary = f"Trade count: {count}. 7-day performance: {performance:.2f}%. Recent trades: " + "; ".join(trades_summary)
    return summary

def summarize_ohlcv(df):
    if df.empty:
        return "No OHLCV data."
    last_close = df['close'].iloc[-1]
    min_close = df['close'].min()
    max_close = df['close'].max()
    avg_close = df['close'].mean()
    return f"Last close: {last_close:.2f}, Min: {min_close:.2f}, Max: {max_close:.2f}, Avg: {avg_close:.2f}"

def summarize_orderbook(orderbook):
    # orderbook: assume it is a dict with key "orderbook_units"
    if not orderbook:
        return "No orderbook data."
    units = orderbook.get("orderbook_units", [])
    if not units:
        return "No orderbook data."
    best_bid = units[0].get("bid_price", "N/A")
    best_ask = units[0].get("ask_price", "N/A")
    return f"Best bid: {best_bid}, Best ask: {best_ask}"

# OHLCV에 기술적 지표 추가
def add_indicators(df):
    logger.info("Adding technical indicators to OHLCV data.")
    indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()

    stoch = ta.momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    df['atr'] = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

    df['obv'] = ta.volume.OnBalanceVolumeIndicator(
        close=df['close'], volume=df['volume']).on_balance_volume()

    return df

def get_fear_and_greed_index():
    logger.info("Fetching Fear and Greed Index.")
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['data'][0]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Fear and Greed Index: {e}")
        return None

# --- 단일 API 호출로 투자 판단과 반성 모두 받아오는 함수 ---
def generate_decision_and_reflection(trades_df, current_market_data, orderbook, df_daily_recent, fear_greed_index):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OpenAI API key is missing or invalid.")
        return None

    # 요약된 데이터를 생성합니다.
    trades_summary = summarize_recent_trades(trades_df)
    ohlcv_summary = summarize_ohlcv(df_daily_recent)
    orderbook_summary = summarize_orderbook(orderbook)

    try:
        # o1-mini 모델은 system 메시지를 지원하지 않으므로, 첫 메시지는 assistant 역할로 지정합니다.
        messages = [
            {
                "role": "assistant",
                "content": "You are an expert in Bitcoin trading focusing on technical analysis."
            },
            {
                "role": "user",
                "content": f"""Recent trading data summary:
{trades_summary}

Current market data:
- Fear and Greed Index: {json.dumps(fear_greed_index)}
- Orderbook: {orderbook_summary}
- Daily OHLCV summary: {ohlcv_summary}

Your tasks:
1. Analyze the above summaries and decide whether to "buy", "sell", or "hold".
   - If buying or selling, specify a percentage (an integer between 1 and 100); if holding, percentage must be 0.
   - Important: If BTC balance is insufficient (i.e., less than 5000 KRW worth), choose "hold" over "sell".
2. Provide a reflection that can aid future decisions. Do NOT restrict your analysis excessively – allow up to 2048 tokens if needed.

Output your response in JSON format with exactly these keys:
{{
  "decision": string,
  "percentage": integer,
  "reason": string,
  "reflection": string
}}

Return only the JSON without any additional commentary.
"""
            }
        ]
        # max_tokens 대신 max_completion_tokens 파라미터 사용
        response = openai.chat.completions.create(
            model="o1-mini",
            messages=messages,
            max_completion_tokens=2048
        )
        return response
    except Exception as e:
        logger.error(f"Error during OpenAI API call: {e}")
        return None

# 메인 AI 트레이딩 로직 (단일 API 호출로 투자 판단과 반성 정보를 받아옴)
def ai_trading():
    start_time = time.time()
    logger.info("Starting AI trading process.")
    global upbit

    logger.info("Fetching account balances.")
    all_balances = upbit.get_balances()
    filtered_balances = [balance for balance in all_balances if balance['currency'] in ['BTC', 'KRW']]

    logger.info("Fetching KRW-BTC orderbook.")
    orderbook = pyupbit.get_orderbook("KRW-BTC")

    logger.info("Fetching daily OHLCV data for KRW-BTC.")
    df_daily = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=180)
    df_daily = dropna(df_daily)
    df_daily = add_indicators(df_daily)

    logger.info("Fetching hourly OHLCV data for KRW-BTC.")
    df_hourly = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=168)
    df_hourly = dropna(df_hourly)
    df_hourly = add_indicators(df_hourly)

    df_daily_recent = df_daily.tail(60)

    fear_greed_index = get_fear_and_greed_index()

    logger.info("Summarizing market data to reduce token usage.")
    current_market_data = {
        "fear_greed_index": fear_greed_index,
        "orderbook": orderbook,
        "daily_ohlcv": df_daily_recent.to_dict()  # 참고: 내부 사용은 summary 함수에서 처리됩니다.
    }

    with sqlite3.connect('bitcoin_trades.db') as conn:
        logger.info("Retrieving recent trades from the database.")
        recent_trades = get_recent_trades(conn)

        logger.info("Requesting trading decision and reflection from OpenAI (single API call).")
        response = generate_decision_and_reflection(recent_trades, current_market_data, orderbook, df_daily_recent, fear_greed_index)
        if response is None:
            logger.error("Failed to get response from OpenAI.")
            return

        response_text = response.choices[0].message.content

        # OpenAI 응답 객체를 딕셔너리로 변환하여 토큰 사용량 확인
        response_dict = response.to_dict()
        token_usage = response_dict.get("usage", {}).get("total_tokens", 0)

        ai_model = "o1-mini"

        def parse_ai_response(response_text):
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                    decision = parsed_json.get('decision')
                    percentage = parsed_json.get('percentage')
                    reason = parsed_json.get('reason')
                    reflection = parsed_json.get('reflection')
                    return {'decision': decision, 'percentage': percentage, 'reason': reason, 'reflection': reflection}
                else:
                    logger.error("No JSON found in AI response.")
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return None

        parsed_response = parse_ai_response(response_text)
        if not parsed_response:
            logger.error("Failed to parse AI response.")
            return

        decision = parsed_response.get('decision')
        percentage = parsed_response.get('percentage')
        reason = parsed_response.get('reason')
        reflection = parsed_response.get('reflection')

        if not decision or reason is None or reflection is None:
            logger.error("Incomplete data in AI response.")
            return

        logger.info(f"AI Decision: {decision.upper()}")
        logger.info(f"Percentage: {percentage}")
        logger.info(f"Decision Reason: {reason}")
        logger.info(f"Reflection: {reflection}")

        order_executed = False

        if decision == "buy":
            my_krw = upbit.get_balance("KRW")
            if my_krw is None:
                logger.error("Failed to retrieve KRW balance.")
                return
            buy_amount = my_krw * (percentage / 100) * 0.9995
            if buy_amount > 5000:
                logger.info(f"Buy Order Executed: {percentage}% of available KRW")
                try:
                    order = upbit.buy_market_order("KRW-BTC", buy_amount)
                    if order:
                        logger.info("Buy order executed successfully.")
                        order_executed = True
                    else:
                        logger.error("Buy order failed.")
                except Exception as e:
                    logger.error(f"Error executing buy order: {e}")
            else:
                logger.warning("Buy Order Failed: Insufficient KRW (less than 5000 KRW)")
        elif decision == "sell":
            my_btc = upbit.get_balance("KRW-BTC")
            if my_btc is None:
                logger.error("Failed to retrieve BTC balance.")
                return
            sell_amount = my_btc * (percentage / 100)
            current_price = pyupbit.get_current_price("KRW-BTC")
            if sell_amount * current_price > 5000:
                logger.info(f"Sell Order Executed: {percentage}% of held BTC")
                try:
                    order = upbit.sell_market_order("KRW-BTC", sell_amount)
                    if order:
                        order_executed = True
                    else:
                        logger.error("Sell order failed.")
                except Exception as e:
                    logger.error(f"Error executing sell order: {e}")
            else:
                logger.warning("Sell Order Failed: Insufficient BTC (less than 5000 KRW worth)")
        elif decision == "hold":
            logger.info("Decision is to hold. No action taken.")
        else:
            logger.error("Invalid decision received from AI.")
            return

        # 업데이트된 잔고 확인 후 DB 기록
        time.sleep(2)
        balances = upbit.get_balances()
        btc_balance = next((float(b['balance']) for b in balances if b['currency'] == 'BTC'), 0)
        krw_balance = next((float(b['balance']) for b in balances if b['currency'] == 'KRW'), 0)
        btc_avg_buy_price = next((float(b['avg_buy_price']) for b in balances if b['currency'] == 'BTC'), 0)
        current_btc_price = pyupbit.get_current_price("KRW-BTC")

        duration = round(time.time() - start_time, 2)

        logger.info(f"AI Model: {ai_model}")
        logger.info(f"Token Usage: {token_usage}")
        logger.info(f"Duration: {duration} seconds")

        log_trade(conn, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, current_btc_price, reflection, ai_model, token_usage, duration)

if __name__ == "__main__":
    init_db()
    trading_in_progress = False
    def job():
        global trading_in_progress
        if trading_in_progress:
            logger.warning("Trading job already in progress; skipping.")
            return
        try:
            trading_in_progress = True
            ai_trading()
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            trading_in_progress = False

    logger.info("Starting trading job.")
    job()

    # 스케줄러로 실행할 경우 아래 주석 해제
    # schedule.every().day.at("00:00").do(job)
    # schedule.every().day.at("04:00").do(job)
    # schedule.every().day.at("08:00").do(job)
    # schedule.every().day.at("12:00").do(job)
    # schedule.every().day.at("16:00").do(job)
    # schedule.every().day.at("20:00").do(job)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)
