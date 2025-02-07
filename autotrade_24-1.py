import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import json
from openai import OpenAI
import ta
from ta.utils import dropna
import time
import requests
import logging
from pydantic import BaseModel
import sqlite3
from datetime import datetime, timedelta
import schedule

# .env 파일에 저장된 환경 변수를 불러오기
load_dotenv()

# 로깅 설정 (최종 결과만 출력)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Upbit 객체 생성
access = os.getenv("UPBIT_ACCESS_KEY")
secret = os.getenv("UPBIT_SECRET_KEY")
if not access or not secret:
    logger.error("API keys not found. Please check your .env file.")
    raise ValueError("Missing API keys. Please check your .env file.")
upbit = pyupbit.Upbit(access, secret)

# OpenAI 투자 판단용 클래스
class TradingDecision(BaseModel):
    decision: str
    percentage: int
    reason: str

# 데이터베이스 초기화 (테이블 스키마 업데이트 포함)
def init_db():
    conn = sqlite3.connect('bitcoin_trades.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            duration REAL
        )
    ''')
    conn.commit()
    c.execute("PRAGMA table_info(trades)")
    columns = [col[1] for col in c.fetchall()]
    required_columns = {
        "ai_model": "TEXT",
        "token_usage": "INTEGER",
        "duration": "REAL"
    }
    for col, col_type in required_columns.items():
        if col not in columns:
            try:
                c.execute(f"ALTER TABLE trades ADD COLUMN {col} {col_type}")
                logger.info(f"컬럼 '{col}' 추가됨.")
            except sqlite3.Error as e:
                logger.error(f"컬럼 '{col}' 추가 중 오류 발생: {e}")
    conn.commit()
    return conn

# 거래 기록 저장 함수
def log_trade(conn, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price,
              reflection='', ai_model='', token_usage=0, duration=None):
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("""
        INSERT INTO trades 
        (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price,
         reflection, ai_model, token_usage, duration)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price,
          reflection, ai_model, token_usage, duration))
    conn.commit()

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

# 공포 탐욕 지수 캐싱 (1시간 동안 캐시)
fear_greed_cache = {"value": None, "timestamp": None}
def get_fear_and_greed_index():
    current_time = time.time()
    # 캐시된 값이 있고 1시간 이내이면 캐시된 값 사용
    if fear_greed_cache["value"] is not None and (current_time - fear_greed_cache["timestamp"] < 3600):
        return fear_greed_cache["value"]
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        fear_greed_cache["value"] = data['data'][0]
        fear_greed_cache["timestamp"] = current_time
        return data['data'][0]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Fear and Greed Index: {e}")
        return None

# OpenAI API 호출: 투자 결정 및 reflection 받기
def get_ai_decision_and_reflection(recent_trades, current_market_data, filtered_balances, orderbook, daily_summary,
                                   hourly_summary, candle_patterns, support_levels, resistance_levels, investment_rules):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        logger.error("OpenAI API key is missing or invalid.")
        return None, None, 0

    prompt = f"""
Recent trading data:
{recent_trades.to_json(orient='records')}

Market data:
Fear & Greed Index: {json.dumps(current_market_data.get('fear_greed_index'))}
Orderbook: {json.dumps(orderbook)}
Daily Summary: {json.dumps(daily_summary)}
Hourly Summary: {json.dumps(hourly_summary)}
Candlestick Patterns: {json.dumps(candle_patterns)}
Support Levels: {json.dumps(support_levels)}
Resistance Levels: {json.dumps(resistance_levels)}
Investment Rules: {investment_rules}

Based on the above, output:
1. Trading decision as a JSON object with schema:
   {{
     "decision": "buy" | "sell" | "hold",
     "percentage": <integer>,
     "reason": "<brief explanation>"
   }}
2. A brief reflection (max 250 words) on recent trading performance and market trends.

Return a JSON object with keys "trading_decision" and "reflection".
    """
    # 진행 상황 표시
    logger.info("Preparing data and sending to OpenAI API...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in Bitcoin trading and technical analysis. "
                        "Analyze the provided data and output a trading decision along with a brief reflection."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "trading_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "trading_decision": {
                                "type": "object",
                                "properties": {
                                    "decision": {"type": "string", "enum": ["buy", "sell", "hold"]},
                                    "percentage": {"type": "integer"},
                                    "reason": {"type": "string"}
                                },
                                "required": ["decision", "percentage", "reason"],
                                "additionalProperties": False
                            },
                            "reflection": {"type": "string"}
                        },
                        "required": ["trading_decision", "reflection"],
                        "additionalProperties": False
                    }
                }
            },
            max_tokens=4095
        )
    except Exception as e:
        logger.error(f"Error in AI decision/reflection: {e}")
        return None, None, 0

    token_usage = response.usage.total_tokens
    raw_result = response.choices[0].message.content
    try:
        result = json.loads(raw_result)
    except Exception as parse_err:
        logger.error(f"JSON parsing error: {parse_err}")
        return None, None, token_usage
    return result.get("trading_decision"), result.get("reflection"), token_usage

def add_indicators(df):
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
    return df

def calculate_change(series, periods):
    if len(series) > periods:
        change = ((series.iloc[-1] - series.iloc[-periods-1]) / series.iloc[-periods-1] * 100)
        return round(change, 2)
    return 0

def summarize_ohlcv_data(df):
    latest_data = df.iloc[-1]
    summary = {
        'current': {
            'close': latest_data['close'],
            'volume': latest_data['volume'],
            'rsi': round(latest_data['rsi'], 2),
            'macd': round(latest_data['macd'], 2),
            'macd_signal': round(latest_data['macd_signal'], 2),
            'bb_upper': round(latest_data['bb_bbh'], 2),
            'bb_lower': round(latest_data['bb_bbl'], 2),
            'sma_20': round(latest_data['sma_20'], 2),
            'ema_12': round(latest_data['ema_12'], 2)
        },
        'price_history': df['close'].tail(7).tolist(),
        'bollinger_width': round((latest_data['bb_bbh'] - latest_data['bb_bbl']) / latest_data['bb_bbm'] * 100, 2)
    }
    return summary

def analyze_candle_pattern(row):
    body = row['close'] - row['open']
    total_length = row['high'] - row['low']
    upper_shadow = row['high'] - max(row['open'], row['close'])
    lower_shadow = min(row['open'], row['close']) - row['low']
    candle_type = 'bullish' if body > 0 else 'bearish'
    body_ratio = abs(body) / total_length if total_length > 0 else 0
    pattern = None
    if body_ratio < 0.1 and total_length > 0:
        pattern = 'doji'
    elif upper_shadow > abs(body) * 2 and body < 0:
        pattern = 'shooting_star' if candle_type == 'bearish' else 'inverted_hammer'
    elif lower_shadow > abs(body) * 2 and body > 0:
        pattern = 'hammer' if candle_type == 'bullish' else 'hanging_man'
    return {
        'type': candle_type,
        'pattern': pattern,
        'body_ratio': round(body_ratio, 2)
    }

def find_candle_patterns(df):
    patterns = []
    for i in range(-5, 0):
        pattern = analyze_candle_pattern(df.iloc[i])
        patterns.append(pattern)
    recent_three = df.tail(3)
    three_soldiers = all(recent_three['close'] > recent_three['open']) and all(recent_three['close'].diff().dropna() > 0)
    three_crows = all(recent_three['close'] < recent_three['open']) and all(recent_three['close'].diff().dropna() < 0)
    return {
        'recent_patterns': patterns,
        'three_soldiers': three_soldiers,
        'three_crows': three_crows
    }

def find_support_resistance_levels(series, periods=30, threshold=0.02):
    series = series.tail(periods)
    price_points = sorted(series.tolist())
    levels = []
    current_price = series.iloc[-1]
    price_threshold = current_price * threshold
    cluster = []
    for price in price_points:
        if not cluster or abs(cluster[-1] - price) <= price_threshold:
            cluster.append(price)
        else:
            if len(cluster) >= 3:
                levels.append(sum(cluster) / len(cluster))
            cluster = [price]
    return [round(level, 2) for level in levels]

def find_support_levels(df):
    return find_support_resistance_levels(df['low'])

def find_resistance_levels(df):
    return find_support_resistance_levels(df['high'])

investment_rules = """
1. Buy when: Price approaches strong support level with increasing volume and clear bullish candlestick patterns
2. Sell when: Price approaches strong resistance level with decreasing volume or reversal candlestick patterns
3. Hold if the trend is unclear or in a consolidation phase
4. Never invest more than 20-30% of available balance in a single position; apply position scaling as needed
5. Focus solely on chart patterns and price action, minimizing the influence of news and external market sentiment
"""

def ai_trading():
    global upbit
    all_balances = upbit.get_balances()
    filtered_balances = [balance for balance in all_balances if balance['currency'] in ['BTC', 'KRW']]
    orderbook = pyupbit.get_orderbook("KRW-BTC")
    df_daily = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
    df_daily = dropna(df_daily)
    df_daily = add_indicators(df_daily)
    df_hourly = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
    df_hourly = dropna(df_hourly)
    df_hourly = add_indicators(df_hourly)
    fear_greed_index = get_fear_and_greed_index()
    daily_summary = summarize_ohlcv_data(df_daily)
    hourly_summary = summarize_ohlcv_data(df_hourly)
    candle_patterns = find_candle_patterns(df_daily)
    support_levels = find_support_levels(df_daily)
    resistance_levels = find_resistance_levels(df_daily)
    
    client_model_name = "gpt-4o"
    try:
        with sqlite3.connect('bitcoin_trades.db') as conn:
            recent_trades = get_recent_trades(conn)
            current_market_data = {"fear_greed_index": fear_greed_index}
            
            logger.info("Preparing data for OpenAI API call...")
            trading_decision, reflection, total_token_usage = get_ai_decision_and_reflection(
                recent_trades,
                current_market_data,
                filtered_balances,
                orderbook,
                daily_summary,
                hourly_summary,
                candle_patterns,
                support_levels,
                resistance_levels,
                investment_rules
            )
            
            if trading_decision is None:
                logger.error("AI did not return a valid trading decision.")
                return
            logger.info(f"Token Usage: {total_token_usage}")
            logger.info(f"AI Decision: {trading_decision.get('decision').upper()}")
            
            order_executed = False
            if trading_decision.get("decision") == "buy":
                my_krw = upbit.get_balance("KRW")
                if my_krw is None:
                    logger.error("Failed to retrieve KRW balance.")
                    return
                buy_amount = my_krw * (trading_decision.get("percentage") / 100) * 0.9995
                if buy_amount > 5000:
                    logger.info(f"Buy Order Executed: {trading_decision.get('percentage')}% of available KRW")
                    try:
                        order = upbit.buy_market_order("KRW-BTC", buy_amount)
                        if order:
                            logger.info(f"Buy order executed successfully: {order}")
                            order_executed = True
                        else:
                            logger.error("Buy order failed.")
                    except Exception as e:
                        logger.error(f"Error executing buy order: {e}")
                else:
                    logger.warning("Buy Order Failed: Insufficient KRW (less than 5000 KRW)")
            elif trading_decision.get("decision") == "sell":
                my_btc = upbit.get_balance("KRW-BTC")
                if my_btc is None:
                    logger.error("Failed to retrieve BTC balance.")
                    return
                sell_amount = my_btc * (trading_decision.get("percentage") / 100)
                current_price = pyupbit.get_current_price("KRW-BTC")
                if sell_amount * current_price > 5000:
                    logger.info(f"Sell Order Executed: {trading_decision.get('percentage')}% of held BTC")
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
            time.sleep(2)
            balances = upbit.get_balances()
            btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
            krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
            btc_avg_buy_price = next((float(balance['avg_buy_price']) for balance in balances if balance['currency'] == 'BTC'), 0)
            current_btc_price = pyupbit.get_current_price("KRW-BTC")
            log_trade(conn, trading_decision.get("decision"), trading_decision.get("percentage") if order_executed else 0,
                      trading_decision.get("reason"), btc_balance, krw_balance, btc_avg_buy_price, current_btc_price,
                      reflection, client_model_name, total_token_usage, None)
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        return

if __name__ == "__main__":
    init_db()
    trading_in_progress = False

    def job():
        global trading_in_progress
        if trading_in_progress:
            logger.warning("Trading job is already in progress, skipping this run.")
            return
        try:
            trading_in_progress = True
            start_time = time.time()
            ai_trading()
            end_time = time.time()
            total_duration = round(end_time - start_time, 2)
            logger.info(f"Total duration: {total_duration:.2f} sec")
            with sqlite3.connect('bitcoin_trades.db') as conn:
                cur = conn.cursor()
                cur.execute("UPDATE trades SET duration = ? WHERE id = (SELECT MAX(id) FROM trades)", (total_duration,))
                conn.commit()
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            trading_in_progress = False

    # ## Uncomment the test
    job()

    # Uncomment for scheduler use:
    # schedule.every().day.at("09:00").do(job)
    # schedule.every().day.at("15:00").do(job)
    # schedule.every().day.at("21:00").do(job)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)