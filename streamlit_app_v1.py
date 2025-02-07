import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pyupbit
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests

# .env 파일에서 환경변수 로드
load_dotenv()

# ==============================
# News Functions
# ==============================
@st.cache_data(ttl=28800)  # 8시간마다 캐시 초기화 (하루 3번 이상 새 뉴스 불러옴)
def get_newsapi_news():
    NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
    url = f"https://newsapi.org/v2/everything?q=bitcoin&apiKey={NEWSAPI_API_KEY}&language=en&sortBy=publishedAt&pageSize=10"
    
    response = requests.get(url)
    if response.status_code == 200:
        news_items = response.json().get("articles", [])
        
        news_list = []
        for item in news_items:
            title = item.get("title")
            source = item.get("source", {}).get("name")
            timestamp = item.get("publishedAt")
            link = item.get("url")
            
            if timestamp:
                try:
                    date_obj = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
                    timestamp = date_obj.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    pass
            
            news_list.append({
                "title": title,
                "source": source,
                "timestamp": timestamp,
                "link": link
            })
        
        # 중복 뉴스 제거 후 최대 5개 뉴스 반환
        news_df = pd.DataFrame(news_list).drop_duplicates(subset=["title"])
        return news_df.head(5)
    else:
        st.error("Failed to fetch news from NewsAPI.")
        return pd.DataFrame()
    
# ==============================
# Database Functions
# ==============================
def get_connection():
    return sqlite3.connect('bitcoin_trades.db')

def load_data():
    conn = get_connection()
    query = "SELECT * FROM trades ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# ==============================
# Analysis Functions
# ==============================
def calculate_initial_investment(df):
    initial_krw_balance = df.iloc[-1]['krw_balance']
    initial_btc_balance = df.iloc[-1]['btc_balance']
    initial_btc_price = df.iloc[-1]['btc_krw_price']
    initial_total_investment = initial_krw_balance + (initial_btc_balance * initial_btc_price)
    return initial_total_investment

def calculate_current_investment(df):
    current_krw_balance = df.iloc[0]['krw_balance']
    current_btc_balance = df.iloc[0]['btc_balance']
    current_btc_price = pyupbit.get_current_price("KRW-BTC")
    if current_btc_price is None:
        current_btc_price = df.iloc[0]['btc_krw_price']
    current_total_investment = current_krw_balance + (current_btc_balance * current_btc_price)
    return current_total_investment

def calculate_market_return(df):
    filtered_df = df[df['decision'].isin(['buy', 'sell'])]
    if filtered_df.empty:
        st.warning("No buy/sell trades found. Market return cannot be calculated.")
        return 0, None
    
    first_trade = filtered_df.iloc[-1]
    last_trade = df.iloc[0]
    
    initial_price = first_trade['btc_krw_price']
    current_price = last_trade['btc_krw_price']
    market_return = ((current_price - initial_price) / initial_price) * 100
    
    return market_return, pd.to_datetime(first_trade['timestamp'])

def calculate_trade_statistics(df):
    stats = {}
    
    stats['total_trades'] = len(df)
    
    decision_counts = df['decision'].value_counts()
    decision_percentages = (decision_counts / len(df) * 100).round(1)
    stats['decisions'] = {
        decision: {
            'count': count,
            'percentage': percentage
        }
        for decision, count, percentage in zip(
            decision_counts.index,
            decision_counts.values,
            decision_percentages.values
        )
    }
    
    buy_sell_df = df[df['decision'].isin(['buy', 'sell'])]
    
    if len(buy_sell_df) > 1:
        time_diffs = buy_sell_df['timestamp'].diff().dropna()
        stats['avg_trade_interval'] = time_diffs.mean()
    else:
        stats['avg_trade_interval'] = pd.Timedelta(0)
    
    if not buy_sell_df.empty:
        buy_sell_df = buy_sell_df.copy()
        buy_sell_df.loc[:, 'next_price'] = buy_sell_df['btc_krw_price'].shift(-1)
        buy_sell_df.loc[:, 'profit'] = np.where(
            buy_sell_df['decision'] == 'buy',
            buy_sell_df['next_price'] - buy_sell_df['btc_krw_price'],
            buy_sell_df['btc_krw_price'] - buy_sell_df['next_price']
        )
        profitable_trades = len(buy_sell_df[buy_sell_df['profit'] > 0])
        stats['success_rate'] = (profitable_trades / len(buy_sell_df)) * 100
    else:
        stats['success_rate'] = 0
    
    if not buy_sell_df.empty:
        profitable = (buy_sell_df['profit'] > 0).astype(int)
        profitable_str = profitable.astype(str)
        success_streaks = profitable_str.str.cat().split('0')
        fail_streaks = profitable_str.str.cat().split('1')
        stats['max_success_streak'] = max(len(s) for s in success_streaks)
        stats['max_fail_streak'] = max(len(s) for s in fail_streaks)
    else:
        stats['max_success_streak'] = 0
        stats['max_fail_streak'] = 0
    
    return stats

# ==============================
# Chart Functions
# ==============================
def load_market_data(interval):
    intervals = {
        '5m': {'count': 144, 'days': 1},
        '15m': {'count': 192, 'days': 2},
        '30m': {'count': 336, 'days': 7},
        '1h': {'count': 720, 'days': 30},
        '4h': {'count': 180, 'days': 30},
        '1d': {'count': 365, 'days': 365},
        '1w': {'count': 200, 'days': 1400},
        '1M': {'count': 60, 'days': 1800},
    }
    
    try:
        market_df = pyupbit.get_ohlcv("KRW-BTC", interval=interval, count=intervals[interval]['count'])
        return market_df
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        return pd.DataFrame()

def create_candlestick_chart(market_df, trades_df, avg_buy_price, current_price):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=market_df.index,
        open=market_df['open'],
        high=market_df['high'],
        low=market_df['low'],
        close=market_df['close'],
        name='OHLC'
    ), row=1, col=1)

    colors = ['red' if row['open'] > row['close'] else 'green' for _, row in market_df.iterrows()]
    fig.add_trace(go.Bar(
        x=market_df.index,
        y=market_df['volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.5
    ), row=2, col=1)

    if avg_buy_price:
        fig.add_trace(go.Scatter(
            x=market_df.index,
            y=[avg_buy_price] * len(market_df),
            mode='lines',
            name='Avg Buy Price',
            line=dict(color='blue', width=1, dash='dot')
        ), row=1, col=1)

    if current_price:
        fig.add_trace(go.Scatter(
            x=market_df.index,
            y=[current_price] * len(market_df),
            mode='lines',
            name='Current Price',
            line=dict(color='green', width=1, dash='dot')
        ), row=1, col=1)

    for decision, color, symbol in [
        ('buy', 'green', 'triangle-up'),
        ('sell', 'red', 'triangle-down'),
        ('hold', 'yellow', 'circle')
    ]:
        decision_points = trades_df[trades_df['decision'] == decision]
        if not decision_points.empty:
            fig.add_trace(go.Scatter(
                x=decision_points['timestamp'],
                y=decision_points['btc_krw_price'],
                mode='markers',
                name=decision.capitalize(),
                marker=dict(color=color, size=10, symbol=symbol),
                hovertemplate=f'<b>{decision.capitalize()} Point</b><br>' +
                              'Price: ₩%{y:,.0f}<br>' +
                              'Date: %{x}<br>' +
                              '<extra></extra>'
            ), row=1, col=1)

    fig.update_layout(
        title='Bitcoin Price with Trading Decisions',
        yaxis_title='Price (KRW)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness=0.1,
        height=800
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig

def create_decision_charts(stats):
    decisions = stats['decisions']
    labels = list(decisions.keys())
    values = [d['count'] for d in decisions.values()]
    percentages = [d['percentage'] for d in decisions.values()]
    
    colors = {'buy': 'green', 'sell': 'red', 'hold': 'yellow'}
    color_list = [colors.get(label, 'gray') for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker_colors=color_list,
        text=[f'{p}%' for p in percentages],
        textinfo='label+text',
        hovertemplate="<b>%{label}</b><br>" +
                      "Count: %{value}<br>" +
                      "Percentage: %{text}<br>" +
                      "<extra></extra>"
    )])
    
    fig.update_layout(
        title='Decisions Distribution',
        showlegend=False,
        height=250,
        margin=dict(t=30, l=0, r=0, b=0)
    )
    
    return fig

# ==============================
# Main Application
# ==============================
def main():
    st.set_page_config(
        page_title="Bitcoin Trades Viewer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # 모던하고 스타일리쉬한 CSS 적용
    st.markdown(
        """
        <style>
        /* 전체 배경 및 폰트 설정 */
        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background-color: #f5f5f5;
        }
        /* 메인 컨테이너 */
        .main-container {
            max-width: 70%;
            margin: auto;
            padding: 20px;
        }
        /* 대제목 */
        .main-title {
            font-size: 2.5em;
            font-weight: bold;
            text-align: left;
            margin-bottom: 30px;
            color: #007bff;
        }
        /* 섹션 제목 */
        .section-title {
            font-size: 1.75em;
            font-weight: bold;
            margin-top: 40px;
            margin-bottom: 15px;
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
        }
        /* 부제목 */
        .subsection-title {
            font-size: 1.25em;
            font-weight: bold;
            margin-top: 25px;
            margin-bottom: 10px;
            color: #333;
        }
        /* 정보 박스 (reason, reflection 등) */
        .info-box {
            background-color: #ffffff;
            border-left: 5px solid #007bff;
            padding: 15px;
            margin-top: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        }
        /* 뉴스 박스 */
        .news-box {
            background-color: #ffffff;
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .news-title {
            font-weight: bold;
            color: #007bff;
            font-size: 1.1em;
        }
        .news-meta {
            color: #666;
            font-size: 0.9em;
        }
        /* 중앙 정렬 */
        .center {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # 메인 컨테이너 시작
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    
    st.markdown("<div class='main-title'>AI Trades Viewer</div>", unsafe_allow_html=True)
    
    # 데이터 로드
    trades_df = load_data()
    if trades_df.empty:
        st.warning('No trade data available.')
        return
    
    # 기본 계산
    initial_investment = calculate_initial_investment(trades_df)
    current_investment = calculate_current_investment(trades_df)
    profit_rate = ((current_investment - initial_investment) / initial_investment) * 100
    market_return, first_trade_date = calculate_market_return(trades_df)
    
    # 수익률 표시 섹션
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title center'>📈 Strategy Return</div>", unsafe_allow_html=True)
        st.metric(
            label="Current Value",
            value=f"{current_investment:,.0f}",
            delta=f"{current_investment - initial_investment:,.0f} ({profit_rate:.2f}%)"
        )
        st.caption(f"Initial Investment: ₩{initial_investment:,.0f}")
    with col2:
        st.markdown("<div class='section-title center'>📊 Market Return</div>", unsafe_allow_html=True)
        if first_trade_date is not None:
            initial_price = trades_df[trades_df['timestamp'] >= first_trade_date].iloc[-1]['btc_krw_price']
            current_price = trades_df.iloc[0]['btc_krw_price']
            market_change = current_price - initial_price
            st.metric(
                label="Current BTC Price",
                value=f"{current_price:,.0f}",
                delta=f"{market_change:,.0f} ({market_return:.2f}%)"
            )
            st.caption(f"Initial BTC Price: ₩{initial_price:,.0f} (Since {first_trade_date.strftime('%Y-%m-%d')})")
        else:
            st.subheader('N/A')
            st.caption('(No buy/sell trades found)')
    
    # 기본 통계 섹션
    st.markdown("<div class='section-title'>Basic Statistics</div>", unsafe_allow_html=True)
    stats_cols = st.columns(3)
    with stats_cols[0]:
        st.metric("Total Trades", len(trades_df))
    with stats_cols[1]:
        st.metric("First Trade", trades_df['timestamp'].min().strftime("%Y-%m-%d"))
    with stats_cols[2]:
        st.metric("Last Trade", trades_df['timestamp'].max().strftime("%Y-%m-%d"))
    
    # Trading Decisions Analysis 섹션
    st.markdown("<div class='section-title'>Decisions Analysis</div>", unsafe_allow_html=True)
    trade_stats = calculate_trade_statistics(trades_df)
    col1, col2 = st.columns([2, 3])
    with col1:
        fig = create_decision_charts(trade_stats)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        col_metrics = st.columns(2)
        with col_metrics[0]:
            st.metric(
                "Trading Success Rate",
                f"{trade_stats['success_rate']:.1f}%",
                help="Percentage of trades that resulted in profit"
            )
            st.metric(
                "Max Success Streak",
                trade_stats['max_success_streak'],
                help="Maximum consecutive profitable trades"
            )
        with col_metrics[1]:
            st.metric(
                "Avg Trade Interval",
                f"{trade_stats['avg_trade_interval'].total_seconds() / 3600:.1f}h",
                help="Average time between trades"
            )
            st.metric(
                "Max Failure Streak",
                trade_stats['max_fail_streak'],
                help="Maximum consecutive unprofitable trades"
            )
    
    # BTC Price Chart 섹션
    st.markdown("<div class='section-title'>Price with Trading Points</div>", unsafe_allow_html=True)
    selected_interval = st.radio(
        "Select time interval",
        ['5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'],
        index=5,
        horizontal=True,
        key='interval'
    )
    market_df = load_market_data(selected_interval)
    if not market_df.empty:
        start_time = market_df.index[0]
        filtered_trades_df = trades_df[trades_df['timestamp'] >= start_time].copy()
        avg_buy_price = filtered_trades_df['btc_avg_buy_price'].mean()
        current_price = pyupbit.get_current_price("KRW-BTC")
        
        fig = create_candlestick_chart(market_df, filtered_trades_df, avg_buy_price, current_price)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Failed to load market data from pyupbit.")
    
    # Trade History 및 최근 거래의 reason과 reflection 표시 섹션
    st.markdown("<div class='section-title'>Trade History</div>", unsafe_allow_html=True)
    if 'filtered_trades_df' in locals():
        st.dataframe(filtered_trades_df, height=400)
        latest_trade = filtered_trades_df.iloc[0]
    else:
        st.dataframe(trades_df, height=400)
        latest_trade = trades_df.iloc[0]

    with st.container():
        st.markdown(f"""
            <div class='info-box'>
                <h4>Latest Trade Infomation</h4>
                <p><b>AI Model</b> : {latest_trade.get('ai_model', 'No ai_model provided')}</p>
                <p><b>Total Token Usage</b> : {latest_trade.get('token_usage', 'No token_usage provided')}</p>                
                <p><b>Total duration time</b> : {latest_trade.get('duration', 'No duration provided')}</p>
            </div>
            <div class='info-box'>
                <h4>Latest Trade Reason</h4>
                <p>{latest_trade.get('reason', 'No reason provided')}</p>
            </div>
            <div class='info-box'>
                <h4>Latest Trade Reflection</h4>
                <p>{latest_trade.get('reflection', 'No reflection provided')}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # NEWSAPI News 섹션
    st.markdown("<div class='section-title'>NEWSAPI News</div>", unsafe_allow_html=True)
    newsapi_news_df = get_newsapi_news()
    if not newsapi_news_df.empty:
        for _, news in newsapi_news_df.iterrows():
            st.markdown(f"""
                <div class='news-box'>
                    <div class='news-title'><a href="{news['link']}" target="_blank">{news['title']}</a></div>
                    <div class='news-meta'>{news['source']} | ⏰ {news['timestamp']}</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No recent news available from NEWSAPI.")
    
    # 메인 컨테이너 종료
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
