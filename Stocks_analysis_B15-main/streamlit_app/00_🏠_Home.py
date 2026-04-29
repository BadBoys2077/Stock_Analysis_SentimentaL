import streamlit as st
from style_utils import inject_global_css, render_sidebar_brand, render_page_header, render_footer

st.set_page_config(
    page_title="StockPulse Analytics",
    page_icon="📊",
    layout="wide",
)

inject_global_css()
render_sidebar_brand()

render_page_header(
    "StockPulse Analytics",
    "Data-driven stock market insights, analysis, and prediction",
)

# Welcome card
st.markdown("""
<div class="glass-card">
    <h3>Welcome</h3>
    <p>
        We are a team of passionate individuals who believe in the power of data.
        Our mission is to provide easy and accessible tools for everyone to understand,
        analyze, and predict the stock market. With real-time data, AI-powered analysis,
        and ML prediction capabilities, we aim to enhance your investment decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="glass-card" style="text-align: center; min-height: 200px;">
        <span class="card-icon">🏛️</span>
        <h3>Stock Information</h3>
        <p>Comprehensive company data with AI-generated summaries powered by LLM. Explore financials, market data, and key metrics at a glance.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card" style="text-align: center; min-height: 200px;">
        <span class="card-icon">📈</span>
        <h3>Stock Prediction</h3>
        <p>LSTM, RNN, and AutoReg ML models for price forecasting. Visualize historical trends and compare multiple prediction algorithms.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card" style="text-align: center; min-height: 200px;">
        <span class="card-icon">📰</span>
        <h3>News Sentiment</h3>
        <p>Real-time news analysis with VADER sentiment scoring. Track bullish and bearish signals to gauge market sentiment.</p>
    </div>
    """, unsafe_allow_html=True)

# Team card
st.markdown("""
<div class="glass-card">
    <h3>Our Team</h3>
    <p><strong>Suyash Singh</strong> &mdash; 2200320100173</p>
    <p><strong>Suryansh Ranjan</strong> &mdash; 2200320100172</p>
</div>
""", unsafe_allow_html=True)

render_footer()
