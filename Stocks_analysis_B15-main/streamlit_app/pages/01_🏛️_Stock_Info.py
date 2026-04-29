import os
import sys

import pandas as pd
import requests
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import *
from style_utils import inject_global_css, render_sidebar_brand, render_page_header, render_footer, get_plotly_template


CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODEL = "llama3.1-8b"


def get_cerebras_api_key():
    secret_key = None
    try:
        secret_key = st.secrets.get("CEREBRAS_API_KEY")
    except Exception:
        secret_key = None
    return secret_key or os.getenv("CEREBRAS_API_KEY")


def generate_stock_summary(stock_name):
    api_key = get_cerebras_api_key()
    if not api_key:
        return "Set CEREBRAS_API_KEY in your environment or Streamlit secrets to generate the stock summary."

    payload = {
        "model": CEREBRAS_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You write concise stock descriptions for a finance dashboard.",
            },
            {
                "role": "user",
                "content": f"Write one paragraph in about 100 words about {stock_name}. Do not ask questions.",
            },
        ],
        "stream": False,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            CEREBRAS_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
        return f"Unable to generate the stock summary right now: {exc}"


st.set_page_config(page_title="Stock Info", page_icon="🏛️", layout="wide")

inject_global_css()
render_sidebar_brand()

# Sidebar
st.sidebar.markdown("## **Controls**")
stock_dict = fetch_stocks()
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))
stock_exchange = st.sidebar.radio("Choose a stock exchange", ("BSE", "NSE"), index=0)
stock_ticker = build_stock_ticker(stock_dict[stock], stock_exchange)
st.sidebar.text_input("Stock ticker code", value=stock_ticker, disabled=True)

# Fetch stock data
try:
    stock_data_info = fetch_stock_info(stock_ticker)
except Exception as e:
    st.error(f"Error fetching stock data: {e}")
    st.stop()

render_page_header("Stock Information", f"Comprehensive data and AI summary for {stock}")

# AI Summary in a glass card
summary = generate_stock_summary(stock)
st.markdown(f"""
<div class="glass-card">
    <h3>AI Summary</h3>
    <p>{summary}</p>
</div>
""", unsafe_allow_html=True)


def render_section(section_title, section_data):
    st.markdown(f"## **{section_title}**")
    if not section_data:
        st.info(f"No {section_title.lower()} available for this stock.")
        return

    section_df = pd.DataFrame(
        [{"Metric": key, "Value": value} for key, value in section_data.items()]
    )
    st.dataframe(section_df, hide_index=True, use_container_width=True)


if not stock_data_info:
    st.warning("No stock information is available for the selected ticker.")
else:
    for section_title, section_data in stock_data_info.items():
        render_section(section_title, section_data)

render_footer()
