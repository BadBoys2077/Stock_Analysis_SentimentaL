import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper import *
from style_utils import (
    inject_global_css,
    render_sidebar_brand,
    render_page_header,
    render_footer,
    get_plotly_template,
    render_news_card,
)

st.set_page_config(
    page_title="Stock News Analysis",
    page_icon="📰",
    layout="wide",
)

inject_global_css()

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


def normalize_yahoo_news_item(item):
    content = item.get('content', item)
    provider = content.get('provider', {}) if isinstance(content, dict) else {}
    click_through = content.get('clickThroughUrl', {}) if isinstance(content, dict) else {}
    canonical = content.get('canonicalUrl', {}) if isinstance(content, dict) else {}

    published_raw = content.get('pubDate') or item.get('pubDate')
    published = published_raw or item.get('published')
    if published_raw:
        try:
            published = datetime.fromisoformat(published_raw.replace('Z', '+00:00'))
        except ValueError:
            published = published_raw
    elif item.get('providerPublishTime'):
        published = datetime.fromtimestamp(item['providerPublishTime'])

    return {
        'title': content.get('title') or item.get('title', ''),
        'publisher': provider.get('displayName') or item.get('publisher', 'Yahoo Finance'),
        'link': click_through.get('url') or canonical.get('url') or item.get('link', ''),
        'published': published,
        'summary': content.get('summary') or content.get('description') or item.get('summary', ''),
        'source': 'Yahoo Finance'
    }


def fetch_company_news(company_name, ticker, max_news=10):
    news_list = []

    try:
        stock = yf.Ticker(ticker)
        yahoo_news = stock.news

        for item in yahoo_news[:max_news]:
            normalized_item = normalize_yahoo_news_item(item)
            if normalized_item['title'] and normalized_item['link']:
                news_list.append(normalized_item)
    except Exception as e:
        st.warning(f"Could not fetch Yahoo Finance news: {e}")

    if len(news_list) < max_news:
        try:
            url = f"https://news.google.com/rss/search?q={company_name}+stock&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')

            for item in items[:max_news - len(news_list)]:
                title = item.title.text
                link = item.link.text
                pub_date = item.pubDate.text
                description = item.description.text if item.description else ''

                news_list.append({
                    'title': title,
                    'publisher': 'Google News',
                    'link': link,
                    'published': pub_date,
                    'summary': description,
                    'source': 'Google News'
                })
        except Exception as e:
            st.warning(f"Could not fetch Google News: {e}")

    return news_list


def analyze_sentiment_vader(text):
    try:
        sid = SentimentIntensityAnalyzer()
        sentiment = sid.polarity_scores(text)
        return sentiment
    except Exception as e:
        st.error(f"Error analyzing sentiment with VADER: {e}")
        return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}


def predict_stock_movement(sentiment_scores):
    avg_compound = sum(score['compound'] for score in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    if avg_compound > 0:
        rise_probability = 50 + (avg_compound * 50)
    else:
        rise_probability = 50 + (avg_compound * 50)

    if rise_probability > 60:
        prediction = "Bullish (Likely Up)"
        direction = "up"
    elif rise_probability < 40:
        prediction = "Bearish (Likely Down)"
        direction = "down"
    else:
        prediction = "Neutral"
        direction = "neutral"

    return prediction, rise_probability, avg_compound, direction


def create_historical_sentiment_chart(ticker, period='3mo'):
    try:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=period)
        company_name = stock.info.get('shortName', ticker)
        news = fetch_company_news(company_name, ticker, max_news=50)

        dates = []
        sentiment_values = []

        current_date = datetime.now()
        for i in range(12):
            end_date = current_date - timedelta(days=i * 7)
            start_date = end_date - timedelta(days=7)

            try:
                period_news = []
                for item in news:
                    published = item['published']
                    if isinstance(published, str):
                        try:
                            published = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %Z')
                        except ValueError:
                            try:
                                published = datetime.fromisoformat(published.replace('Z', '+00:00')).replace(tzinfo=None)
                            except ValueError:
                                continue
                    elif hasattr(published, 'tzinfo') and published.tzinfo is not None:
                        published = published.replace(tzinfo=None)

                    if start_date <= published <= end_date:
                        period_news.append(item)

                sentiments = []
                for item in period_news:
                    sentiment = analyze_sentiment_vader(item['title'] + ' ' + item['summary'])
                    sentiments.append(sentiment)

                if sentiments:
                    avg_sentiment = sum(s['compound'] for s in sentiments) / len(sentiments)
                    dates.append(start_date)
                    sentiment_values.append(avg_sentiment)
            except Exception as e:
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                st.warning(f"Error getting news for period {start_date_str} to {end_date_str}: {e}")

        sentiment_df = pd.DataFrame({
            'Date': dates,
            'Sentiment': sentiment_values
        })

        plotly_style = get_plotly_template()
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['Close'],
            name='Stock Price',
            line=dict(color='#82aaff'),
        ))

        fig.add_trace(go.Scatter(
            x=sentiment_df['Date'],
            y=sentiment_df['Sentiment'],
            name='News Sentiment',
            line=dict(color='#f07178'),
            yaxis='y2',
        ))

        fig.update_layout(**plotly_style)
        fig.update_layout(
            title=f'{ticker} Stock Price vs News Sentiment',
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(
                title=dict(text='Stock Price', font=dict(color='#82aaff')),
                tickfont=dict(color='#82aaff'),
                gridcolor="rgba(56,68,91,0.3)",
            ),
            yaxis2=dict(
                title=dict(text='Sentiment Score', font=dict(color='#f07178')),
                tickfont=dict(color='#f07178'),
                anchor='x',
                overlaying='y',
                side='right',
                range=[-1, 1],
                gridcolor="rgba(56,68,91,0.15)",
            ),
            legend=dict(x=0, y=1.1, orientation='h'),
        )

        return fig
    except Exception as e:
        st.error(f"Error creating historical sentiment chart: {e}")
        return None


# Sidebar
render_sidebar_brand()
st.sidebar.markdown("## **Controls**")
stock_dict = fetch_stocks()
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))
stock_exchange = st.sidebar.radio("Choose a stock exchange", ("BSE", "NSE"), index=0)
stock_ticker = build_stock_ticker(stock_dict[stock], stock_exchange)
st.sidebar.text_input("Stock ticker code", value=stock_ticker, disabled=True)

st.sidebar.markdown("### **News Settings**")
max_news = st.sidebar.slider("Maximum News to Fetch", 5, 20, 10)
news_period = st.sidebar.selectbox(
    "Historical Sentiment Period",
    ["1mo", "3mo", "6mo", "1y"],
    index=1,
)

# Main content
render_page_header("Stock News Analysis", f"Track news sentiment and predict movements for {stock}")

company_name = stock
news_list = fetch_company_news(company_name, stock_ticker, max_news)

st.markdown("## **Latest News**")

if news_list:
    sentiment_scores = []

    for i, news_item in enumerate(news_list):
        sentiment = analyze_sentiment_vader(news_item['title'] + ' ' + news_item['summary'])
        sentiment_scores.append(sentiment)

        published = news_item['published']
        if isinstance(published, datetime):
            published = published.strftime('%b %d, %Y %H:%M')

        render_news_card(
            index=i + 1,
            title=news_item['title'],
            publisher=news_item['publisher'],
            published=str(published),
            summary=news_item['summary'],
            link=news_item['link'],
            sentiment=sentiment,
        )

    # Sentiment prediction
    prediction, rise_probability, avg_compound, direction = predict_stock_movement(sentiment_scores)

    st.markdown("## **Sentiment-Based Prediction**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Average Sentiment", f"{avg_compound:.2f}")
    with col2:
        st.metric("Prediction", prediction)
    with col3:
        st.metric("Confidence", f"{rise_probability:.1f}%")

    # Gauge chart
    plotly_style = get_plotly_template()
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rise_probability,
        number=dict(font=dict(color="#e6edf3", size=36)),
        title=dict(text="Bullish Probability", font=dict(color="#8b949e", size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#6e7681")),
            bar=dict(color="#56c6cc"),
            bgcolor="rgba(22,27,34,0.6)",
            bordercolor="rgba(56,68,91,0.4)",
            steps=[
                dict(range=[0, 40], color="rgba(240,113,120,0.2)"),
                dict(range=[40, 60], color="rgba(255,203,107,0.2)"),
                dict(range=[60, 100], color="rgba(195,232,141,0.2)"),
            ],
            threshold=dict(
                line=dict(color="#e6edf3", width=3),
                thickness=0.75,
                value=rise_probability,
            ),
        ),
    ))
    fig.update_layout(**plotly_style)
    fig.update_layout(
        height=350,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Historical sentiment chart
    st.markdown("## **Historical Sentiment and Stock Price**")
    hist_fig = create_historical_sentiment_chart(stock_ticker, news_period)
    if hist_fig:
        st.plotly_chart(hist_fig, use_container_width=True)
    else:
        st.warning("Could not create historical sentiment chart.")
else:
    st.warning("No news found for the selected stock.")

render_footer()
