import os
import streamlit as st


def inject_global_css():
    """Inject the global designing.css into the page. Call after st.set_page_config()."""
    css_path = os.path.join(os.path.dirname(__file__), "designing.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar_brand():
    """Render the branded sidebar header with app name."""
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <h2>StockPulse</h2>
        <p>Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)


def render_page_header(title, subtitle=""):
    """Render a consistent page header with optional subtitle."""
    subtitle_html = f'<p class="subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    """Render the footer with maker credits."""
    st.markdown("""
    <div class="app-footer">
        <p>StockPulse Analytics Dashboard</p>
        <p class="credits">
            Built by <strong>Suyash Singh</strong> (2200320100173)
            &amp; <strong>Suryansh Ranjan</strong> (2200320100172)
        </p>
    </div>
    """, unsafe_allow_html=True)


def get_plotly_template():
    """Return a Plotly layout dict for consistent chart styling."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.6)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color="#8b949e",
            size=12,
        ),
        title=dict(font=dict(color="#e6edf3", size=16, family="Inter")),
        xaxis=dict(
            gridcolor="rgba(56,68,91,0.3)",
            zerolinecolor="rgba(56,68,91,0.3)",
            title_font=dict(color="#8b949e"),
            tickfont=dict(color="#6e7681"),
        ),
        yaxis=dict(
            gridcolor="rgba(56,68,91,0.3)",
            zerolinecolor="rgba(56,68,91,0.3)",
            title_font=dict(color="#8b949e"),
            tickfont=dict(color="#6e7681"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e"),
        ),
        colorway=[
            "#56c6cc",  # teal
            "#f07178",  # coral
            "#ffcb6b",  # amber
            "#82aaff",  # blue
            "#c3e88d",  # green
            "#c792ea",  # purple
            "#ff9cac",  # pink
            "#89ddff",  # cyan
        ],
    )


def render_news_card(index, title, publisher, published, summary, link, sentiment):
    """Render a glassmorphism news card with sentiment-colored border."""
    compound = sentiment["compound"]
    if compound > 0.05:
        sentiment_class = "sentiment-positive"
        badge_class = "positive"
        badge_text = "Bullish"
    elif compound < -0.05:
        sentiment_class = "sentiment-negative"
        badge_class = "negative"
        badge_text = "Bearish"
    else:
        sentiment_class = "sentiment-neutral"
        badge_class = "neutral"
        badge_text = "Neutral"

    # Truncate summary for the card view
    display_summary = summary[:250] + "..." if len(summary) > 250 else summary

    st.markdown(f"""
    <div class="news-card {sentiment_class}">
        <div class="news-title">{index}. {title}</div>
        <div class="news-meta">
            {publisher} &bull; {published}
            &nbsp;&nbsp;
            <span class="sentiment-badge {badge_class}">{badge_text} ({compound:.2f})</span>
        </div>
        <div class="news-summary">{display_summary}</div>
        <div style="margin-top: 8px;">
            <a href="{link}" target="_blank">Read full article &rarr;</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
