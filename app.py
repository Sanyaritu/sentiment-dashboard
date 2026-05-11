import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(__file__))
from src.predict import predict_sentiment, predict_batch

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────
st.title("💬 Sentiment Analysis Dashboard")
st.markdown("Analyze sentiment of Amazon product reviews using ML — built with TF-IDF + Logistic Regression.")
st.divider()

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Review", "📂 Bulk Analysis", "ℹ️ About"])


# ════════════════════════════════════════════════════════
# TAB 1 — Single Review
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Analyze a Single Review")
    st.markdown("Paste any product review below and get an instant sentiment prediction.")

    user_input = st.text_area(
        "Enter your review:",
        placeholder="e.g. This product is amazing! Best purchase I've made all year...",
        height=150
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if analyze_btn and user_input.strip():
        result = predict_sentiment(user_input)

        st.divider()

        # Result display
        col_a, col_b = st.columns([1, 2])

        with col_a:
            color_map = {'Positive': '🟢', 'Neutral': '🟡', 'Negative': '🔴'}
            st.markdown(f"### {result['emoji']} {result['label']}")
            st.metric(
                label="Confidence",
                value=f"{result['confidence']*100:.1f}%"
            )
            st.caption(f"{color_map[result['label']]} Model is **{result['confidence']*100:.1f}%** confident")

        with col_b:
            # Confidence bar chart
            labels = list(result['scores'].keys())
            values = [float(v) for v in result['scores'].values()]
            colors = ['#2ecc71' if l == 'Positive' else '#e74c3c' if l == 'Negative' else '#f39c12' for l in labels]

            fig = go.Figure(go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"{v*100:.1f}%" for v in values],
                textposition='outside'
            ))
            fig.update_layout(
                title="Confidence Scores by Class",
                yaxis=dict(tickformat=".0%", range=[0, 1.1]),
                height=300,
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    elif analyze_btn and not user_input.strip():
        st.warning("Please enter a review to analyze.")


# ════════════════════════════════════════════════════════
# TAB 2 — Bulk Analysis
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Bulk Sentiment Analysis")
    st.markdown("Upload a CSV file with a column named **`review`** to analyze multiple reviews at once.")

    # Download sample CSV button
    sample_data = pd.DataFrame({
        'review': [
            "Absolutely love this product! Works perfectly.",
            "Terrible quality. Broke after one use.",
            "It's okay, nothing special but gets the job done.",
            "Best purchase ever! Highly recommend to everyone.",
            "Very disappointed. Not as described at all."
        ]
    })
    st.download_button(
        "📥 Download Sample CSV",
        sample_data.to_csv(index=False),
        "sample_reviews.csv",
        "text/csv"
    )

    uploaded = st.file_uploader("Upload your CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)

        if 'review' not in df.columns:
            st.error("❌ CSV must have a column named 'review'. Please check your file.")
        else:
            st.info(f"📊 Loaded {len(df)} reviews. Analyzing...")

            with st.spinner("Running predictions..."):
                results = predict_batch(df['review'].astype(str).tolist())
                df['sentiment'] = [r['label'] for r in results]
                df['confidence'] = [f"{r['confidence']*100:.1f}%" for r in results]
                df['emoji'] = [r['emoji'] for r in results]

            st.success(f"✅ Analysis complete!")
            st.divider()

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            counts = df['sentiment'].value_counts()

            with col1:
                st.metric("😊 Positive", counts.get('Positive', 0))
            with col2:
                st.metric("😐 Neutral", counts.get('Neutral', 0))
            with col3:
                st.metric("😠 Negative", counts.get('Negative', 0))

            # Pie chart
            fig_pie = px.pie(
                values=counts.values,
                names=counts.index,
                color=counts.index,
                color_discrete_map={
                    'Positive': '#2ecc71',
                    'Neutral': '#f39c12',
                    'Negative': '#e74c3c'
                },
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Results table
            st.subheader("Detailed Results")
            st.dataframe(
                df[['review', 'emoji', 'sentiment', 'confidence']],
                use_container_width=True,
                height=400
            )

            # Download results
            st.download_button(
                "📥 Download Results CSV",
                df.to_csv(index=False),
                "sentiment_results.csv",
                "text/csv"
            )


# ════════════════════════════════════════════════════════
# TAB 3 — About
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("About This Project")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🧠 Model Details
        - **Algorithm:** Logistic Regression
        - **Features:** TF-IDF with bigrams (15,000 features)
        - **Training data:** 30,000 Amazon food reviews
        - **Classes:** Positive, Neutral, Negative
        - **Accuracy:** 73% (balanced 3-class)
        - **Dataset:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
        """)

    with col2:
        st.markdown("""
        ### ⚙️ Tech Stack
        - **Python** — core language
        - **scikit-learn** — TF-IDF + Logistic Regression
        - **NLTK** — text preprocessing & stopwords
        - **Streamlit** — web dashboard
        - **Plotly** — interactive charts
        - **Pandas** — data manipulation

        ### 👩‍💻 Built By
        - Sanya — B.Tech Data Science, IILM
        """)

    st.divider()
    st.markdown("""
    ### 🔍 How It Works
    1. Raw review text is cleaned — HTML removed, lowercased, stopwords filtered
    2. Cleaned text is converted to a TF-IDF vector (word frequency matrix)
    3. Logistic Regression predicts the sentiment class + confidence scores
    4. Results displayed with interactive confidence charts
    """)