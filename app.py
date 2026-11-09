
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"

st.set_page_config(
    page_title="FashionPulse AI",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv(
        OUTPUT / "reviews_with_sentiment.csv"
    )

@st.cache_resource
def load_model():
    return joblib.load(
        ROOT / "models" / "review_model.joblib"
    )

df = load_data()
model = load_model()

st.title("FashionPulse AI")
st.caption("Fashion Retail Customer Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Text Analytics",
        "Complaint Intelligence",
        "Predict Recommendation"
    ]
)

if page == "Executive Dashboard":

    st.header("Customer Experience Overview")

    categories = sorted(
        df["Class Name"].dropna().unique()
    )

    selected = st.multiselect(
        "Filter clothing categories",
        categories,
        default=categories
    )

    filtered = df[
        df["Class Name"].isin(selected)
    ].copy()

    if filtered.empty:
        st.warning("No reviews match the selected categories.")
        st.stop()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Reviews", f"{len(filtered):,}")

    c2.metric(
        "Recommendation Rate",
        f"{filtered['Recommended IND'].mean():.1%}"
    )

    c3.metric(
        "Average Rating",
        f"{filtered['Rating'].mean():.2f}"
    )

    left, right = st.columns(2)

    with left:
        sentiment = (
            filtered["sentiment_label"]
            .value_counts()
            .reset_index()
        )
        sentiment.columns = ["Sentiment", "Count"]

        fig = px.pie(
            sentiment,
            names="Sentiment",
            values="Count",
            title="Customer Sentiment Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        category = (
            filtered.groupby("Class Name")
            ["Recommended IND"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            category,
            x="Class Name",
            y="Recommended IND",
            title="Recommendation Rate by Category",
            labels={
                "Recommended IND": "Recommendation Rate"
            }
        )
        fig.update_yaxes(range=[0, 1])

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Review Explorer")

    st.dataframe(
        filtered[[
            "Class Name",
            "Rating",
            "Review Text",
            "sentiment_label",
            "Recommended IND"
        ]],
        use_container_width=True
    )

elif page == "Text Analytics":

    st.header("Text Feature Analysis")

    sentiment_choice = st.selectbox(
        "Select recommendation group",
        ["Not Recommended", "Recommended"]
    )

    filename = (
        "negative_terms.csv"
        if sentiment_choice == "Not Recommended"
        else "positive_terms.csv"
    )

    terms = pd.read_csv(OUTPUT / filename)

    fig = px.bar(
        terms.head(20),
        x="mean_tfidf",
        y="term",
        orientation="h",
        title="Top TF-IDF Terms"
    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    st.plotly_chart(fig, use_container_width=True)

elif page == "Predict Recommendation":

    st.header("Review Recommendation Predictor")

    review = st.text_area(
        "Enter customer feedback",
        "The dress looks beautiful but the fabric is cheap."
    )

    if st.button("Analyze"):
        if not review.strip():
            st.warning("Enter a review first.")
        else:
            probability = model.predict_proba([review])[0, 1]

            st.metric(
                "Recommendation Probability",
                f"{probability:.1%}"
            )

            st.progress(float(probability))

            if probability >= 0.5:
                st.success("Predicted: Recommended")
            else:
                st.error("Predicted: Not Recommended")


elif page == "Complaint Intelligence":

    st.header("Customer Complaint Intelligence")

    complaints_df = pd.read_csv(
        OUTPUT / "reviews_with_complaints.csv"
    )

    complaint_columns = [
        "Sizing",
        "Fabric Quality",
        "Price",
        "Delivery",
        "Design"
    ]

    counts = (
        complaints_df[complaint_columns]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    counts.columns = ["Complaint", "Count"]

    fig = px.bar(
        counts,
        x="Complaint",
        y="Count",
        title="Detected Customer Complaints",
        color="Complaint"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Complaints by Clothing Category")

    category_complaints = (
        complaints_df.groupby("Class Name")[
            complaint_columns
        ]
        .mean()
        .mul(100)
    )

    fig = px.imshow(
        category_complaints,
        labels={
            "x": "Complaint Category",
            "y": "Clothing Category",
            "color": "Detected (%)"
        },
        aspect="auto",
        title="Complaint Detection Rate by Product Category"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Explore Customer Complaints")

    selected_complaint = st.selectbox(
        "Choose complaint category",
        complaint_columns
    )

    filtered = complaints_df[
        complaints_df[selected_complaint] == 1
    ]

    st.metric(
        "Matching Reviews",
        len(filtered)
    )

    st.dataframe(
        filtered[
            [
                "Class Name",
                "Review Text",
                "complaints",
                "Rating"
            ]
        ],
        use_container_width=True
    )
