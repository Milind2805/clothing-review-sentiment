# app.py
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Clothing Review Sentiment", page_icon="🧵")

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="MilindSingh316/distilbert-clothing-review-sentiment",
        top_k=None
    )

pipe = load_model()

st.title("Clothing Review Sentiment Classifier")
st.write("Paste a clothing review below to see the predicted sentiment.")

text = st.text_area("Review text", height=150,
                    placeholder="e.g. Loved the fabric, but it ran a size small...")

if st.button("Predict") and text.strip():
    results = pipe(text)[0]
    results = sorted(results, key=lambda x: -x["score"])

    top = results[0]
    st.subheader(f"Prediction: {top['label'].capitalize()} ({top['score']:.1%} confidence)")

    for r in results:
        st.write(f"{r['label'].capitalize()}: {r['score']:.1%}")
        st.progress(r["score"])