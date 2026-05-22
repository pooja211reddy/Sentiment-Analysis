import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# ========================
# Load Model
# ========================

model_path = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

# ========================
# Custom CSS (UI Styling)
# ========================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
}
.main {
    background: transparent;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: white;
}
.subtitle {
    text-align: center;
    color: #ddd;
    margin-bottom: 30px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
.positive {
    background-color: #d4edda;
    color: #155724;
}
.negative {
    background-color: #f8d7da;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

# ========================
# Prediction Function
# ========================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs.max().item()

    return pred, confidence

# ========================
# UI
# ========================
st.markdown('<div class="big-title">💬 AI Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by BERT 🤖 | Real-time Prediction</div>', unsafe_allow_html=True)

# Input box
user_input = st.text_area("✍️ Enter your text here:", height=150)

# Button
if st.button("🚀 Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        pred, confidence = predict(user_input)

        if pred == 1:
            st.markdown(
                f'<div class="result-box positive">😊 Positive<br>Confidence: {confidence:.2f}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box negative">😡 Negative<br>Confidence: {confidence:.2f}</div>',
                unsafe_allow_html=True
            )

# ========================
# Sidebar (Extra Features)
# ========================
st.sidebar.title("🧠 About Model")
st.sidebar.write("""
- Model: BERT
- Task: Sentiment Analysis
- Classes: Positive / Negative
""")

st.sidebar.title("💡 Try Examples")
if st.sidebar.button("Positive Example"):
    st.write("I absolutely love this product!")

if st.sidebar.button("Negative Example"):
    st.write("This is the worst experience ever.")

# Footer
st.markdown("---")
st.markdown("✨ Built with Streamlit | NLP Project 🚀")
