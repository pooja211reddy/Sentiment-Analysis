<p align="center">
  <img src="assets/banner_new.png" alt="AI Sentiment Analysis Banner" width="100%"/>
</p>

# 🧠 Sentiment Analysis with LSTM, ULMFiT, and BERT

A comparative NLP project for **binary sentiment classification** using the IMDb Movie Reviews dataset.  
This project trains, evaluates, compares, and deploys three models:

- Custom **LSTM** using PyTorch
- Pretrained **AWD-LSTM using ULMFiT**
- Fine-tuned **BERT-base-uncased**

The final model is deployed through a real-time **Streamlit Sentiment Analysis App**.

---

## 🚀 Project Highlights

- End-to-end sentiment analysis pipeline
- Text preprocessing, tokenization, and padding
- Accuracy, Precision, Recall, F1-score comparison
- Convergence and generalization analysis
- Training vs validation loss visualization
- Confusion matrix visualization
- Real-time Streamlit UI for predictions

---

## 📌 Dataset

**IMDb Movie Reviews Dataset**

| Property | Description |
|---|---|
| Source | Stanford AI Lab / Hugging Face Datasets |
| Task | Binary Sentiment Classification |
| Classes | Negative, Positive |
| Size | 50,000 labeled reviews |
| Train Samples | 25,000 |
| Test Samples | 25,000 |

---

## 🧠 Models Implemented

### 1. Custom LSTM

A manually built PyTorch LSTM model used as a baseline.

**Key characteristics:**

- Learns embeddings from scratch
- Sequential text processing
- Requires multiple epochs
- Moderate generalization

---

### 2. AWD-LSTM with ULMFiT

A transfer-learning based recurrent model using FastAI.

**Key characteristics:**

- Uses pretrained AWD-LSTM architecture
- Trained using ULMFiT methodology
- Uses gradual fine-tuning
- Performs better than basic LSTM

---

### 3. BERT-base-uncased

A transformer-based model fine-tuned for sentiment classification.

**Key characteristics:**

- Uses WordPiece tokenization and supervised fine-tuning
- Bidirectional contextual understanding
- Fast convergence
- Best overall performance

---

## ⚙️ Hyperparameters

| Model | Learning Rate | Batch Size | Epochs | Sequence Length | Dropout | Optimizer |
|---|---:|---:|---:|---:|---:|---|
| LSTM | 0.001 | 32 | 5 | 50 | 0.5 | Adam |
| ULMFiT | 2e-2 / 1e-2 | 4 | 3 | 50 | 0.7 | FastAI Adam |
| BERT | 2e-5 | 8 | 1 | 128 | Default | AdamW |

---

## 📊 Final Results

| Model | Accuracy | Precision | Recall | F1 Score | Convergence | Generalization |
|---|---:|---:|---:|---:|---|---|
| LSTM | 0.824 | 0.852 | 0.783 | 0.816 | Slow (5 epochs) | 0.165 (Poor) |
| ULMFiT | 0.825 | 0.796 | 0.874 | 0.833 | Medium (3 epochs) | 0.139 (Poor) |
| BERT | **0.887** | **0.882** | **0.893** | **0.888** | **Fast (1 epoch)** | **0.019 (Very Good)** |

---

## 📈 Key Findings

- **BERT achieved the best performance** across all evaluation metrics.
- **ULMFiT improved over LSTM**, showing the benefit of transfer learning.
- **LSTM showed slower convergence** and a higher generalization gap.
- **BERT generalized best**, with the lowest train-validation loss gap.
- Transformer-based architectures are more effective for sentiment classification than traditional recurrent models.

---

## 📉 Training Behavior

### LSTM
- Training loss decreased steadily.
- Validation loss fluctuated after later epochs.
- Indicates mild overfitting.

### ULMFiT
- Showed good initial learning.
- Validation loss increased in the final epoch.
- Indicates instability / overfitting.

### BERT
- Achieved low validation loss quickly.
- Minimal generalization gap.
- Best convergence and stability.

---

## 🖼️ Visualizations

The project includes:

- Training vs Validation Loss curves
- Model comparison bar chart
- Confusion matrix
- Normalized confusion matrix
- Accuracy / Precision / Recall / F1 comparison
<p align="center">
  <img src="assets/output.png" width="700"/>
</p>
---

## 💬 Streamlit App

A real-time sentiment prediction app is included.

### Features

- Clean modern UI
- Text input box
- Real-time sentiment prediction
- Positive / Negative classification
- Confidence score
- BERT-powered inference

---
## 💻 Example Predictions

| Input Text                          | Prediction        |
|------------------------------------|-------------------|
| "I love this product!"             | 😊 **Positive**   |
| "This is the worst experience ever." | 😡 **Negative**   |
| "Not bad, could be better."        | 😐 **Neutral-ish** |


---
## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
