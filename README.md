# LSTM Sentiment Analysis (RNN/NLP)

An end-to-end **sentiment analysis system** built using **LSTM (RNN)** with **TensorFlow/Keras**, starting from raw text data and deployed as an interactive **Streamlit web application**.

This project focuses on **correct sequence modeling fundamentals**, including word-level tokenization, vocabulary control, padding & masking, and trainable embeddings.

---

## 📌 Project Overview

The goal of this project is to classify text reviews into **Positive** or **Negative** sentiment using a **Many-to-One LSTM architecture**.

Unlike toy examples, this project:
- Uses **raw text data** from a Kaggle sentiment dataset  
- Implements the **entire NLP pipeline manually**
- Separates **training (research)** from **deployment (product)**
- Is designed to be **resume-grade and interview-defensible**

---

## 🧠 Key Concepts Implemented

- Word-level tokenization
- Vocabulary size control and `<OOV>` handling
- Sequence padding and masking
- Trainable embedding layer
- LSTM for sequence modeling (Many-to-One)
- Binary classification with sigmoid output
- Model serialization and reuse
- Streamlit-based deployment

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **LSTM (Recurrent Neural Network)**
- **NumPy & Pandas**
- **Streamlit**

---

## 📂 Project Structure

```

lstm-sentiment-analysis-streamlit/
│
├── notebook/
│   └── sentiment_lstm_kaggle.ipynb   # Colab notebook (training & evaluation)
│
├── model/
│   ├── lstm_sentiment.h5             # Trained LSTM model
│   └── tokenizer.pkl                 # Saved tokenizer
│
├── app/
│   ├── app.py                        # Streamlit application
│   └── requirements.txt              # App dependencies
│
├── README.md
└── .gitignore

```

---

## 📊 Dataset

- **Source**: Kaggle Sentiment Analysis Dataset  
- **Type**: Raw text (CSV format)  
- **Labels**: Positive / Negative  
- **Why this dataset?**
  - Requires full NLP preprocessing
  - Better skill signal than pre-tokenized datasets
  - Suitable for demonstrating real-world NLP workflows

---

## 🏗️ Model Architecture

```

Input Text
↓
Word-level Tokenization
↓
Padding & Masking (fixed sequence length)
↓
Embedding Layer (trainable)
↓
LSTM (Many-to-One)
↓
Dense Layer (Sigmoid)
↓
Sentiment Probability

````

**Core design choices:**
- Vocabulary size: `20,000`
- Sequence length: `200`
- Embedding dimension: `128`
- LSTM units: `128`

---

## 🚀 Training Summary

- Loss Function: Binary Crossentropy  
- Optimizer: Adam  
- Batch Size: 32  
- Epochs: 5  

The model achieves strong and stable performance on held-out test data, demonstrating effective sequence learning without overfitting.

---

## 🌐 Streamlit Deployment

The trained model is deployed using **Streamlit**, allowing users to:
- Enter custom text reviews
- Receive sentiment predictions in real time
- View confidence scores

### Run Locally

```bash
cd app
streamlit run app.py
````

---

## 📌 Example Use Case

> *"The movie started slow but turned out to be emotionally powerful."*
> → **Positive**

> *"This was boring, predictable, and a waste of time."*
> → **Negative**

---

## 📄 Resume Bullet (Suggested)

> Built an end-to-end sentiment analysis system using LSTM on raw text data, implementing custom word-level tokenization, padding & masking, and trainable embeddings in TensorFlow/Keras; deployed the trained model as an interactive Streamlit web application.

---

## 🔮 Future Improvements

* GRU vs LSTM comparison
* Bidirectional LSTM
* Attention mechanism
* Error analysis & interpretability
* Cloud deployment (Streamlit Cloud)

---

## 👤 Author

Developed as a **resume-grade deep learning NLP project** focused on mastering **sequence modeling with RNNs/LSTMs**.

---

## 📜 License

This project is for educational and portfolio purposes.

```

---

### ✅ What this README does well
- Sounds **professional**, not inflated
- Clearly explains **what you did and why**
- Easy for recruiters to scan in 30–60 seconds
- Defensible in interviews (no buzzword lies)

If you want next, I can:
- Rewrite this for **academic tone**
- Shorten it for **hackathon style**
- Prepare **interview Q&A based on this README**

Just tell me.
```
