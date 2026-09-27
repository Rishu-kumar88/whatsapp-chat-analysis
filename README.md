# 💬 WhatsApp Chat Analyzer

A Python web application built using **Streamlit**, **Pandas**, **Plotly**, and **NLTK / VADER** to parse and analyze exported WhatsApp chat data.

---

## 📌 Features

- **General Statistics**: Total messages, words, media files, links shared, and average words per message.
- **Activity & Timelines**: Monthly/daily message trends, most active days/hours, and interactive activity heatmaps.
- **Top Users**: Breakdown of top contributors in group chats.
- **Word & Emoji Analytics**: Interactive WordCloud, top words (with Hinglish & English stopword filtering), bigrams/trigrams, and emoji distribution graphs.
- **Sentiment Analysis**: Sentiment classification (positive, neutral, negative) and average mood trends over time using VADER NLP.
- **Reply Matrix**: Interaction heatmap showing reply frequency between group members.

---

## 📂 Project Structure

```
whatsapp-chat-analysis/
├── app.py               # Streamlit web app interface
├── preprocessor.py      # WhatsApp chat log parser (Android & iOS formats)
├── helper.py            # Data calculations, sentiment analysis, and chart generation
├── stop_hinglish.txt    # Custom stopwords list (English & Hinglish)
├── requirements.txt     # Python dependencies
└── tests/               # Unit tests
    ├── test_preprocessor.py
    └── test_helper.py
```

---

## 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rishu-kumar88/whatsapp-chat-analysis.git
   cd whatsapp-chat-analysis
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

4. Open `http://localhost:8501` in your browser.

---

## 🧪 Testing

Run pytest to test the preprocessor and helper modules:
```bash
pytest
```
