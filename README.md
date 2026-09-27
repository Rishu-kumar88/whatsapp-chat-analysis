# 💬 WhatsApp Chat Intelligence & NLP Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CI Pipeline](https://github.com/Rishu-kumar88/whatsapp-chat-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Rishu-kumar88/whatsapp-chat-analysis/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Data Science, NLP, and Behavioral Analytics web application that transforms raw WhatsApp chat exports into interactive dashboards with VADER sentiment scoring, N-gram frequency extraction, user interaction matrices, and activity heatmaps.

---

## 🌟 Key Features

- ⚙️ **Universal Multi-Format Parser**: Robust regex fallback engine supporting Android & iOS 12-hour and 24-hour timestamp formats, varied date separators (`.`, `/`, `-`), multi-line messages, and narrow non-breaking space characters (`\u202f`).
- 🎭 **VADER NLP Sentiment Engine**: Evaluates compound sentiment scores over time, classifies positive/neutral/negative messages, and ranks participant positivity.
- 📊 **Interactive Plotly Visualizations**: Responsive time-series plots, daily activity trends, and day-of-week vs. hour-of-day heatmaps.
- 🔤 **N-Gram & Emoji Analytics**: Bigram/trigram extraction and emoji distribution calculations using custom Hinglish & English stopword filters.
- 🕸️ **User Reply Interaction Matrix**: Analyzes reply pairs between chat participants to map conversation flow.
- 🔒 **Data Privacy First**: 100% in-memory data processing with zero server or disk storage.

---

## 📐 Project Architecture

```
whatsapp-chat-analysis/
├── app.py                   # Streamlit interactive UI entry point
├── preprocessor.py          # Multi-pattern regex date parser & feature engineering
├── helper.py                # Statistics, VADER NLP, Plotly graph generators, and sample data
├── stop_hinglish.txt        # Custom Hinglish & English stopword lexicon
├── requirements.txt         # Production dependency specifications
├── tests/                   # Automated Pytest suite
│   ├── test_preprocessor.py
│   └── test_helper.py
└── .github/workflows/       # GitHub Actions CI pipeline
    └── ci.yml
```

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.9+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Rishu-kumar88/whatsapp-chat-analysis.git
cd whatsapp-chat-analysis
```

### 2. Set up virtual environment & install dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Streamlit application
```bash
streamlit run app.py
```

---

## 🧪 Running Automated Tests

Run the full pytest suite to verify parser and analytics integrity:
```bash
pytest -v
```

---

## 💼 CV / Resume Bullet Points

> **WhatsApp Chat Intelligence & NLP Dashboard** | *Python, Streamlit, VADER NLP, Plotly, Pandas, Pytest*
> - Engineered a universal regex parser handling multi-OS (Android/iOS) and 12h/24h timestamp variations across custom export formats.
> - Integrated VADER sentiment analysis and N-gram frequency algorithms to extract mood trends and conversation topics over time.
> - Built interactive Plotly dashboards featuring activity heatmaps, user interaction matrices, and emoji usage metrics.
> - Implemented unit test coverage with Pytest and continuous integration via GitHub Actions.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more details.
