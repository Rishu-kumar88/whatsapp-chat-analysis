import os
import re
from collections import Counter
import pandas as pd
import numpy as np
from urlextract import URLExtract
from wordcloud import WordCloud
import emoji
from sklearn.feature_extraction.text import CountVectorizer


def fetch_stats(selected_user: str, df: pd.DataFrame):
    """Fetch high-level statistics for selected user or overall chat."""
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]

    # Filter out system notifications and media tokens for word calculation
    clean_messages = df[~df['user'].isin(['group_notification']) & ~df['is_media']]['message']

    words = []
    for msg in clean_messages:
        words.extend(msg.split())

    num_media_messages = df[df['is_media']].shape[0]

    # Links extraction
    extractor = URLExtract()
    links = []
    for msg in df['message']:
        try:
            links.extend(extractor.find_urls(msg))
        except Exception:
            pass

    avg_words_per_msg = round(len(words) / num_messages, 1) if num_messages > 0 else 0

    return num_messages, len(words), num_media_messages, len(links), avg_words_per_msg


def most_busy_users(df: pd.DataFrame):
    """Return top active users and percentage contribution dataframe."""
    filtered_df = df[df['user'] != 'group_notification']
    if filtered_df.empty:
        return pd.Series(dtype=int), pd.DataFrame(columns=['user', 'percent'])

    user_counts = filtered_df['user'].value_counts()
    top_users = user_counts.head(10)
    percent_df = pd.DataFrame({
        'user': user_counts.index,
        'percent': np.round((user_counts.values / filtered_df.shape[0]) * 100, 2)
    })
    return top_users, percent_df


def load_stopwords(stop_words_path: str = 'stop_hinglish.txt') -> set:
    """Load stopwords from file and merge with default English stopwords."""
    stop_words = set()
    if os.path.exists(stop_words_path):
        with open(stop_words_path, 'r', encoding='utf-8') as f:
            stop_words = set(f.read().splitlines())
    
    # Common system/media tokens
    stop_words.update(['media', 'omitted', 'null', 'message', 'deleted', 'attached', 'photo', 'video', 'audio'])
    return stop_words


def create_wordcloud(selected_user: str, df: pd.DataFrame, stop_words_path: str = 'stop_hinglish.txt'):
    """Generate WordCloud for selected user."""
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    stop_words = load_stopwords(stop_words_path)

    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['is_media']]

    def remove_stopwords(text):
        words = [w.lower() for w in text.split() if w.lower() not in stop_words and len(w) > 2]
        return " ".join(words)

    clean_text = temp['message'].apply(remove_stopwords).str.cat(sep=" ")
    if not clean_text.strip():
        clean_text = "No Sufficient Data"

    wc = WordCloud(
        width=800,
        height=400,
        min_font_size=10,
        background_color='white',
        colormap='viridis'
    )
    return wc.generate(clean_text)


def most_common_words(selected_user: str, df: pd.DataFrame, top_n: int = 20, stop_words_path: str = 'stop_hinglish.txt') -> pd.DataFrame:
    """Extract most frequent non-stop words."""
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    stop_words = load_stopwords(stop_words_path)
    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['is_media']]

    words = []
    for msg in temp['message']:
        for word in msg.lower().split():
            clean_w = re.sub(r'[^\w\s]', '', word)
            if clean_w and clean_w not in stop_words and len(clean_w) > 2:
                words.append(clean_w)

    most_common_df = pd.DataFrame(Counter(words).most_common(top_n), columns=['word', 'count'])
    return most_common_df


def get_ngrams(selected_user: str, df: pd.DataFrame, n: int = 2, top_n: int = 10) -> pd.DataFrame:
    """Extract top N-grams (Bigrams / Trigrams)."""
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    stop_words = list(load_stopwords())
    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['is_media']]

    texts = temp['message'].dropna().tolist()
    if len(texts) < 5:
        return pd.DataFrame(columns=['ngram', 'count'])

    try:
        cv = CountVectorizer(ngram_range=(n, n), stop_words=stop_words, max_features=top_n)
        ngram_matrix = cv.fit_transform(texts)
        counts = ngram_matrix.sum(axis=0).A1
        words = cv.get_feature_names_out() if hasattr(cv, 'get_feature_names_out') else cv.get_feature_names()
        ngram_df = pd.DataFrame(list(zip(words, counts)), columns=['ngram', 'count']).sort_values(by='count', ascending=False)
        return ngram_df
    except Exception:
        return pd.DataFrame(columns=['ngram', 'count'])


def emoji_analysis(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Extract and count emojis used in messages."""
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for msg in df['message']:
        for char in msg:
            if emoji.is_emoji(char):
                emojis.append(char)

    emoji_counts = Counter(emojis)
    emoji_df = pd.DataFrame(emoji_counts.most_common(), columns=['emoji', 'count'])
    total_emojis = emoji_df['count'].sum() if not emoji_df.empty else 1
    emoji_df['percentage'] = np.round((emoji_df['count'] / total_emojis) * 100, 2)

    return emoji_df


def monthly_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Generate monthly time-series message counts."""
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'months']).size().reset_index(name='message')
    timeline['time'] = timeline.apply(lambda r: f"{r['months'][:3]}-{r['year']}", axis=1)
    timeline = timeline.sort_values(by=['year', 'month_num'])
    return timeline


def daily_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Generate daily time-series message counts."""
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    df['only_date'] = df['date'].dt.date
    daily = df.groupby('only_date').size().reset_index(name='message_count')
    daily.columns = ['date', 'message_count']
    return daily


def activity_maps(selected_user: str, df: pd.DataFrame):
    """Return day-of-week and month activity counts."""
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    busy_day = df['day_name'].value_counts()
    busy_month = df['months'].value_counts()

    # Reorder by standard calendar order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    busy_day = busy_day.reindex([d for d in days_order if d in busy_day.index])

    return busy_day, busy_month


def activity_heatmap(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Generate Pivot table for Day of week vs Hour period."""
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    pivot = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    existing_days = [d for d in days_order if d in pivot.index]
    pivot = pivot.loc[existing_days]
    return pivot


def get_sentiment_analyzer():
    """Returns SentimentIntensityAnalyzer or fast rule-based fallback."""
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        nltk.data.find('sentiment/vader_lexicon.zip')
        return SentimentIntensityAnalyzer()
    except Exception:
        class RuleBasedAnalyzer:
            def polarity_scores(self, text):
                pos_words = {'good', 'great', 'awesome', 'happy', 'love', 'nice', 'thanks', 'haha', 'lol', 'yeah', 'wow', 'perfect'}
                neg_words = {'bad', 'sad', 'hate', 'angry', 'worst', 'sorry', 'no', 'stop', 'late', 'error'}
                words = set(text.lower().split())
                pos = len(words.intersection(pos_words))
                neg = len(words.intersection(neg_words))
                compound = 0.5 if pos > neg else (-0.5 if neg > pos else 0.0)
                return {'compound': compound, 'pos': pos, 'neg': neg, 'neu': 1}
        return RuleBasedAnalyzer()


def analyze_sentiment(selected_user: str, df: pd.DataFrame):
    """
    Perform sentiment analysis on chat messages.
    Returns overall sentiment distribution, monthly sentiment trend, and user sentiment ranking.
    """
    if selected_user != 'overall':
        df = df[df['user'] == selected_user].copy()
    else:
        df = df.copy()

    sia = get_sentiment_analyzer()

    # Exclude system notifications and media
    clean_df = df[~df['user'].isin(['group_notification']) & ~df['is_media']].copy()
    if clean_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    scores = clean_df['message'].apply(lambda m: sia.polarity_scores(m)['compound'])
    clean_df['sentiment_score'] = scores

    def classify_score(s):
        if s >= 0.05:
            return 'Positive'
        elif s <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    clean_df['sentiment_category'] = clean_df['sentiment_score'].apply(classify_score)

    # 1. Category Distribution
    counts = clean_df['sentiment_category'].value_counts()
    sentiment_dist = pd.DataFrame({'category': counts.index, 'count': counts.values})

    # 2. Monthly Trend
    clean_df['time'] = clean_df.apply(lambda r: f"{r['months'][:3]}-{r['year']}", axis=1)
    monthly_sentiment = clean_df.groupby(['year', 'month_num', 'time'])['sentiment_score'].mean().reset_index()
    monthly_sentiment = monthly_sentiment.sort_values(by=['year', 'month_num'])

    # 3. User Sentiment Leaderboard
    user_sentiment = clean_df.groupby('user')['sentiment_score'].agg(['mean', 'count']).reset_index()
    user_sentiment.columns = ['user', 'avg_sentiment', 'message_count']
    user_sentiment = user_sentiment[user_sentiment['message_count'] >= 1].sort_values(by='avg_sentiment', ascending=False)

    return sentiment_dist, monthly_sentiment, user_sentiment


def interaction_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute reply interaction counts between consecutive users in the chat."""
    filtered = df[df['user'] != 'group_notification'].copy()
    if len(filtered) < 2:
        return pd.DataFrame()

    filtered['next_user'] = filtered['user'].shift(-1)
    interactions = filtered[filtered['user'] != filtered['next_user']]
    matrix = interactions.groupby(['user', 'next_user']).size().unstack(fill_value=0)
    return matrix


def get_sample_chat() -> str:
    """Return realistic sample WhatsApp chat data for demonstration."""
    return """18/09/2024, 09:15 - Messages and calls are end-to-end encrypted. No one outside of this chat can read or listen to them.
18/09/2024, 09:16 - Alice: Hey team! Good morning! Are we still meeting for the AI project discussion today? 😊
18/09/2024, 09:18 - Bob: Morning Alice! Yes, absolutely. I've prepared the dataset and preliminary EDA charts.
18/09/2024, 09:20 - Charlie: Good morning guys! I might be 10 minutes late due to traffic, but I will join online.
18/09/2024, 09:22 - Alice: No worries Charlie! Check out the documentation here: https://github.com/Rishu-kumar88/whatsapp-chat-analysis
18/09/2024, 09:25 - Bob: <Media omitted>
18/09/2024, 09:26 - Bob: Here is the initial dashboard design screenshot! 📊
18/09/2024, 09:28 - Charlie: Wow, that looks amazing Bob! Great color scheme and interactive components.
18/09/2024, 09:30 - Alice: I agree! We should also add sentiment analysis and topic modeling using NLP.
18/09/2024, 09:35 - Bob: That will definitely make our portfolio project stand out for data science internships! 🚀
18/09/2024, 09:40 - Charlie: Let's do it! Meeting starts at 10:00 AM. See you all soon!
19/09/2024, 14:10 - Alice: Did everyone submit their pull requests on GitHub?
19/09/2024, 14:15 - Bob: Yes! Pytest unit tests passed with 100% coverage.
19/09/2024, 14:20 - Charlie: Perfect! Let's deploy it on Streamlit Community Cloud. 🎉
"""
