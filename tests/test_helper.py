import pytest
import pandas as pd
from preprocessor import preprocess
from helper import fetch_stats, analyze_sentiment, emoji_analysis, get_sample_chat


@pytest.fixture
def sample_df():
    sample_raw = get_sample_chat()
    return preprocess(sample_raw)


def test_fetch_stats(sample_df):
    num_messages, words, num_media, num_urls, avg_words = fetch_stats('overall', sample_df)
    assert num_messages > 0
    assert words > 0
    assert num_media >= 1
    assert num_urls >= 1


def test_analyze_sentiment(sample_df):
    sentiment_dist, monthly_sentiment, user_sentiment = analyze_sentiment('overall', sample_df)
    assert not sentiment_dist.empty
    assert 'category' in sentiment_dist.columns


def test_emoji_analysis(sample_df):
    emoji_df = emoji_analysis('overall', sample_df)
    assert not emoji_df.empty
    assert 'emoji' in emoji_df.columns
    assert 'percentage' in emoji_df.columns
