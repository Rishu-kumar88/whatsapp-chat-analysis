import pytest
import pandas as pd
from preprocessor import preprocess


def test_preprocess_android_12hr():
    raw_data = """18/09/24, 12:58 PM - Alice: Hello World!
18/09/24, 12:59 PM - Bob: Hi Alice, how are you?
18/09/24, 01:00 PM - Alice: <Media omitted>
"""
    df = preprocess(raw_data)
    assert not df.empty
    assert len(df) == 3
    assert set(df['user'].unique()) == {'Alice', 'Bob'}
    assert df['is_media'].sum() == 1


def test_preprocess_android_24hr():
    raw_data = """18/09/2024, 14:30 - Charlie: Meeting at 3 PM
18/09/2024, 14:32 - Dave: Got it!
"""
    df = preprocess(raw_data)
    assert not df.empty
    assert len(df) == 2
    assert 'Charlie' in df['user'].values
    assert 'Dave' in df['user'].values


test_ios_data = """[18.09.24, 14:58:32] Eve: Testing iOS format
[18.09.24, 14:59:01] Frank: Works fine!
"""

def test_preprocess_ios():
    df = preprocess(test_ios_data)
    assert not df.empty
    assert len(df) == 2
    assert 'Eve' in df['user'].values


def test_preprocess_empty():
    df = preprocess("")
    assert df.empty
