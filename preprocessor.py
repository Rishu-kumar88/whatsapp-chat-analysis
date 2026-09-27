import re
import pandas as pd
from dateutil import parser


def preprocess(data: str) -> pd.DataFrame:
    """
    Robust WhatsApp export preprocessor supporting Android and iOS formats,
    12-hour/24-hour clock formats, multi-line messages, and varied date separators.
    """
    if not data or not isinstance(data, str):
        return pd.DataFrame()

    # Normalize unicode spaces and hidden control characters
    data = data.replace('\u202f', ' ').replace('\xa0', ' ').replace('\u200e', '')

    # Common WhatsApp date-time header regex patterns
    patterns = [
        # Android 12-hr: 18/09/24, 12:58 PM -
        (r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)\s-\s', '%m/%d/%y, %I:%M %p - '),
        # Android 24-hr: 18/09/24, 14:58 -
        (r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s', '%m/%d/%y, %H:%M - '),
        # Dot date 12-hr/24-hr: 18.09.24, 14:58 -
        (r'\d{1,2}\.\d{1,2}\.\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\s-\s', None),
        # Hyphen date: 18-09-24, 14:58 -
        (r'\d{1,2}-\d{1,2}-\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\s-\s', None),
        # iOS 12/24-hr bracketed: [18/09/24, 12:58:32 PM] or [18.09.24, 14:58:32]
        (r'\[\d{1,2}[\/.\-]\d{1,2}[\/.\-]\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\]\s', None),
    ]

    matched_pattern = None
    date_strings = []
    message_bodies = []

    for pattern, fmt in patterns:
        matches = re.findall(pattern, data)
        if len(matches) > 0:
            matched_pattern = pattern
            date_strings = matches
            message_bodies = re.split(pattern, data)[1:]
            break

    if not matched_pattern or len(date_strings) == 0:
        # Fallback regex matching general timestamp lines
        generic_pattern = r'(\[\d{1,4}[.\/-]\d{1,2}[.\/-]\d{1,4}.*?\]|\d{1,4}[.\/-]\d{1,2}[.\/-]\d{1,4}.*?-\s)'
        date_strings = re.findall(generic_pattern, data)
        message_bodies = re.split(generic_pattern, data)[1:]

    if len(date_strings) == 0:
        return pd.DataFrame()

    df = pd.DataFrame({
        "date_raw": date_strings[:len(message_bodies)],
        "message_raw": message_bodies
    })

    # Clean raw date strings (strip trailing separators, brackets)
    clean_dates = []
    for d in df["date_raw"]:
        cleaned = d.strip()
        cleaned = re.sub(r'^[\[\s]+|[\]\s\-]+$', '', cleaned)
        clean_dates.append(cleaned)

    # Parse timestamps cleanly
    parsed_dates = []
    for d in clean_dates:
        try:
            parsed_dates.append(parser.parse(d, dayfirst=True))
        except Exception:
            parsed_dates.append(pd.NaT)

    df["date"] = parsed_dates
    df = df.dropna(subset=["date"]).copy()

    users = []
    texts = []

    for msg in df["message_raw"]:
        msg_str = msg.strip()
        # Match "User Name: Message content"
        match = re.match(r'^(.*?):\s(.*)', msg_str, re.DOTALL)
        if match:
            users.append(match.group(1).strip())
            texts.append(match.group(2).strip())
        else:
            users.append("group_notification")
            texts.append(msg_str)

    df["user"] = users
    df["message"] = texts

    # Feature Engineering
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["months"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["day_name"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    
    # Hour period representation (e.g. 14-15)
    periods = []
    for h in df["hour"]:
        if h == 23:
            periods.append("23-00")
        elif h == 0:
            periods.append("00-01")
        else:
            periods.append(f"{h:02d}-{(h+1):02d}")
    df["period"] = periods

    # Message stats
    df["word_count"] = df["message"].apply(lambda m: len(m.split()))
    df["char_count"] = df["message"].apply(len)
    
    # Media and Link detection
    media_tokens = ['<media omitted>', 'photo.jpg (file attached)', 'video.mp4 (file attached)', 
                    'audio.opus (file attached)', 'document (file attached)', '<media omitted\n>']
    df["is_media"] = df["message"].apply(
        lambda m: any(token in m.lower() for token in media_tokens)
    )

    df.drop(columns=["date_raw", "message_raw"], inplace=True, errors="ignore")

    return df
