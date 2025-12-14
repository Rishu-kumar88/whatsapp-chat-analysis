import re
import pandas as pd


def preprocess(data):

    # Match: 9/18/24, 12:58 AM -
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}[ \u202f](?:AM|PM)\s-\s'

    # Split on each message start
    messages = re.split(pattern, data)[1:]
    date_raw = re.findall(pattern, data)

    # Clean any weird spaces in dates
    dates = [d.replace('\u202f', ' ').replace('\xa0', ' ') for d in date_raw]

    df = pd.DataFrame({
        "message_raw": messages,
        "dates": dates
    })

    users = []
    texts = []

    for msg in df["message_raw"]:
        # Match "user: message" (full name before first colon)
        m = re.match(r'^(.*?):\s(.*)', msg, re.DOTALL)

        if m:
            users.append(m.group(1))   # full user name
            texts.append(m.group(2))   # message text
        else:
            users.append("group_notification")
            texts.append(msg)

    df["user"] = users
    df["message"] = texts
    df.drop(columns=["message_raw"], inplace=True)

    # Parse datetime like: 9/18/24, 12:58 AM -
    df["dates_of_message"] = pd.to_datetime(
        df["dates"],
        format="%m/%d/%y, %I:%M %p - ",
        errors="coerce"
    )

    df.drop(columns=["dates"], inplace=True)

    df["year"] = df["dates_of_message"].dt.year
    df["months"] = df["dates_of_message"].dt.month_name()
    df['day_name'] = df['dates_of_message'].dt.day_name()
    df['month_num'] = df['dates_of_message'].dt.month
    df["day"] = df["dates_of_message"].dt.day
    df["hour"] = df["dates_of_message"].dt.hour
    df["minute"] = df["dates_of_message"].dt.minute

    return df
