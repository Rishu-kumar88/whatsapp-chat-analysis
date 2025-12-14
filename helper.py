def fetch_stats(selected_user,df):

    if selected_user != 'overall':
        df=df[df['user']==selected_user]

    num_messages=df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetching link shared

    from urlextract import URLExtract
    extract = URLExtract()


    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    x=df['user'].value_counts().head()
    new_df=round((df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns={'user':'name','count':'percent'})
    return x,new_df

from wordcloud import WordCloud

def create_wordcloud(selected_user,df):
    if selected_user != 'overall':
        df=df[df['user']==selected_user]

    wc=WordCloud(width=500,height=500,min_font_size=10,background_color='black')
    df_wc=wc.generate(df['message'].str.cat(sep=" "))
    return df_wc

import pandas as pd

def most_common_word(selected_user,df):

    if selected_user != 'overall':
        df=df[df['user']==selected_user]

    f = open('stop_hinglish.txt', 'r')
    stop_word = f.read()

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_word:
                words.append(word)

    from collections import Counter
    return_df=pd.DataFrame(Counter(words).most_common(20))
    return return_df


import emoji
from collections import Counter

def emoji_ana(selected_user,df):
    if selected_user != 'overall':
        df=df[df['user']==selected_user]

    emojis = []

    for message in df['message']:
        for c in message:
            if emoji.is_emoji(c):
                emojis.append(c)

    emojis_df=pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emojis_df

def monthly_timeline(selected_user,df):
    if selected_user != 'overall':
        df=df[df['user']==selected_user]

    timeline = df.groupby(['year', 'month_num', 'months']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['months'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time
    return timeline
def day_name_count(selected_user,df):
    if selected_user != 'overall':
        df=df[df['user']==selected_user]


    return df['day_name'].value_counts()

def month_activity_count(selected_user,df):
    if selected_user != 'overall':
        df=df[df['user']==selected_user]

    return df['months'].value_counts()

def heat_map(selected_user,df):
    if selected_user != 'overall':
        df=df[df['user']==selected_user]

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))

        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))

        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period
    heatmap_data = df.groupby(['day_name', 'period']).size().unstack(fill_value=0)

    return heatmap_data
