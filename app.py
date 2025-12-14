import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import seaborn as sns



st.sidebar.title("whatsapp chat analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")


    df=preprocessor.preprocess(data)


# fetch unique users

    user_list=df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,'overall')

    selected_user=st.sidebar.selectbox("show analysis wrt user list",user_list)

    if st.sidebar.button("Show Analysis"):

        num_messages,words,num_media_messages,num_urls=helper.fetch_stats(selected_user,df)

        st.title("Top Statistics")
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Number of Words")
            st.title(words)
        with col3:
            st.header("Number of Media Messages")
            st.title(num_media_messages)
        with col4:
            st.header("Number of URLs")
            st.title(num_urls)

# timeline analysis
        st.title("monthly time_line analysis")
        timeline=helper.monthly_timeline(selected_user,df)
        fig,ax=plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)


# activity map
        st.title("Activity Map")

        col1,col2=st.columns(2)
        with col1:
            st.title("most busy day")
            week_day=helper.day_name_count(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(week_day.index,week_day.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.title("most busy month")
            month = helper.month_activity_count(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(month.index, month.values, color='indigo')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
# heat map

        st.title("Heat Map")
        heatmap_data = helper.heat_map(selected_user, df)
        fig, ax = plt.subplots()
        sns.heatmap(heatmap_data)
        st.pyplot(fig)


# most busiest users

        if selected_user=='overall':
            x,new_df=helper.most_busy_users(df)

            fig,ax=plt.subplots()


            col1,col2=st.columns(2)
            with col1:
                st.title("Most busy users")
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.title("Data in percent")
                st.dataframe(new_df)
# word cloud
        st.title("Word Cloud")
        df_wc=helper.create_wordcloud(selected_user,df)
        fig,ax=plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

# top 20 most common words

        most_common_df=helper.most_common_word(selected_user,df)
        st.title("Most Common Words")
        fig,ax=plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1],color='violet')
        plt.xticks(rotation='vertical')

        st.pyplot(fig)


# emoji analysis

        emoji=helper.emoji_ana(selected_user,df)
        st.title("emoji analysis")

        col1,col2=st.columns(2)
        with col1:
            st.dataframe(emoji)

        with col2:
            fig,ax=plt.subplots()
            ax.pie(emoji[1].head(),labels=emoji[0].head(),autopct="%0.2f")
            st.pyplot(fig)