import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import preprocessor
import helper

# Page Configuration
st.set_page_config(
    page_title="WhatsApp Chat Intelligence & NLP Dashboard",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #1E88E5;
    }
    .privacy-box {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 10px 15px;
        border-radius: 5px;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">💬 WhatsApp Chat Intelligence & NLP Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-grade Data Science & NLP analytics platform for exported WhatsApp chat logs</div>', unsafe_allow_html=True)

st.markdown("""
<div class="privacy-box">
    🔒 <b>Data Privacy First:</b> Chat files are processed strictly in-memory. Zero data is recorded or stored on disk or server.
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.title("🛠️ Configuration")

uploaded_file = st.sidebar.file_uploader("Upload WhatsApp Chat Export (.txt)", type=["txt"])
use_sample = st.sidebar.checkbox("Or use Demo Sample Data", value=False)

data_content = None

if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.getvalue()
        data_content = bytes_data.decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
elif use_sample:
    data_content = helper.get_sample_chat()
    st.sidebar.info("Using built-in sample chat dataset.")

if data_content is None:
    st.info("👈 Please upload a WhatsApp exported chat `.txt` file or check **'Or use Demo Sample Data'** in the sidebar to view the interactive dashboard!")
    
    # Render Preview Features Showcase
    st.markdown("### 🌟 What this platform analyzes:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📊 Timeline & Heatmaps")
        st.write("Monthly message trends, hourly activity heatmaps, and peak communication periods.")
    with col2:
        st.markdown("#### 🎭 VADER Sentiment & NLP")
        st.write("Mood classification over time, sentiment distributions, and word/n-gram frequencies.")
    with col3:
        st.markdown("#### 🕸️ Interaction Matrix")
        st.write("Social network reply pairs, user contribution ratios, and emoji analytics.")
else:
    df = preprocessor.preprocess(data_content)

    if df.empty:
        st.error("⚠️ Failed to parse the chat file. Please verify the exported file contains standard WhatsApp timestamps.")
    else:
        # Fetch Unique Users
        user_list = df['user'].unique().tolist()
        if 'group_notification' in user_list:
            user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, 'overall')

        selected_user = st.sidebar.selectbox("Select User Perspective", user_list)

        st.sidebar.markdown("---")
        st.sidebar.caption(f"📁 Parsed Messages: `{len(df):,}`")
        st.sidebar.caption(f"📅 Date Range: `{df['date'].min().strftime('%d %b %Y')} - {df['date'].max().strftime('%d %b %Y')}`")

        # Tabs Navigation
        tab_overview, tab_users, tab_words, tab_sentiment, tab_network, tab_resume = st.tabs([
            "📊 Executive Overview",
            "👥 User Activity",
            "🔤 Words & Emojis",
            "🎭 Sentiment Analysis",
            "🕸️ Interactions",
            "💼 Portfolio & Resume"
        ])

        # ==========================================
        # TAB 1: EXECUTIVE OVERVIEW
        # ==========================================
        with tab_overview:
            num_messages, words, num_media, num_urls, avg_words = helper.fetch_stats(selected_user, df)

            st.subheader("📌 Key Performance Indicators")
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("Total Messages", f"{num_messages:,}")
            kpi2.metric("Total Words", f"{words:,}")
            kpi3.metric("Media Shared", f"{num_media:,}")
            kpi4.metric("Links Shared", f"{num_urls:,}")
            kpi5.metric("Avg Words / Msg", f"{avg_words}")

            st.markdown("---")

            # Timelines
            col_t1, col_t2 = st.columns(2)

            with col_t1:
                st.subheader("📈 Monthly Message Volume")
                timeline = helper.monthly_timeline(selected_user, df)
                fig_m = px.line(
                    timeline, x='time', y='message',
                    markers=True,
                    labels={'time': 'Month-Year', 'message': 'Messages'},
                    color_discrete_sequence=['#1E88E5']
                )
                fig_m.update_layout(xaxis_tickangle=-45, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_m, use_container_width=True)

            with col_t2:
                st.subheader("📅 Daily Timeline Trend")
                daily = helper.daily_timeline(selected_user, df)
                fig_d = px.area(
                    daily, x='date', y='message_count',
                    labels={'date': 'Date', 'message_count': 'Messages'},
                    color_discrete_sequence=['#00ACC1']
                )
                fig_d.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_d, use_container_width=True)

            # Activity Maps
            st.subheader("⏰ Peak Activity Patterns")
            col_a1, col_a2 = st.columns(2)

            busy_day, busy_month = helper.activity_maps(selected_user, df)

            with col_a1:
                st.markdown("##### Most Active Days of Week")
                fig_day = px.bar(
                    x=busy_day.index, y=busy_day.values,
                    labels={'x': 'Day of Week', 'y': 'Message Count'},
                    color=busy_day.values, color_continuous_scale='Blues'
                )
                fig_day.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=20))
                st.plotly_chart(fig_day, use_container_width=True)

            with col_a2:
                st.markdown("##### Most Active Months")
                fig_month = px.bar(
                    x=busy_month.index, y=busy_month.values,
                    labels={'x': 'Month', 'y': 'Message Count'},
                    color=busy_month.values, color_continuous_scale='Purples'
                )
                fig_month.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=20))
                st.plotly_chart(fig_month, use_container_width=True)

            st.markdown("##### 🌡️ Hourly Activity Heatmap (Day of Week vs Time Slot)")
            heatmap_data = helper.activity_heatmap(selected_user, df)
            if not heatmap_data.empty:
                fig_hm = px.imshow(
                    heatmap_data,
                    labels=dict(x="Hour Slot", y="Day of Week", color="Messages"),
                    color_continuous_scale="Viridis",
                    aspect="auto"
                )
                st.plotly_chart(fig_hm, use_container_width=True)

        # ==========================================
        # TAB 2: USER ACTIVITY
        # ==========================================
        with tab_users:
            if selected_user == 'overall':
                st.subheader("👥 User Participation Benchmark")
                top_users, percent_df = helper.most_busy_users(df)

                col_u1, col_u2 = st.columns(2)

                with col_u1:
                    st.markdown("##### Top Active Participants")
                    fig_u = px.bar(
                        top_users, x=top_users.index, y=top_users.values,
                        labels={'x': 'User', 'y': 'Message Count'},
                        color=top_users.values, color_continuous_scale='Tealgrn'
                    )
                    st.plotly_chart(fig_u, use_container_width=True)

                with col_u2:
                    st.markdown("##### Share of Total Conversation (%)")
                    fig_pie = px.pie(
                        percent_df.head(7), names='user', values='percent',
                        hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("##### Detailed Breakdown")
                st.dataframe(percent_df, use_container_width=True)
            else:
                st.info(f"Viewing user activity profile for **{selected_user}**. Switch to 'overall' to view full group comparisons.")
                user_df = df[df['user'] == selected_user]
                st.write(f"- Total Messages Sent: `{len(user_df)}`")
                st.write(f"- Total Media Shared: `{user_df['is_media'].sum()}`")
                st.write(f"- Active Days Count: `{user_df['date'].dt.date.nunique()}`")

        # ==========================================
        # TAB 3: WORDS & EMOJIS
        # ==========================================
        with tab_words:
            col_w1, col_w2 = st.columns(2)

            with col_w1:
                st.subheader("☁️ Word Cloud")
                wc = helper.create_wordcloud(selected_user, df)
                fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis("off")
                st.pyplot(fig_wc)

            with col_w2:
                st.subheader("📊 Most Frequent Words")
                most_common_df = helper.most_common_words(selected_user, df)
                if not most_common_df.empty:
                    fig_mc = px.bar(
                        most_common_df.sort_values(by='count'),
                        x='count', y='word', orientation='h',
                        color='count', color_continuous_scale='Sunset'
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)

            st.markdown("---")
            col_n1, col_n2 = st.columns(2)

            with col_n1:
                st.subheader("🔤 Top Bigrams (2-Word Combinations)")
                bigrams = helper.get_ngrams(selected_user, df, n=2, top_n=8)
                if not bigrams.empty:
                    fig_bg = px.bar(bigrams, x='count', y='ngram', orientation='h', color_discrete_sequence=['#42A5F5'])
                    st.plotly_chart(fig_bg, use_container_width=True)
                else:
                    st.write("Insufficient word data for bigram extraction.")

            with col_n2:
                st.subheader("😀 Emoji Usage Analysis")
                emoji_df = helper.emoji_analysis(selected_user, df)
                if not emoji_df.empty:
                    fig_e = px.pie(
                        emoji_df.head(6), names='emoji', values='count',
                        title="Top Emojis Share", hole=0.3
                    )
                    st.plotly_chart(fig_e, use_container_width=True)
                else:
                    st.write("No emojis detected in chat data.")

        # ==========================================
        # TAB 4: SENTIMENT ANALYSIS
        # ==========================================
        with tab_sentiment:
            st.subheader("🎭 VADER Natural Language Sentiment Analysis")
            sentiment_dist, monthly_sentiment, user_sentiment = helper.analyze_sentiment(selected_user, df)

            if not sentiment_dist.empty:
                col_s1, col_s2 = st.columns(2)

                with col_s1:
                    st.markdown("##### Sentiment Category Distribution")
                    color_map = {'Positive': '#66BB6A', 'Neutral': '#FFA726', 'Negative': '#EF5350'}
                    fig_s_pie = px.pie(
                        sentiment_dist, names='category', values='count',
                        color='category', color_discrete_map=color_map, hole=0.4
                    )
                    st.plotly_chart(fig_s_pie, use_container_width=True)

                with col_s2:
                    st.markdown("##### Monthly Mood Trend (Average Sentiment Score)")
                    if not monthly_sentiment.empty:
                        fig_s_line = px.line(
                            monthly_sentiment, x='time', y='sentiment_score',
                            markers=True, labels={'time': 'Month-Year', 'sentiment_score': 'Avg Sentiment (-1 to +1)'},
                            color_discrete_sequence=['#AB47BC']
                        )
                        fig_s_line.add_hline(y=0, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig_s_line, use_container_width=True)

                if selected_user == 'overall' and not user_sentiment.empty:
                    st.markdown("##### 🏆 User Sentiment Ranking (Most Positive Participants)")
                    fig_u_sent = px.bar(
                        user_sentiment, x='user', y='avg_sentiment',
                        color='avg_sentiment', color_continuous_scale='RdYlGn',
                        labels={'avg_sentiment': 'Average Sentiment Score', 'user': 'Participant'}
                    )
                    st.plotly_chart(fig_u_sent, use_container_width=True)
            else:
                st.write("Insufficient data for sentiment classification.")

        # ==========================================
        # TAB 5: INTERACTIONS & NETWORK
        # ==========================================
        with tab_network:
            st.subheader("🕸️ User Reply Interaction Matrix")
            matrix = helper.interaction_matrix(df)
            if not matrix.empty:
                st.write("This matrix shows how often one participant sends a message immediately after another participant:")
                fig_mat = px.imshow(
                    matrix,
                    labels=dict(x="Replied To (User B)", y="Sender (User A)", color="Replies"),
                    color_continuous_scale="Plasma", aspect="auto"
                )
                st.plotly_chart(fig_mat, use_container_width=True)
            else:
                st.info("Interaction matrix requires multi-user conversation data.")

        # ==========================================
        # TAB 6: PORTFOLIO & RESUME INSIGHTS
        # ==========================================
        with tab_resume:
            st.subheader("💼 Resume & Portfolio Optimization Guide")
            st.markdown("""
            Here is how you can present this project on your **Resume** and during **Data Science Interviews**:

            #### 📄 Bullet Points for Your CV:
            - **WhatsApp Analytics & NLP Intelligence Platform** | *Python, Streamlit, VADER, NLTK, Plotly, Pandas*
              - Engineered a universal regex-driven WhatsApp chat parser handling multi-OS (Android/iOS) and 12h/24h timestamp variants.
              - Implemented VADER sentiment analysis and N-gram frequency algorithms to extract mood trends and conversation topics across time.
              - Built interactive Plotly dashboards featuring activity heatmaps, user interaction matrices, and emoji distributions.
              - Ensured data privacy through 100% in-memory data processing with zero server retention.

            #### 🎤 Interview Talking Points:
            1. **Data Engineering Challenge:** *"Handling varied datetime formats across WhatsApp exports (Android vs iOS, 12h vs 24h, hidden unicode spaces like `\\u202f`) required robust regex fallback pipelines."*
            2. **NLP Implementation:** *"Used VADER for fast sentiment scoring over conversational text, providing mood trends over months without requiring heavy GPU infrastructure."*
            3. **User Experience & Performance:** *"Transitioned static Matplotlib plots into responsive Plotly charts, organized into modular Streamlit tabs."*
            """)
