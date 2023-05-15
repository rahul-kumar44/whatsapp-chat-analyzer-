import nltk
import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# import plotly.graph_objs as go
nltk.download('vader_lexicon')

st. sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Object
    sentiments = SentimentIntensityAnalyzer()
# see here
    # Creating different columns for (Positive/Negative/Neutral)
    df["po"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]  # Positive
    df["ne"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]  # Negative
    df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]  # Neutral

# see here also
    # To indentify true sentiment per row in message column
    def sentiment(data):
        if data["po"] >= data["ne"] and data["po"] >= data["nu"]:
            return 1
        if data["ne"] >= data["po"] and data["ne"] >= data["nu"]:
            return -1
        if data["nu"] >= data["po"] and data["nu"] >= data["ne"]:
            return 0

    # Creating new column & Applying function
    df['value'] = df.apply(lambda row: sentiment(row), axis=1)

    # fetch unique analysis
    monthlyTimeline = dailyTimeline = activityMap = mostBusyUsers = 0
    usersContribution = wordCloud = commonWords = emojiAnalysis = 0
    user_analysis = ["Overall", "Monthly Timeline", "Daily Timeline",
                     "Activity Map", "Most Busy Users", "Word Cloud", "Common Words", "Emoji Analysis", "Users Contribution"]
    # user_list.remove('group_notification')
    user_analysis.insert(0, "Overall")

    selected_analysis = st.sidebar.selectbox("Show analysis on", user_analysis)


    # fetch unique users
    user_list = df['user'].unique().tolist()
    # user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        if selected_analysis == "Overall":
            monthlyTimeline = dailyTimeline = activityMap = mostBusyUsers = 1
            wordCloud = commonWords = emojiAnalysis = usersContribution =  1
        elif selected_analysis == "Monthly Timeline":
            monthlyTimeline = 1
        elif selected_analysis == "Daily Timeline":
            dailyTimeline = 1
        elif selected_analysis == "Activity Map":
            activityMap = 1
        elif selected_analysis == "Most Busy Users":
            mostBusyUsers = 1
        elif selected_analysis == "Word Cloud":
            wordCloud = 1
        elif selected_analysis == "Common Words":
            commonWords = 1
        elif selected_analysis == "Emoji Analysis":
            emojiAnalysis = 1
        elif selected_analysis == "Users Contribution":
            usersContribution = 1



    # Stats Area
    num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

    # st.title("Top Statistics")
    col1, col2, col3, col4 = st.columns(4)
    st.title("Top Statistics")

    with col1:
        st.header("Total Messages")
        st.title(num_messages)
    with col2:
        st.header("Total Words")
        st.title(words)
    with col3:
        st.header("Media Shared")
        st.title(num_media_messages)
    with col4:
        st.header("Links Shared")
        st.title(num_links)

    # monthly timeline
    if monthlyTimeline == 1:
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

    # daily timeline
    if dailyTimeline == 1:
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

    # activity map
    if activityMap == 1:
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

    # finding the busiest users in the group
    if mostBusyUsers == 1:
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

    # WordCloud
    if wordCloud == 1:
        st.title("WordCloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

    # most common words
    if commonWords == 1:
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()

        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most common words')
        st.pyplot(fig)

    # emoji analysis
    if emojiAnalysis == 1:
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            #fig,ax = plt.subplots()
           # ax.pie(emoji_df[1],labels=emoji_df[0],autopct="%0.2f")
           # st.pyplot(fig)
            #fig = px.pie(emoji_df, values='1', names='0', title='Emoji usage')
           # st.plotly_chart(fig, use_container_width=True)
            #fig = go.Figure(data=[go.Pie(labels=emoji_df['emoji'], values=emoji_df['count'], hole=.3)])
           # st.plotly_chart(fig)
            fig = px.pie(emoji_df, values=1, names=0, title='Emoji Occurrences')
            fig.update_traces(textinfo='label', text=emoji_df[0])
            st.plotly_chart(fig)

    # Percentage contributed
    if usersContribution == 1:
        if selected_user == 'Overall':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.title("Most Positive Contribution")
                x = helper.percentage(df, 1)

                # Displaying
                st.dataframe(x)
            with col2:
                st.title("Most Neutral Contribution")
                y = helper.percentage(df, 0)

                # Displaying
                st.dataframe(y)
            with col3:
                st.title("Most Negative Contribution")
                z = helper.percentage(df, -1)

                # Displaying
                st.dataframe(z)

        # Most Positive,Negative,Neutral User...
        if selected_user == 'Overall':
            # Getting names per sentiment
            x = df['user'][df['value'] == 1].value_counts().head(10)
            y = df['user'][df['value'] == -1].value_counts().head(10)
            z = df['user'][df['value'] == 0].value_counts().head(10)

            col1, col2, col3 = st.columns(3)
            with col1:
                # heading
                st.title("Most Positive Users")

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # heading
                st.title("Most Neutral Users")

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                # heading
                st.title("Most Negative Users")

                # Displaying
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
