import streamlit as st
import pandas as pd
import nltk
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import re

nltk.data.path.append('./nltk_data')

# Load Data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Tokenize Data
def tokenize_column(column):
    return column.apply(
        lambda x: [token for token in word_tokenize(str(x).lower()) if re.match(r'^[a-zA-Z0-9]+$', token)]
    )

# Count Tokens
def count_tokens(tokenized_df):
    all_tokens = [token for tokens in tokenized_df for token in tokens]
    return Counter(all_tokens)

# Word Cloud
def generate_wordcloud(token_counts):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(token_counts)
    return wordcloud

# Streamlit App
def main():
    st.title("Tokenization Dashboard")
    st.sidebar.title("Configuration")

    # File Upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("### Dataset Preview", df.head())

        # Dynamic Column Selection (Multiple Columns)
        column_options = df.columns.tolist()
        selected_columns = st.sidebar.multiselect("Select Columns for Tokenization", column_options)

        if selected_columns:
            # Tokenization for Selected Columns
            st.write("### Tokenization")
            stop_words = set(stopwords.words('english'))
            df['tokens'] = df[selected_columns].apply(
                lambda row: [
                    token
                    for col in selected_columns
                    for token in tokenize_column(pd.Series([row[col]]))[0]
                    if token not in stop_words
                ],
                axis=1
            )

            st.write(df[['tokens']].head())

            # Token Counts
            token_counts = count_tokens(df['tokens'])

            # Token Frequency Distribution
            st.write("### Token Frequency Distribution")
            top_n = st.sidebar.slider("Select Top N Tokens", min_value=10, max_value=100, value=20)
            most_common_tokens = token_counts.most_common(top_n)
            tokens, counts = zip(*most_common_tokens)
            token_freq_df = pd.DataFrame({'Token': tokens, 'Frequency': counts})
            fig = px.bar(token_freq_df, x='Token', y='Frequency', title='Top Tokens', text='Frequency')
            fig.update_traces(textposition='outside')
            fig.update_layout(xaxis_title='Tokens', yaxis_title='Frequency')
            st.plotly_chart(fig)

            # Word Cloud
            st.write("### Word Cloud")
            wordcloud = generate_wordcloud(token_counts)
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

            # Token Length Distribution
            st.write("### Token Length Distribution")
            token_lengths = [len(token) for token in token_counts.keys()]
            token_length_df = pd.DataFrame({'Length': token_lengths})
            fig = px.histogram(token_length_df, x='Length', nbins=20, title='Token Length Distribution')
            fig.update_layout(xaxis_title='Token Length', yaxis_title='Frequency')
            st.plotly_chart(fig)

            # Tokenized Length Distribution by Row
            st.write("### Tokenized Length Distribution by Row")
            row_lengths = df['tokens'].apply(len)
            row_length_df = pd.DataFrame({'Row': range(len(row_lengths)), 'Token Count': row_lengths})
            fig = px.histogram(row_length_df, x='Token Count', nbins=20, title='Tokenized Length by Row')
            fig.update_layout(xaxis_title='Token Count', yaxis_title='Frequency')
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
