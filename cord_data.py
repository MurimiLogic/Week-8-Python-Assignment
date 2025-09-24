import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from wordcloud import WordCloud
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import streamlit as st

#Part 1: Data Loading and Basic Exploration
    #1. Load the dataset from a CSV file.
df = pd.read_csv("metadata.csv")
    #2. Basic Exploration
print("First 5 rows of the dataset:")
print(df.head())
print("\nFirst 10 rows:")
print(df.head(10))
        #Datatypes and dataframe dimensions
print("\nDataframe info")
print(df.info())
print("\nDataframe Dimensions:",df.shape)
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nData Types:")
print(df.dtypes)
        #Check for missing values in important columns
print("\nMissing values in each column:")
print(df.isnull().sum())
        #Missing values in important columns
important_columns = ['title', 'abstract', 'publish_time', 'journal']
print("\nMissing values in important columns:")
print(df[important_columns].isnull().sum())
        #Basic statistics of numerical columns
print("\nStatistical summary of numerical columns:")
print(df.describe())


#Part 2: Data Cleaning and Preparation
    #3. Handle missing data
            #Columns with many missing values
missing_counts = df.isnull().sum()
print("\nColumns with missing values:")
print(missing_counts[missing_counts > 0])
                #Columns with more than 50% missing values
print("\nColumns with more than 50% missing values:")
print(missing_counts[missing_counts > 0.5*len(df)])
        #Removal of columns with too many missing values
cols_to_drop = missing_counts[missing_counts > 0.5*len(df)].index
df_cleaned = df.drop(columns=cols_to_drop)
print(f"\nDropped columns: {list(cols_to_drop)}")

         #Cleaned version of the dataset
print("\nCleaned Dataframe info:")
print(df_cleaned.info())
print("\nCleaned Dataframe Dimensions:", df_cleaned.shape)
print(f"Number of rows: {df_cleaned.shape[0]}")
print(f"Number of columns: {df_cleaned.shape[1]}")  
     
    #4. Preparation for analysis
        #Convert publish_time to datetime   
df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')
        #Extract publication year   
df_cleaned['publication_year'] = df_cleaned['publish_time'].dt.year
        #Calculate abstract word count  
df_cleaned['abstract_word_count'] = df_cleaned['abstract'].fillna('').apply(lambda x: len(x.split()))
print("\nDataframe with publication year and abstract word count:")
print(df_cleaned[['publish_time', 'publication_year', 'abstract_word_count']].head())

#Part 3: Data Analysis and Visualization
    #5. Basic Analysis
      #Count papers by publication year
print("\nPublications by Year:")
print(df_cleaned['publication_year'].value_counts().sort_index())    
      #Top journals publishing COVID-19 research   
print("\nTop 10 Journals publishing COVID-19 research:")
print(df_cleaned['journal'].value_counts().head(10))
      #Most frequent words used  in titles 
all_titles = " ".join(df_cleaned['title'].dropna().astype(str).values).lower()
words = all_titles.split()
filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 2]
word_counts = Counter(filtered_words)
print("\nMost frequent words in titles:")
print(word_counts.most_common(10))  
    #6. Visualizations
        #Number of publications by year
yearly_counts = df_cleaned['publication_year'].value_counts().sort_index()
plt.figure(figsize=(10, 5)) 
sns.barplot(x=yearly_counts.index.astype(str), y=yearly_counts.values, palette="viridis")
plt.title("Publications by Year")
plt.xlabel("Year")  
plt.ylabel("Number of Papers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

        #Bar chart of top 10 journals
top_journals = df_cleaned['journal'].value_counts().head(10)    
plt.figure(figsize=(8, 5))
sns.barplot(x=top_journals.values, y=top_journals.index, palette="rocket")
plt.title("Top 10 Journals")    
plt.xlabel("Number of Papers")
plt.ylabel("")  
plt.tight_layout()
plt.show()
        #Word cloud of paper titles
title_wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(all_titles)
plt.figure(figsize=(10, 5))     
plt.imshow(title_wordcloud, interpolation='bilinear')
plt.axis('off') 
plt.title("Word Cloud of Titles")
plt.tight_layout()
plt.show()
        # distribution of paper counts by source
source_counts = df_cleaned['source_x'].fillna('Unknown').value_counts().head(1000)
plt.figure(figsize=(8, 5))      
sns.barplot(x=source_counts.values, y=source_counts.index, palette="mako")
plt.title("Top Sources")
plt.xlabel("Number of Papers")
plt.ylabel("")  
plt.tight_layout()
plt.show()

#Part 4: Streamlit Application
     #7. Simple streamlit App

st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("CORD-19 Data Explorer")
st.write("Simple exploration of COVID-19 research paperst")

# Loading data and caching
@st.cache_data
def load_data():
    df = pd.read_csv("metadata.csv")
    
    # Data cleaning
    missing_counts = df.isnull().sum()
    cols_to_drop = missing_counts[missing_counts > 0.5*len(df)].index
    df_cleaned = df.drop(columns=cols_to_drop)
    
    # Data preparation
    df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')
    df_cleaned['publication_year'] = df_cleaned['publish_time'].dt.year
    df_cleaned['abstract_word_count'] = df_cleaned['abstract'].fillna('').apply(lambda x: len(x.split()))
    
    return df_cleaned

# Load the data
df_cleaned = load_data()

# Sidebar for filters and controls
st.sidebar.header("Filters and Controls")

# Year range slider
min_year = int(df_cleaned['publication_year'].min())
max_year = int(df_cleaned['publication_year'].max())
year_range = st.sidebar.slider(
    "Select publication year range",
    min_value=min_year,
    max_value=max_year,
    value=(2020, max_year)
)

# Filter data based on year range
filtered_df = df_cleaned[
    (df_cleaned['publication_year'] >= year_range[0]) & 
    (df_cleaned['publication_year'] <= year_range[1])
]

# Journal selection
top_journals = filtered_df['journal'].value_counts().head(20).index.tolist()
selected_journals = st.sidebar.multiselect(
    "Filter by journal (top 20)",
    options=top_journals,
    default=[]
)

if selected_journals:
    filtered_df = filtered_df[filtered_df['journal'].isin(selected_journals)]

# Main content area
st.header("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Papers", len(filtered_df))

with col2:
    st.metric("Time Period", f"{year_range[0]} - {year_range[1]}")

with col3:
    st.metric("Journals Represented", filtered_df['journal'].nunique())

with col4:
    avg_words = filtered_df['abstract_word_count'].mean()
    st.metric("Avg Abstract Words", f"{avg_words:.0f}")

# Visualizations
st.header("Visualizations")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Publications Timeline", 
    "Top Journals", 
    "Title Word Cloud", 
    "Top Sources",
    "Word Frequency"
])

with tab1:
    st.subheader("Publications by Year")
    yearly_counts = filtered_df['publication_year'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=yearly_counts.index.astype(str), y=yearly_counts.values, palette="viridis", ax=ax)
    ax.set_title("Publications by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Papers")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Top Journals")
    top_journals_viz = filtered_df['journal'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top_journals_viz.values, y=top_journals_viz.index, palette="rocket", ax=ax)
    ax.set_title("Top 10 Journals")
    ax.set_xlabel("Number of Papers")
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Word Cloud of Paper Titles")
    
    # Generate word cloud
    all_titles = " ".join(filtered_df['title'].dropna().astype(str).values).lower()
    
    if all_titles.strip():
        title_wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            max_words=200
        ).generate(all_titles)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(title_wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Word Cloud of Titles")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No titles available for the selected filters")

with tab4:
    st.subheader("Top Sources")
    source_counts = filtered_df['source_x'].fillna('Unknown').value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=source_counts.values, y=source_counts.index, palette="mako", ax=ax)
    ax.set_title("Top Sources")
    ax.set_xlabel("Number of Papers")
    plt.tight_layout()
    st.pyplot(fig)

with tab5:
    st.subheader("Most Frequent Words in Titles")
    
    # Calculate word frequencies
    all_titles = " ".join(filtered_df['title'].dropna().astype(str).values).lower()
    words = all_titles.split()
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 2]
    word_counts = Counter(filtered_words)
    
    # Display top words
    top_words = word_counts.most_common(20)
    words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=words_df, x='Frequency', y='Word', palette="coolwarm", ax=ax)
    ax.set_title("Top 20 Words in Titles")
    ax.set_xlabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show data table
    st.dataframe(words_df, use_container_width=True)

# Additional data exploration
st.header("Data Exploration")

# Show sample data
if st.checkbox("Show sample data"):
    st.subheader("Sample Papers")
    st.dataframe(filtered_df[['title', 'journal', 'publication_year', 'abstract_word_count']].head(10))

# Journal statistics
if st.checkbox("Show journal statistics"):
    st.subheader("Journal Statistics")
    journal_stats = filtered_df['journal'].value_counts().reset_index()
    journal_stats.columns = ['Journal', 'Paper Count']
    st.dataframe(journal_stats.head(20), use_container_width=True)

# Yearly statistics
if st.checkbox("Show yearly statistics"):
    st.subheader("Yearly Publication Statistics")
    yearly_stats = filtered_df['publication_year'].value_counts().sort_index().reset_index()
    yearly_stats.columns = ['Year', 'Paper Count']
    st.dataframe(yearly_stats, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("CORD-19 Data Explorer | COVID-19 Research Papers Analysis")