import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\eesha\Documents\Intern\Lumaa\imdb_top_1000.csv")
    return df

df_sample = load_data()

# Load pre-trained Sentence Transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Create a combined textual representation of movies
@st.cache_data
def preprocess_text(df):
    df['Combined_Info'] = df.apply(lambda row: 
        f"Overview: {row['Overview']} Genre: {row['Genre']} Director: {row['Director']} "
        f"Stars: {row['Star1']}, {row['Star2']} ", axis=1)
    return df

df_sample = preprocess_text(df_sample)

# Compute movie embeddings using the combined text
@st.cache_data
def compute_movie_embeddings(df):
    return model.encode(df['Combined_Info'].fillna(""), convert_to_tensor=True)

movie_embeddings = compute_movie_embeddings(df_sample)

# Function to recommend movies
def recommend_movies(user_input, min_rating, min_year):
    # Encode user input
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    
    # Compute similarity scores
    similarity_scores = util.pytorch_cos_sim(user_embedding.cpu(), movie_embeddings.cpu())[0]
    top_indices = similarity_scores.argsort(descending=True)[:15]

    # Get recommended movies
    recos = df_sample.iloc[top_indices][['Series_Title', 'IMDB_Rating', 'Released_Year', 'Genre', 'Director', 'Star1', 'Star2', 'Poster_Link']].copy()
    recos['Similarity_Score'] = similarity_scores[top_indices].cpu().numpy()

    # Convert to numeric and apply filters
    recos['IMDB_Rating'] = pd.to_numeric(recos['IMDB_Rating'], errors='coerce')
    recos['Released_Year'] = pd.to_numeric(recos['Released_Year'], errors='coerce')
    recos = recos.dropna(subset=['IMDB_Rating', 'Released_Year'])
    recos = recos[(recos['IMDB_Rating'] >= min_rating) & (recos['Released_Year'] >= min_year)]

    # Sort and get top recommendations
    if len(recos) > 5:
        if choose_what == "Bottom":
            recos = recos.sort_values(by=['IMDB_Rating', 'Similarity_Score'], ascending=[True, False]).head(5)
        else:
            recos = recos.sort_values(by=['IMDB_Rating', 'Similarity_Score'], ascending=[False, False]).head(5)
    else:
        if choose_what == "Top":
            recos = recos.sort_values(by=['IMDB_Rating', 'Similarity_Score'], ascending=[True, False])
        else:
            recos = recos.sort_values(by=['IMDB_Rating', 'Similarity_Score'], ascending=[False, False])

    return recos.reset_index(drop=True)

# Streamlit UI
st.title("Movie Recommendation System")

user_input = st.text_area("What kind of movies would you like to watch? Enter a short description here:", "Sci-fi adventure with space battles")
choose_what = st.selectbox("Do you want the top 5 movies or bottom 5?", ["Top", "Bottom"])
min_rating = st.slider("Minimum IMDB Rating:", 0.0, 10.0, 7.0)
min_year = st.slider("Minimum Release Year:", 1900, 2025, 2000)

if st.button("Get Recommendations"):
    recommendations = recommend_movies(user_input, min_rating, min_year)
    
    if recommendations.empty:
        st.warning("No recommendations found. Try different inputs!")
    else:
        st.write("### You can watch these movies:")
        for index, row in recommendations.iterrows():
            st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 10px; border: 2px solid #ccc; border-radius: 10px; margin: 10px;">
                <img src="{row['Poster_Link']}" style="width: 120px; height: 180px; border-radius: 10px; margin-right: 15px;">
                <div>
                    <h3>{row['Series_Title']} ({int(row['Released_Year'])})</h3>
                    <p>Genre: {row['Genre']}<br> Director: {row['Director']}<br> Stars: {row['Star1']}, {row['Star2']}<br> IMDB Rating: {row['IMDB_Rating']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
