import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyAtVD0b4ULneDRdeqo9X8WvIa-izsMquxA"
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df['overview'] = df['overview'].fillna('')
    return df

movies = load_data()

@st.cache_data
def create_similarity_matrix(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = create_similarity_matrix(movies)

def get_gemini_recommendations(query):
    prompt = (
        f"You are a movie expert. Recommend 5 movies similar in theme, vibe, or genre "
        f"to the movie '{query}'. Focus on popular and highly-rated titles that the user "
        f"is likely to enjoy. Only return the movie names, each on a new line, without "
        f"extra symbols, numbers, or explanations."
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    movies_list = [m.strip(" -*") for m in response.text.split("\n") if m.strip()]
    return movies_list

def get_recommendations(title, cosine_sim=cosine_sim):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]['title'].tolist()

st.markdown("""
    <style>
    /* Gradient title */
    .gradient-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #ff512f, #dd2476);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding-top: 10px;
    }

    /* Movie boxes */
    .movie-box {
        background: rgba(255,255,255,0.05);
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
    }
    .movie-box:hover {
        background: rgba(255,255,255,0.15);
        transform: translateX(5px) scale(1.02);
    }

    /* Gradient button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ff512f, #dd2476);
        color: white;
        padding: 0.6em 1.2em;
        border-radius: 30px;
        border: none;
        font-size: 1.1em;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0px 4px 12px rgba(221, 36, 118, 0.4);
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.05);
        box-shadow: 0px 6px 16px rgba(221, 36, 118, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("Logo.png", width=150)
    st.title("🍿 The Binge Button")
    st.markdown("""
    **🎬 Your AI Movie Wingman**  
    Discover your next favorite movie in seconds — no endless scrolling, no guessing.  
    Just type a movie you love, hit **The Binge Button**, and let AI do the magic.
    
    **✨ Why you'll love it:**  
    - 🔍 **Smart Picks** – Finds similar gems from our huge database  
    - 🤖 **AI Backup** – If we don’t have it, Gemini AI steps in  
    - ⚡ **Instant Results** – No waiting, just binging  
    - 💡 **Personal Touch** – Recommendations that *actually* match your vibe
    
    ---
    Ready to binge smarter?  
    **Hit the button and start watching!**
    """)

col1, col2 = st.columns([1, 8])
with col1:
    st.image("Logo.png", width=60)
with col2:
    st.markdown('<h1 class="gradient-title">The Binge Button</h1>', unsafe_allow_html=True)

movie_input = st.text_input("Enter a movie you like:")

if st.button("🔍 Recommendations"):
    if movie_input:
        recs = get_recommendations(movie_input)
        if not recs:
            recs = get_gemini_recommendations(movie_input)
        if recs:
            for movie in recs:
                st.markdown(f'<div class="movie-box">{movie}</div>', unsafe_allow_html=True)
        else:
            st.warning("No recommendations found.")
    else:
        st.warning("Please enter a movie title.")

st.markdown("""
    <style>
    /* Footer styling with fade-in */
    .footer {
        position: center;
        left: 0;
        bottom: 0;
        width: 100%;
        color: white;
        text-align: center;
        padding: 8px;
        font-size: 0.9rem;
        font-family: 'Arial', sans-serif;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.2);
        z-index: 100;
        animation: fadeIn 1.2s ease-in-out;
    }
    </style>
    <div class="footer">
        Made with ❤️ by <b>Jhalak Verma</b> | 🚀 Powered by <b>Gemini Vision Pro</b>
    </div>
""", unsafe_allow_html=True)
