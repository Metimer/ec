import streamlit as st
import pandas as pd
from streamlit_authenticator import Authenticate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.corpus import stopwords

# Configuration de la page
st.set_page_config(
    page_title="Cinéma EDEN",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS pour améliorer le design
st.markdown(
    """
    <style>
    /* Fond général de l'application */
    .stApp {
        background-color: #1E1E1E !important;
        color: white !important;
    }

    /* Style de la sidebar */
    [data-testid="stSidebar"] {
        background-color: #2C2F33 !important;
        color: white !important;
    }

    /* Style des titres */
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #FFD700;
    }

    /* Style des cartes */
    .film-card {
        background-color: #2C2F33;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        text-align: center;
    }

    /* Style des affiches */
    .film-card img {
        border-radius: 10px;
        margin-bottom: 10px;
    }

    /* Style du texte des cartes */
    .film-card h3 {
        color: #FFD700;
    }

    .film-card p {
        font-size: 14px;
        color: #CCCCCC;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Logo dans la sidebar
logo_url = "https://i.postimg.cc/NMB7FRsV/0948bbca-e8f6-48e4-b3b5-00ec7e35e6e5.png"
st.sidebar.image(logo_url, use_container_width=True)

# Authentification
lesDonneesDesComptes = {
    'usernames': {
        'utilisateur': {'name': 'utilisateur', 'password': 'WILD','email': 'utilisateur@gmail.com',
            'failed_login_attempts': 0,
            'logged_in': False,
            'role': 'utilisateur'},
        'root': {'name': 'root', 'password': 'rootMDP','email': 'admin@gmail.com',
            'failed_login_attempts': 0,
            'logged_in': False,
            'role': 'administrateur'}
    }
}
authenticator = Authenticate(lesDonneesDesComptes, "cookie_name", "cookie_key", 30)
authenticator.login()

nltk.download('stopwords')
url_info='https://drive.google.com/uc?export=download&id=11hBENeUzl-XOKcPD9ANUHYDYq6zY8u_H'
url_ml='https://drive.google.com/uc?export=download&id=1IaJZ6JSs_MGWxS5pyE8Qsswzt-1YqJsE'

if not st.session_state.get("authentication_status"):
    st.write("Veuillez vous connecter.")
else:
    menu_options = ["Accueil", "Catalogue Films", "Assistant de recommandation"]
    selection = st.sidebar.selectbox("Choisissez une section", menu_options)

    # 🌟 **Accueil**
    if selection == "Accueil":
        st.markdown('<div class="title">Bienvenue au Cinéma Eden 🎬</div>', unsafe_allow_html=True)
        eden_url = "https://www.ville-lasouterraine.fr/app/uploads/2022/04/eden.jpg"
        eden_url2='https://api.cloudly.space/resize/clip/1900/1080/75/aHR0cHM6Ly9jZHQ2NC5tZWRpYS50b3VyaW5zb2Z0LmV1L3VwbG9hZC8xNzYwMDAxNjgtNC5qcGc=/image.jpg'
        st.markdown(
                f"""
                <div class="film-card">
                    <img src="{eden_url2}" width="1000">
                    <h3><p><b>Le Cinéma Eden est situé à La Souterraine, dans le département de la Creuse. Ce cinéma historique est un lieu incontournable pour les passionnés de films et de culture. Il propose une programmation variée allant des films récents aux classiques, et est un lieu de rencontre pour tous les amoureux du 7ème art. 
                    Venez découvrir une expérience cinématographique dans un cadre chaleureux et convivial !</b> </p> </h3>
                    
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown(
                f"""                
                <div class="film-card">                    
                    <img src="{eden_url}" width="1000">
                    <h3><p><b>Il propose une programmation variée allant des films récents aux classiques, et est un lieu de rencontre pour tous les amoureux du 7ème art. 
                    Venez découvrir une expérience cinématographique dans un cadre chaleureux et convivial !</b> </p> </h3>                    
                    
                </div>
                """,
                unsafe_allow_html=True
            )
    # 🎞 **Catalogue Films**
    elif selection == "Catalogue Films":
        st.markdown('<div class="title">🎞 Catalogue des films</div>', unsafe_allow_html=True)

        df = pd.read_csv(url_info)

        films_names = df['titre_fr'].unique()
        film_choice = st.sidebar.selectbox("Choisissez un film", films_names)

        selected_film = df[df['titre_fr'] == film_choice]

        if not selected_film.empty:
            poster_path = selected_film['poster'].values[0]
            poster_url = 'https://image.tmdb.org/t/p/original' + poster_path

            # Carte film avec affiche
            st.markdown(
                f"""
                <div class="film-card">
                    <img src="{poster_url}" width="600">
                    <h3>{film_choice}</h3>
                    <p><b>Date de sortie :</b> {selected_film["date_de_sortie"].values[0]}</p>
                    <p><b>Genres :</b> {selected_film["genres"].values[0]}</p>
                    <p><b>Popularité :</b> {round(selected_film["popularite"].values[0], 2)}</p>
                    <p><b>Synopsis :</b> {selected_film["synopsis"].values[0]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 🤖 **Assistant de recommandation**
    elif selection == "Assistant de recommandation":
        st.markdown('<div class="title">🎯 Assistant de recommandation</div>', unsafe_allow_html=True)

        dfml = pd.read_csv(url_ml).fillna('')

        dfml['features'] = (
            (dfml['synopsis'] + ' ')*2 + (dfml['genres'] + ' ')*2 + (dfml['acteurs'] + ' ') +
            (dfml['actrices'] + ' ') + (dfml['realisateurs'] + ' ')*2 + (dfml['producteurs'] + ' ')*1 + (dfml['scenaristes'] + ' ')
        )
        
        french_stopwords = stopwords.words('french')

        tfidf = TfidfVectorizer(stop_words=french_stopwords, max_features=50000)
        tfidf_matrix = tfidf.fit_transform(dfml['features'])

        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        def recommend(title, df, cosine_sim):
            if title not in dfml['titre_fr'].values:
                return []
            idx = df[df['titre_fr'] == title].index[0]
            similar_movies = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:6]
            return [(df['titre_fr'].iloc[i[0]], df['poster'].iloc[i[0]]) for i in similar_movies]

        selected_movie = st.sidebar.selectbox("Choisissez un film", dfml['titre_fr'].unique())
        selected_film2 = dfml[dfml['titre_fr'] == selected_movie]
        if selected_movie:
            recommendations = recommend(selected_movie, dfml, cosine_sim)

        if recommendations:
           st.write("### 🎬 Films recommandés :")
    
           for movie_title, poster in recommendations:
              movie_info = dfml[dfml['titre_fr'] == movie_title]  # Récupère les infos du film

              if not movie_info.empty:
                  st.markdown(
                  f"""
                  <div class="film-card">
                    <img src="https://image.tmdb.org/t/p/original{poster}" width="350">
                    <h3>{movie_title}</h3>
                    <p><b>Genres :</b> {movie_info["genres"].values[0]}</p>
                    <p><b>Popularité :</b> {round(movie_info["popularite"].values[0], 2)}</p>
                    <p><b>Synopsis :</b> {movie_info["synopsis"].values[0]}</p>
                    <p><b>Réalisateur :</b> {movie_info["realisateurs"].values[0]}</p>
                    <p><b>Acteurs principaux :</b> {movie_info["acteurs"].values[0]},{movie_info["actrices"].values[0]}</p>
                  </div>
                  """,
                  unsafe_allow_html=True
            )
        else:
            st.write("❌ Aucune recommandation trouvée.")


    authenticator.logout("Déconnexion", "sidebar")

