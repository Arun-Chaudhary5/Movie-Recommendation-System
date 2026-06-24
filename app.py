import streamlit as st
import pandas as pd
import pickle
from scipy import sparse

# Load files
data = pickle.load(open("movies_data.pkl", "rb"))
nn_model = pickle.load(open("nn_model.pkl", "rb"))
tfidf_matrix = sparse.load_npz("tfidf_matrix.npz")

# Create title index
title_to_index = pd.Series(
    data.index,
    index=data['title']
).drop_duplicates()

# Recommendation Function
def recommend(movie_name, n=10):

    if movie_name not in title_to_index:
        return None

    idx = title_to_index[movie_name]

    distances, indices = nn_model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=n+1
    )

    recommendations = []

    for i, d in zip(
        indices.flatten()[1:],
        distances.flatten()[1:]
    ):

        similarity = 1 - d

        rating = data.iloc[i]['vote_average']

        final_score = (
            0.7 * similarity +
            0.3 * (rating / 10)
        )

        recommendations.append({
            "Movie": data.iloc[i]['title'],
            "Similarity": round(similarity, 4),
            "Rating": rating,
            "Score": round(final_score, 4)
        })

    recommendations = pd.DataFrame(recommendations)

    recommendations = recommendations.sort_values(
        "Score",
        ascending=False
    )

    return recommendations


# UI
st.title("🎬 Movie Recommendation System")

movie = st.text_input("Enter Movie Name")

if st.button("Recommend"):

    results = recommend(movie)

    if results is None:
        st.error("Movie not found!")
    else:
        st.dataframe(
    results.reset_index(drop=True),
    use_container_width=True
)