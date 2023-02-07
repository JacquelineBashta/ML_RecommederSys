
import streamlit as st
import pandas as pd
import seaborn as sns

# To prepare requirements.txt file
#print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))

rel_ds_path = "datasets\ml-latest-small"


def prepare_data(path):
    links_df = pd.read_csv(path + "\links.csv")
    movies_df = pd.read_csv(path + "\movies.csv")
    ratings_df = pd.read_csv(path + "\\ratings.csv")
    tags_df = pd.read_csv(path + "\\tags.csv")
    return (links_df, movies_df, ratings_df, tags_df)


def get_genres_list(movies):
    all_genres_l = movies.genres.str.split("|").sum()
    unique_genres_l = []
    for x in all_genres_l:
        # check if exists in unique_list or not
        if x not in unique_genres_l:
            unique_genres_l.append(x)
    unique_genres_l.remove('(no genres listed)')
    unique_genres_l.insert(0, "Surprise Me!")
    unique_genres_l.insert(0, "Choose Mode")

    return (tuple(unique_genres_l))


def n_top_movies_weighted(names_df, ratings_df, n_top=10, mode="Surprise Me!"):

    rating_info = (
        ratings_df
        .groupby("movieId")
        .agg(rate_mean=("rating", "mean"), rate_count=("rating", "count"))
        .reset_index()
    )
    rating_info["weighted_score"] = (
        rating_info.rate_count/rating_info.rate_count.sum()) * rating_info.rate_mean

    rating_info = rating_info.merge(names_df, how="left")
    if mode != "Surprise Me!":
        rating_info = rating_info.query("genres.str.contains(@mode)")

    rating_info = rating_info.nlargest(n_top, "weighted_score")

    # return rating_info[["title", "genres", "weighted_score", "rate_mean", "rate_count"]].reset_index().drop(columns="index")
    return rating_info[["title", "genres"]].reset_index().drop(columns="index")


links_df, movies_df, ratings_df, tags_df = prepare_data(rel_ds_path)
genres_list = get_genres_list(movies_df)

st.title("Jackify Movie Recommender ")

# with st.expander(" :cinema: **About Jackyfy** :cinema:"):
#     st.write("Test ")

with st.container():
    st.write(" :clapper: Here Comes Top rated Movies :clapper:")
    option = st.selectbox('What is your mode today ?', genres_list)
    if option != "Choose Mode":
        top_movies_list = n_top_movies_weighted(
            n_top=50, names_df=movies_df, ratings_df=ratings_df, mode=option)
        st.write(f"Here is some high rated {option} movies for you :")
        st.write(top_movies_list[0:10])
        if st.button('More', key=1):
            st.write(top_movies_list[10:20])

st.balloons()
