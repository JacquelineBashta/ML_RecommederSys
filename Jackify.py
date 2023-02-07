
import streamlit as st
import pandas as pd

# To prepare requirements.txt file
#print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
DEFAULT_MOOD = "Choose a Mood"
RANDOM_MOOD = "Surprise Me!"
###################################################################################
# Static Functions
################################################################################


def prepare_data():
    links_df = pd.read_csv("datasets/ml-latest-small/links.csv")
    movies_df = pd.read_csv("datasets/ml-latest-small/movies.csv")
    ratings_df = pd.read_csv("datasets/ml-latest-small/ratings.csv")
    tags_df = pd.read_csv("datasets/ml-latest-small/tags.csv")
    return (links_df, movies_df, ratings_df, tags_df)


def get_genres_list(movies):
    all_genres_l = movies.genres.str.split("|").sum()
    unique_genres_l = []
    for x in all_genres_l:
        # check if exists in unique_list or not
        if x not in unique_genres_l:
            unique_genres_l.append(x)
    unique_genres_l.remove('(no genres listed)')
    unique_genres_l.insert(0, RANDOM_MOOD)
    unique_genres_l.insert(0, DEFAULT_MOOD)

    return (unique_genres_l)


def n_top_movies_weighted(names_df, ratings_df, n_top=10, mode=RANDOM_MOOD):

    rating_info = (
        ratings_df
        .groupby("movieId")
        .agg(rate_mean=("rating", "mean"), rate_count=("rating", "count"))
        .reset_index()
    )
    rating_info["weighted_score"] = (
        rating_info.rate_count/rating_info.rate_count.sum()) * rating_info.rate_mean

    rating_info = rating_info.merge(names_df, how="left")
    if mode != RANDOM_MOOD:
        rating_info = rating_info.query("genres.str.contains(@mode)")

    rating_info = rating_info.nlargest(n_top, "weighted_score")

    # return rating_info[["title", "genres", "weighted_score", "rate_mean", "rate_count"]].reset_index().drop(columns="index")
    return rating_info[["title", "genres"]].reset_index().drop(columns="index")


def recommend_similar_movies(n, chosen_movie_title, criteria):
    chosen_movie_id = int(movies_df.query(
        "title == @chosen_movie_title").movieId)

    cross_table = pd.pivot_table(
        data=ratings_df, values="rating", columns="movieId", index="userId")

    corr_table = pd.DataFrame({})
    corr_table["cross_corr"] = cross_table.corrwith(
        cross_table[chosen_movie_id]).dropna().drop(chosen_movie_id)
    # print(corr_table)

    corr_table_view = corr_table.merge(movies_df, on="movieId")[
        ["title", "genres", "cross_corr", "movieId"]]

    if criteria == "n_largest_corr":
        final_table = corr_table_view.drop(
            columns="movieId").nlargest(n, "cross_corr")

    elif criteria == "50_plus_rate_count":
        rating = ratings_df.groupby("movieId").agg(
            rate_count=("rating", "count")).reset_index()
        final_table = (
            corr_table_view
            .merge(rating, on="movieId")
            .drop(columns="movieId")
            .query("rate_count >= 50")
            .nlargest(n, "cross_corr")
        )

    elif criteria == "weighted_rate":
        rating = (
            ratings_df
            .groupby("movieId")
            .agg(rate_mean=("rating", "mean"), rate_count=("rating", "count"))
            .reset_index())

        rating["weighted_rate"] = (
            rating.rate_count/rating.rate_count.sum()) * 100 * rating.rate_mean
        print(rating.weighted_rate.min(), rating.weighted_rate.max())
        final_table = (
            corr_table_view
            .merge(rating, on="movieId")
            .drop(columns="movieId")
            .query("weighted_rate >= 0.2")
            .nlargest(n, "cross_corr")
        )

    elif criteria == "bayes_average":
        # TODO: implement it
        pass

    return final_table
###################################################################################
# Start of the Website Page
###########################################################################


links_df, movies_df, ratings_df, tags_df = prepare_data()


st.title("Jackify Movie Recommender ")


# with st.expander(" :cinema: **About Jackyfy** :cinema:"):
#     st.write("Test ")


genres_list = get_genres_list(movies_df)
with st.container():
    option = st.selectbox('What is your mood today ?', genres_list)
    if option != DEFAULT_MOOD:
        top_movies_list = n_top_movies_weighted(
            n_top=50, names_df=movies_df, ratings_df=ratings_df, mode=option)
        st.write(
            f":clapper: Here is some highly rated movies for your mood :clapper:")
        st.write(top_movies_list[0:10])
        if st.button('More', key=1):
            st.write(top_movies_list[10:20])


with st.container():
    all_movies = list(movies_df.title)
    all_movies.insert(0, "Select a Movie")
    title = st.selectbox('Tell me a Movie that you like', all_movies)
    if title != "Select a Movie":
        st.write(
            f" :heart: Because you loved {title}, you might enjoy these ones :heart:")

        similar_movie_list = recommend_similar_movies(
            50, title, "50_plus_rate_count")
        st.write(similar_movie_list[0:10])
        if st.button('More', key=2):
            st.write(similar_movie_list[10:20])

st.balloons()
