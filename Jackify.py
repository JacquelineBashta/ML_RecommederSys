import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity

# To prepare requirements.txt file
# print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))


###################################################################################
# Global Variables
###################################################################################
DEFAULT_MOOD = "Choose a Mood"
RANDOM_MOOD = "Surprise Me!"
N_MOVIES = 10


###################################################################################
# Static Functions
###################################################################################
def do_config():
    st.set_page_config(layout="wide")
#-----------------------------------------------------------------------------------------#    
    
def prepare_data():
    links_df = pd.read_csv("datasets/ml-latest-small/links.csv")
    movies_df = pd.read_csv("datasets/ml-latest-small/movies.csv")
    ratings_df = pd.read_csv("datasets/ml-latest-small/ratings.csv")
    tags_df = pd.read_csv("datasets/ml-latest-small/tags.csv")
    return (links_df, movies_df, ratings_df, tags_df)
#-----------------------------------------------------------------------------------------#


def construct_imdb_url(movie_id):
    imdb_tag = int(links_df.query("movieId == @movie_id").imdbId)
    imdb_tag = str(imdb_tag).rjust(7, "0")
    imdb_tag = "tt"+imdb_tag

    url = "https://www.imdb.com/title/"+imdb_tag
    return (url)
#-----------------------------------------------------------------------------------------#


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
#-----------------------------------------------------------------------------------------#


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
    return rating_info[["title", "genres", "movieId"]].reset_index().drop(columns="index")

#-----------------------------------------------------------------------------------------#


def get_item_based_recommendation(n, chosen_movie_title, criteria):
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
        final_table = corr_table_view.nlargest(n, "cross_corr")

    elif criteria == "50_plus_rate_count":
        rating = ratings_df.groupby("movieId").agg(
            rate_count=("rating", "count")).reset_index()
        final_table = (
            corr_table_view
            .merge(rating, on="movieId")
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
            .query("weighted_rate >= 0.2")
            .nlargest(n, "cross_corr")
        )

    elif criteria == "bayes_average":
        rating = (
            ratings_df
            .groupby("movieId")
            .agg(rate_mean=("rating", "mean"), rate_count=("rating", "count"))
            .reset_index())
        m = rating.rate_count.quantile(0.95)
        c = rating.rate_mean.mean()
        qualified_movies = rating[(rating.rate_count > m)
                                  & (rating.rate_mean > c)]
        qualified_movies = qualified_movies.assign(weighted_rate=lambda x: (
            x.rate_count / (x.rate_count + m)*x.rate_mean) + (m/(x.rate_count + m)*c))
        top_movies = qualified_movies.sort_values(
            "weighted_rate", ascending=False)
        

    return final_table[["title", "genres","movieId"]].reset_index().drop(columns=["index"])

#-----------------------------------------------------------------------------------------#

def get_user_based_recommendation( n=10, user_id= 1):
    chosen_user = user_id
    n_top = n
    cross_table = pd.pivot_table(data=ratings_df, values="rating",columns="movieId",index="userId")
    cross_table = cross_table.fillna(0)
    
    usrs_similarities = pd.DataFrame(cosine_similarity(cross_table),columns=cross_table.index, index=cross_table.index)
    
    user_weight_col = usrs_similarities.query("userId!=@chosen_user")[chosen_user]
    weights = user_weight_col/sum(user_weight_col)
    
    not_rated_items = cross_table.loc[cross_table.index!=chosen_user , cross_table.loc[chosen_user,:]== 0 ]
    predicted_rates = pd.DataFrame(not_rated_items.T.dot(weights),columns=["predicted_rates"])
    
    recommended_items = predicted_rates.merge(movies_df, on="movieId")[["title","genres","predicted_rates"]].nlargest(n_top,"predicted_rates")
    
    return recommended_items




###################################################################################
# Start of the Website Page
###################################################################################

do_config()
links_df, movies_df, ratings_df, tags_df = prepare_data()

st.title(":violet[Jackify Movie Recommender] ")
st.markdown('&nbsp;')

# with st.expander(" :cinema: **About Jackyfy** :cinema:"):
#     st.write("Test ")


## Popularity recommendation
with st.container():
    st.header("What is your mood today ?")
    genres_list = get_genres_list(movies_df)
    option = st.selectbox(label="", options=genres_list,label_visibility='collapsed')
    if option != DEFAULT_MOOD:
        st.write(f":clapper: Here is some highly rated movies for your mood :clapper:")
        top_movies_list = n_top_movies_weighted(n_top=N_MOVIES*2
                                                , names_df=movies_df
                                                , ratings_df=ratings_df
                                                , mode=option)

        top_movies_list.title=top_movies_list.apply(lambda x: f'<a target="_blank" href="{construct_imdb_url(x.movieId)}">{x.title}</a>',axis=1)
        top_movies_list = top_movies_list.drop(columns="movieId")
        st.write(top_movies_list[0:N_MOVIES].to_html(escape=False), unsafe_allow_html=True)
        placeholder = st.empty()
        isclick = placeholder.button(label='More')
        if isclick:
            #placeholder.empty()
            st.write(top_movies_list[N_MOVIES:N_MOVIES*2].to_html(escape=False), unsafe_allow_html=True)



st.markdown('&nbsp;')
st.markdown('&nbsp;')


## Item based recommendation
with st.container():
    all_movies = list(movies_df.title)
    all_movies.insert(0, "Select a Movie")
    st.header("Tell me a Movie that you like")
    title = st.selectbox("", all_movies)
    if title != "Select a Movie":
        st.write(f" :heart: Because you loved {title}, you might enjoy these movies :heart:")
        similar_movie_list = get_item_based_recommendation(N_MOVIES*2, title, "50_plus_rate_count")
        
        similar_movie_list.title=similar_movie_list.apply(lambda x: f'<a target="_blank" href="{construct_imdb_url(x.movieId)}">{x.title}</a>',axis=1)
        similar_movie_list = similar_movie_list.drop(columns="movieId")
        
        
        st.write(similar_movie_list[0:N_MOVIES].to_html(escape=False), unsafe_allow_html=True)
        placeholder2 = st.empty()
        isclick2 = placeholder2.button(label='More',key=2)
        if isclick2:
            st.write(similar_movie_list[N_MOVIES:N_MOVIES*2].to_html(escape=False), unsafe_allow_html=True)


st.markdown('&nbsp;')
st.markdown('&nbsp;')

def shoe_updated_movie_rate(selected_movies_l,rates_l):
    selected_movies_df = pd.DataFrame({
            "Movie":[selected_movies_l],
            "Given Rate":[rates_l]
        })
## User based recommendation
with st.container():
    all_movies = list(movies_df.title)
    all_movies.insert(0, "Select a Movie")
    st.header("Rate at least 5 Movies ( rates from 1 to 10)")
    
    movie_1 = st.selectbox("", all_movies,key=2)
    selected_movies_l=[]
    rates_l=[]
    if title != "Select a Movie":
        
        st.write(f" :heart: Because you loved {title}, you might enjoy these movies :heart:")
        similar_movie_list = get_item_based_recommendation(N_MOVIES*2, title, "50_plus_rate_count")
        
        similar_movie_list.title=similar_movie_list.apply(lambda x: f'<a target="_blank" href="{construct_imdb_url(x.movieId)}">{x.title}</a>',axis=1)
        similar_movie_list = similar_movie_list.drop(columns="movieId")
        
        
        st.write(similar_movie_list[0:N_MOVIES].to_html(escape=False), unsafe_allow_html=True)
        placeholder2 = st.empty()
        isclick2 = placeholder2.button(label='More',key=2)
        if isclick2:
            st.write(similar_movie_list[N_MOVIES:N_MOVIES*2].to_html(escape=False), unsafe_allow_html=True)

st.snow()
