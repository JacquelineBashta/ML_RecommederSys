import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import requests

# To prepare requirements.txt file
#print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))


###################################################################################
# Global Variables
###################################################################################
DEFAULT_MOOD = "Choose a Mood"
RANDOM_MOOD = "Surprise Me!"
NO_MOVIE_SELECTED  = "Select a Movie"
N_MOVIES = 10

key_count = 0
###################################################################################
# Static Functions
###################################################################################
def do_config():
    st.set_page_config(
    page_title="Movify Recommender",
    layout="wide")
    
#-----------------------------------------------------------------------------------------#  
  
def get_unique_key():
    global key_count
    key_count +=1
    return key_count

#-----------------------------------------------------------------------------------------#   
 
def weighted_rating(x, m, C):
    v = x['rate_count']
    R = x['rate_mean']
    return (v/(v+m) * R) + (m/(m+v) * C)

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
@st.cache(suppress_st_warning=True)
def scrap_movie_data (url):
    headers = {'Accept-Language': 'en-US,en;q=0.8','User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers = headers)

    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    image_url = (soup
                 .select("div.ipc-poster.ipc-poster--baseAlt.ipc-poster--dynamic-width.sc-30a29d44-0.dktfIa.celwidget.ipc-sub-grid-item")[0]
                 .select("a")[0]
                 .get("href"))
    image_url = "https://www.imdb.com"+image_url
    response2 = requests.get(image_url, headers = headers)
    response2.raise_for_status()
    soup2 = BeautifulSoup(response2.content, "html.parser")
    image_jpg = soup2.select("div.media-viewer")[0].select("img")[0].get("srcset").split(" ")[0]
    #return st.image(image_jpg,width=100)
    #return st.markdown("![image_title](image_jpg)")
    return image_jpg

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

def get_popularity_recommendation(names_df, ratings_df, n_top=10, mode=RANDOM_MOOD,criteria="weighted rate"):

    rating_info = (
        ratings_df
        .groupby("movieId")
        .agg(rate_mean=("rating", "mean"), rate_count=("rating", "count"))
        .reset_index()
    )
    if criteria == "weighted rate":
        rating_info["weighted_score"] = (rating_info.rate_count/rating_info.rate_count.sum()) * rating_info.rate_mean
        rating_info = rating_info.merge(names_df,on="movieId", how="left")
        
        if mode != RANDOM_MOOD:
            rating_info = rating_info.query("genres.str.contains(@mode)")
            
        rating_info = rating_info.nlargest(n_top, "weighted_score")
        
    elif criteria == "baysian average":
        c = ratings_df.rating.mean()
        m = (
            ratings_df
            .groupby("movieId")
            .agg(rate_count=("rating","count"))
            .reset_index()
        ).rate_count.quantile(0.9)
        
        rating_info=(
            ratings_df
            .groupby("movieId")
            .agg(rate_mean =("rating","mean"),rate_count=("rating","count"))
            .reset_index()
        )

        rating_info = rating_info.loc[rating_info.rate_count >= m]
        rating_info["rate_bayes"] = rating_info.apply(weighted_rating, axis=1,args=(m,c))
        rating_info = rating_info.merge(names_df,on="movieId",how="left").nlargest(n_top,"rate_bayes")
        if mode != RANDOM_MOOD:
            rating_info = rating_info.query("genres.str.contains(@mode)")
        
    return rating_info[["title", "genres", "movieId"]].reset_index(drop=True)

#-----------------------------------------------------------------------------------------#

def get_item_based_recommendation(n, chosen_movie_title, criteria):
    chosen_movie_id = int(movies_df.query("title == @chosen_movie_title").movieId)

    cross_table = pd.pivot_table(data=ratings_df, values="rating", columns="movieId", index="userId")
    
    corr_table = pd.DataFrame({})
    corr_table["cross_corr"] = cross_table.corrwith(cross_table[chosen_movie_id]).dropna().drop(chosen_movie_id)

    corr_table_view = corr_table.merge(movies_df, on="movieId")[["title", "genres", "cross_corr", "movieId"]]

    if criteria == "n_largest_corr":
        final_table = corr_table_view.nlargest(n, "cross_corr")

    elif criteria == "50_plus_rate_count":
        rating = ratings_df.groupby("movieId").agg(rate_count=("rating", "count")).reset_index()
        final_table = (corr_table_view
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

        rating["weighted_rate"] = (rating.rate_count/rating.rate_count.sum()) *100* rating.rate_mean
        weighted_rate_threshold = (rating.weighted_rate.max()-rating.weighted_rate.min())/2
        final_table = (
            corr_table_view
            .merge(rating, on="movieId")
            .query("weighted_rate >= @weighted_rate_threshold")
            .nlargest(n, "cross_corr")
        )
        
    return final_table[["title", "genres","movieId"]].reset_index().drop(columns=["index"])

#-----------------------------------------------------------------------------------------#

def get_user_based_recommendation( inp_ratings_df,n=10, user_id= 1):
    chosen_user = user_id
    n_top = n
    cross_table = pd.pivot_table(data=inp_ratings_df, values="rating",columns="movieId",index="userId")
    cross_table = cross_table.fillna(0)
    
    usrs_similarities = pd.DataFrame(cosine_similarity(cross_table),columns=cross_table.index, index=cross_table.index)
    
    user_weight_col = usrs_similarities.query("userId!=@chosen_user")[chosen_user]
    weights = user_weight_col/sum(user_weight_col)
    
    not_rated_items = cross_table.loc[cross_table.index!=chosen_user , cross_table.loc[chosen_user,:]== 0 ]
    predicted_rates = pd.DataFrame(not_rated_items.T.dot(weights),columns=["predicted_rates"])
    
    recommended_items = predicted_rates.merge(movies_df, on="movieId").nlargest(n_top,"predicted_rates")[["title","genres","movieId"]]
    
    return recommended_items

#-----------------------------------------------------------------------------------------#
#@st.cache(suppress_st_warning=True)
def visualize_result(result_df, msg,fail_msg):
    if len(result_df) != 0:
        result_df = result_df.rename(columns={"title":"Title","genres":"Genres"})
        st.write(msg)
        result_df.Title=result_df.apply(lambda x: f'<a target="_blank" href="{construct_imdb_url(x.movieId)}">{x.Title}</a>',axis=1)
        result_df["Poster"] = result_df.apply(lambda x: f'<img src="{scrap_movie_data(construct_imdb_url(x.movieId))}", width="100">',axis=1)
        result_df = result_df.drop(columns="movieId")
        
        st.write(result_df[0:N_MOVIES].to_html(escape=False,justify="left"), unsafe_allow_html=True)
        if len(result_df[N_MOVIES:N_MOVIES*2]) != 0:
            if st.button(label='More',key=get_unique_key()):
                st.write(result_df[N_MOVIES:N_MOVIES*2].to_html(escape=False,justify="left"), unsafe_allow_html=True)
    else:
        st.info(fail_msg)


###################################################################################
# Start of the Website Page
###################################################################################

do_config()
links_df, movies_df, ratings_df, tags_df = prepare_data()

st.title(":red[Movify :popcorn: The Movie Recommender :popcorn:] ")
st.markdown('&nbsp;')


## Popularity recommendation
###############################
with st.container():
    st.header("What is your mood today ?")
    genres_list = get_genres_list(movies_df)
    option = st.selectbox(label="", options=genres_list,key=get_unique_key())
    
    if option != DEFAULT_MOOD:
        top_movies_list = get_popularity_recommendation(n_top=N_MOVIES*2
                                                , names_df=movies_df
                                                , ratings_df=ratings_df
                                                , mode=option
                                                , criteria="baysian average")
        
        msg= f":clapper: Here is some popular movies for your mood :clapper:"
        fail_msg = "Sorry my Database is teeny-tiny .. There is no popular Movies with that mood :pleading_face: "
        visualize_result(top_movies_list, msg, fail_msg)


st.markdown('&nbsp;')
st.markdown('&nbsp;')


    
## Item based recommendation
###############################
with st.container():
    all_movies = list(movies_df.title)
    all_movies.insert(0, NO_MOVIE_SELECTED)
    st.header("Tell me a Movie that you like")
    title = st.selectbox("", all_movies)
    if title != NO_MOVIE_SELECTED:
        similar_movie_list = get_item_based_recommendation(N_MOVIES*2, title, "weighted_rate")
        msg= f" :heart: Because you loved {title}, you might enjoy these movies :heart:"
        fail_msg = f"Sorry my Database is teeny-tiny .. There is no Movies Similar enough to {title} :pleading_face: "
        visualize_result(similar_movie_list, msg, fail_msg)


st.markdown('&nbsp;')
st.markdown('&nbsp;')


## User based recommendation
###############################
#N_MOVIES_TO_RATE = ratings_df.groupby("userId").agg(movies_count=("movieId","count")).movies_count.min()
N_MOVIES_TO_RATE = 5
RATING_MIN = float(ratings_df.rating.unique().min())
RATING_MAX = float(ratings_df.rating.unique().max())
RATING_MID = ((RATING_MAX-RATING_MIN)/2) + RATING_MIN

movies_l=[]
rating_l=[]

with st.container():
    all_movies = list(movies_df.title)
    all_movies.insert(0, NO_MOVIE_SELECTED)
    st.header(f"Can you rate 5 Movies of your selection ")
    
    for n in range(N_MOVIES_TO_RATE):
        col1, col2 = st.columns([3,1])
        movie = col1.selectbox("", all_movies,key=get_unique_key())
        rating = col2.slider('',RATING_MIN,RATING_MAX,RATING_MID,step=0.25,key=get_unique_key())
        movies_l.append(movie)
        rating_l.append(rating)
    
    
    if (NO_MOVIE_SELECTED in movies_l) :
        st.warning(f"Please rate all {N_MOVIES_TO_RATE} movies")
    elif (len(movies_l) != len(set(movies_l))):
        st.warning(f"Please rate {N_MOVIES_TO_RATE} different movies")
    else:
        # add new user data to the database
        ratings_ex_df = ratings_df.copy()
        user_id_ex = int(ratings_df.userId.max()+1)
        for i,movie in enumerate(movies_l):
            rating_ex = rating_l[i]
            movie_id_ex = int(movies_df.query("title == @movie").movieId.values[0])
            new_row = {'userId':user_id_ex, 'movieId':movie_id_ex, 'rating':rating_ex, 'timestamp':0}
            ratings_ex_df = pd.concat([ratings_ex_df,pd.DataFrame([new_row])],ignore_index=True)
        user_based_list = get_user_based_recommendation( ratings_ex_df,N_MOVIES*2, user_id_ex)
        
        msg= "These Movies might fit your taste"
        fail_msg = f"Sorry my Database is teeny-tiny .. There is no Movies to your taste :pleading_face: "
        visualize_result(user_based_list, msg, fail_msg)

#st.balloons()
