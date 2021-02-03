from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np



plays = pd.read_csv('data/user_artists.dat', sep='\t')
artists = pd.read_csv('data/artists.dat', sep='\t', usecols=['id','name'])

# Merge artist and user pref data
ap = pd.merge(artists, plays, how="inner", left_on="id", right_on="artistID")
#print("ap",ap)#.name[0:10])
ap = ap.rename(columns={"weight": "playCount"})
#print("ap_rename",ap)

# Group artist by name
artist_rank = ap.groupby(['name']) \
    .agg({'userID' : 'count', 'playCount' : 'sum'}) \
    .rename(columns={"userID" : 'totalUsers', "playCount" : "totalPlays"}) \
    .sort_values(['totalPlays'], ascending=False)

artist_rank['avgPlays'] = artist_rank['totalPlays'] / artist_rank['totalUsers']

# Merge into ap matrix
ap = ap.join(artist_rank, on="name", how="inner") \
    .sort_values(['playCount'], ascending=False)

# Preprocessing
pc = ap.playCount
play_count_scaled = (pc - pc.min()) / (pc.max() - pc.min())
ap = ap.assign(playCountScaled=play_count_scaled)
#print("ap",ap)

# Build a user-artist rating matrix 
ratings_df = ap.pivot(index='userID', columns='artistID', values='playCountScaled')




##################### fonction pour afficher la liste de tous les artists ###############################

def artist_data():
    artists = pd.read_csv('data/artists.dat', sep='\t', usecols=['id','name'])

    artist = list(artists['name'])#[0:100]
    artist = sorted(artist, reverse=False)
    return artist


###################### fonction recommendation avec LightFM pour les artists les plus ecoutés #################

def get_recommandation(artist_choice):
    list_artist_choice = []
    add_user = [0]*17632
    num = []

    for item in artist_choice:
        #each = item[0]
        #print('each',each)
        artist_index = list(artists.index[artists['name']==item])#.values)
        num.append(artist_index)
        #print(num)
        for i in num:
            for j in i:
                index = j#[0]
                add_user[index]=1
            new_ratings_df = np.vstack((ratings_df, add_user))
    #convert array to DataFrame
            ratings_DF= pd.DataFrame(new_ratings_df) 
    #ratings_DF = add_user(artist_choice)
    new_userID = (ratings_DF.shape[0] - 1)
    ratings = ratings_DF.fillna(0).values
    # Build a sparse matrix
    X = csr_matrix(ratings)
    n_users, n_items = ratings_DF.shape
    user_ids = ratings_DF.index.values
    artist_names = ap.sort_values("artistID")["name"].unique()
    Xcoo = X.tocoo()
    data = Dataset()
    data.fit(np.arange(n_users), np.arange(n_items))
    interactions, weights = data.build_interactions(zip(Xcoo.row, Xcoo.col, Xcoo.data)) 
    train, test = random_train_test_split(interactions)
    # model with best parameters
    model = LightFM(k = 10, n = 10, learning_rate = 0.5, learning_schedule = 'adadelta', loss='warp')
    model.fit(train, epochs=10, num_threads=2)
    
    #prediction
    n_users, n_items = X.shape
    #print(X.shape)
    
    scores = model.predict(new_userID, np.arange(n_items))
    top_items = artist_names[np.argsort(-scores)]
    return top_items[0:10]


################################# fonction recommendation avec LightFM + cosine_sim #####################################


def recommandation_for_you(artist_choice):
    list_artist_choice = []
    add_user = [0]*17632
    num = []

    for item in artist_choice:
        #each = item[0]
        #print('each',each)
        artist_index = list(artists.index[artists['name']==item])#.values)
        num.append(artist_index)
        #print(num)
        for i in num:
            for j in i:
                index = j#[0]
                add_user[index]=1
            new_ratings_df = np.vstack((ratings_df, add_user))
    #convert array to DataFrame
    ratings_DF= pd.DataFrame(new_ratings_df) 
    ratings = ratings_DF.fillna(0).values
    new_userID = (ratings_DF.shape[0] - 1)
    # Build a sparse matrix
    X = csr_matrix(ratings)
    n_users, n_items = ratings_DF.shape
    user_ids = ratings_DF.index.values
    artist_names = ap.sort_values("artistID")["name"].unique()
    Xcoo = X.tocoo()
    data = Dataset()
    data.fit(np.arange(n_users), np.arange(n_items))
    interactions, weights = data.build_interactions(zip(Xcoo.row, Xcoo.col, Xcoo.data)) 
    train, test = random_train_test_split(interactions)
    #calculate cosine_sim 
    cosine_sim = cosine_similarity(X[-1,:], X)
    # model with best parameters
    model = LightFM(k = 10, n = 10, learning_rate = 0.5, learning_schedule = 'adadelta', loss='warp')
    model.fit(train, epochs=10, num_threads=2)
    
    #prediction
    n_users, n_items =  cosine_sim.shape
    #print(X.shape)
    
    scores = model.predict(new_userID, np.arange(n_items))
    top_items = artist_names[np.argsort(-scores)]
    return top_items[0:10]

###################### fonction recommendation avec calcul de la moyenne de chaque ligne ######################

def for_you(artist_choice):
    I = []
    res = []
    list_artist_choice = []
    num_index = []
    add_user = [0]*17632
    num_id = []
    liste_recom = []
    for item in artist_choice:
        index_artist = list(artists.index[artists['name']==item])
        id_artist = list(artists.id[artists['name']==item])
        #print(id_artist)
        num_id.append(id_artist)
        num_index.append(index_artist)
        #print(num_id)
        for i in num_index:
            #print(i)
            for j in i:
                add_user[j]=1
                add_user_array = np.array(add_user).reshape(1,17632)
                add_user_True_False = add_user_array > 0
                ratings_True_False = ratings_df > 0
                add_user_DataFrame = pd.DataFrame(add_user_True_False ) 
                indice = list(range(0,ratings_df.shape[0]))
                indice2  = list(range(0,ratings_df.shape[1]))
                for i in indice:
                    ligne = ratings_True_False.iloc[i]
                    if (np.array(ligne).mean() == np.array(add_user_DataFrame).mean()):#.all():
                        res.append(ligne)
                        I.append(i)
                        l = list(plays.artistID[plays.userID==I[0]])
                        for each in l:
                            p = list(artists.name[artists.id==each])
                            liste_recom.append(p)
    return liste_recom[0:10]



####################### fonction recommendation avec le calcul de similarity cosine_sim #######################


def for_you_sim(artist_choice):
    I = []
    dict_artist = {}
    list_artist_choice = []
    num_index = []
    add_user = [0]*17632
    num_id = []
    liste_recom = []
    for item in artist_choice:
        index_artist = list(artists.index[artists['name']==item])
        id_artist = list(artists.id[artists['name']==item])
        #print(id_artist)
        num_id.append(id_artist)
        num_index.append(index_artist)
        #print(num_id)
        for i in num_index:
            #print(i)
            for j in i:
                add_user[j]=1
                add_user_array = np.array(add_user).reshape(1,17632)
                add_user_True_False = add_user_array > 0
                ratings_True_False = ratings_df > 0
                add_user_DataFrame = pd.DataFrame(add_user_True_False ) 
                indice = list(range(0,ratings_df.shape[0]))
                indice2  = list(range(0,ratings_df.shape[1]))
                for i in indice:
                    ligne = ratings_True_False.iloc[i]
                    cosine_sim = cosine_similarity(np.array(ligne).reshape(1,17632), add_user_DataFrame)
                    if cosine_sim >= 0.10:
                        dict_artist[i]=cosine_sim
                        valeur = [(value, key) for key, value in dict_artist.items()]
                        max_val = max(valeur)[1]
                        I.append(max_val)
                        l = list(plays.artistID[plays.userID==I[0]])
                        for each in l:
                            p = list(artists.name[artists.id==each])#.values
                            liste_recom.append(p)
    return liste_recom[0:10]