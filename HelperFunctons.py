from scipy import sparse
import numpy as np
import pandas as pd 
import pickle 
from lightfm.data import Dataset


df1 = pd.read_csv('ratings.csv')
df2 = pd.read_csv('movies.csv')
rating_df = df2.merge(df1, on = 'movieId' ).dropna()
df = rating_df[['userId','movieId','rating']]
df4 = rating_df[['userId','movieId','rating','title','genres']]
train = pickle.load(open("train",'rb'))

# Preparation des données et paramétres pour le model_2
uf = [] 
col = ['movieId']*len(df4['movieId'].unique()) + ['rating']*len(df4['rating'].unique()) + ['title']*len(df4['title'].unique()) + ['genres']*len(df4['genres'].unique()) 
unique_f1 = list(df4['movieId'].unique()) + list(df4['rating'].unique()) + list(df4['title'].unique()) + list(df4['genres'].unique()) 
#print('f1:', unique_f1)
for x,y in zip(col, unique_f1):
    res = str( x)+ ":" +str(y)
    uf.append(res)
    #print(res)


# Helper function that takes the user features and converts them into the proper "feature:value" format
def feature_colon_value(my_list):
    """
    Takes as input a list and prepends the columns names to respective values in the list.
    For example: if my_list = [1,1,0,'del'],
    resultant output = ['movieId','rating','title','genres']
   
    """
    result = []
    ll = ['movieId:','rating:','title:','genres:']
    aa = my_list
    for x,y in zip(ll,aa):
        res = str(x) +""+ str(y)
        result.append(res)
    return result



# Using the helper function to generate user features in proper format for ALL users
ad_subset = df4[['movieId','rating','title','genres']] 
ad_list = [list(x) for x in ad_subset.values]
feature_list = []
for item in ad_list:
    feature_list.append(feature_colon_value(item))
#print(f'Final output: {feature_list}')

data = Dataset()
#data.fit(df.userId.unique(), df.movieId.unique())
data.fit( 
        df4['userId'].unique(), 
        df4.movieId.unique(), # tous les éléments
        user_features = uf # fonctionnalités utilisateur supplémentaires
 )









from scipy import sparse
def format_newuser_input(user_feature_map, user_feature_list):
    num_features = len(user_feature_list)
    normalised_val = 1.0 
    target_indices = []
    for feature in user_feature_list:
        try:
            target_indices.append(user_feature_map[feature])
        except KeyError:
            print("new user feature encountered '{}'".format(feature))
            pass

    new_user_features = np.zeros(len(user_feature_map.keys()))
    for i in target_indices:
        new_user_features[i] = normalised_val
    new_user_features = sparse.csr_matrix(new_user_features)
    return(new_user_features)

    


def recommend(model, user_id):
    n_users, n_items = train.shape
    user_id_map, user_feature_map, movie_id_map, movie_feature_map = data.mapping()

    user_feature_list = ['movieId:1', 'rating:1', 'title:Toy Story (1995)', 'genres:Adventure|Animation|Children|Comedy|Fantasy']
    new_user_features = format_newuser_input(user_feature_map, user_feature_list)


    best_rated = rating_df[(rating_df.userId == user_id) & (rating_df.rating >= 4.5)].movieId.values
    if best_rated.shape[0] == 0 :
        scores = model.predict(0, np.arange(n_items), user_features=new_user_features) 
        top_items = rating_df['title'][np.argsort(-scores)]
        print("\nRecommended:")
        for x in top_items[:10]:
            print(x)
        return [[],top_items[:10].values.tolist()]

    else : 
        known_positives = rating_df.loc[rating_df['movieId'].isin(best_rated)].title.values
        scores = model.predict(user_id, np.arange(n_items)) 
        top_items = rating_df['title'][np.argsort(-scores)]
        ls = []
        for x in known_positives:
            if x not in ls :
                ls.append(x)
                
        print("User %s likes:" % user_id)
        for k in ls:
            print(k)
            
        print("\nRecommended:")
        tp = []
        for x in top_items[:10]:
            print(x)
        
        return [ls,top_items[:10].values.tolist()]
    
