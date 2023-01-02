import pandas as pd 
import numpy as np
from lightfm.data import Dataset
import pickle
from HelperFunctons import *
from flask import Flask,request,app,render_template

app=Flask(__name__)

#charger les données
'''df1 = pd.read_csv('ratings.csv')
df2 = pd.read_csv('movies.csv')
rating_df = df2.merge(df1, on = 'movieId' ).dropna()
df = rating_df[['userId','movieId','rating']]
df4 = rating_df[['userId','movieId','rating','title','genres']]
train = pickle.load(open("train",'rb'))'''

#charger les 2 model
model_2 = pickle.load(open("model.pkl",'rb'))
# model_1= pickle.load(open("model1.pkl",'rb'))


# Preparation des données et paramétres pour le model_2
uf = [] 
col = ['movieId']*len(df4['movieId'].unique()) + ['rating']*len(df4['rating'].unique()) + ['title']*len(df4['title'].unique()) + ['genres']*len(df4['genres'].unique()) 
unique_f1 = list(df4['movieId'].unique()) + list(df4['rating'].unique()) + list(df4['title'].unique()) + list(df4['genres'].unique()) 
#print('f1:', unique_f1)
for x,y in zip(col, unique_f1):
    res = str( x)+ ":" +str(y)
    uf.append(res)
    #print(res)
                   
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

user_id_map, user_feature_map, movie_id_map, movie_feature_map = data.mapping()


@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    user=int(request.form['user_id'])
    recommendation=recommend(model_2,user)
    return render_template("index.html",prediction_text=recommendation[0],prediction_text2=recommendation[1])



if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=5000)