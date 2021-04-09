import random
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
import pickle
from surprise import accuracy
from surprise import BaselineOnly
from surprise.model_selection import train_test_split

# loading the trained model
model=pickle.load(open('final_algo.pkl','rb'))
data = pd.read_csv('full_rating_dataframe.csv')
dff = pd.read_csv('Mapping_df.csv')

id_to_name = {}
list1 = list(dff['article_id'])
list2 = list(dff['title'])
for i in range(len(list1)-1):
    id_to_name[list1[i]] = list2[i]

def main():
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:#002E6D;padding:20px;font-weight:15px"> 
    <h1 style ="color:white;text-align:center;"> IBM book Recommendation System</h1> 
    </div> 
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)
    default_value_goes_here = ""
    # recommend_id = st.text_input("Please enter the user id you want to recommend for", default_value_goes_here)
    # number_of_books = st.text_input("How many books do you want to predict", default_value_goes_here)
    recommend_id = st.number_input("Please enter the user id you want to recommend for", 0, 100000000, 0)
    number_of_books= st.number_input("How many books do you want to predict", 0, 100000000, 0)



    result = ""

    # Display Books
    if st.button("Predict"):
        # global result
        all_book_id = data.book_id.unique()
        top_n = []
        # if recommend_id in list(data['user_id'].unique()):
        if min(list(data['user_id'])) <= recommend_id <= max(list(data['user_id'])):
            for book_id in all_book_id:
                top_n.append(model.predict(uid=recommend_id, iid=book_id))
                top_n.sort(key=lambda x: x.est, reverse=True)
            if(number_of_books<len(top_n)):
                result = [id_to_name[pred.iid] for pred in top_n[:number_of_books]]
            else:

                result = "Choose a number that is less than "+str(len(top_n))
        else:
            result = "Choose an ID that is between "+str(min(list(data['user_id'])))+"  and  "+str(max(list(data['user_id'])))

        st.write(result)


if __name__ == '__main__':
    main()
