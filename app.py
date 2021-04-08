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
model=pickle.load(open('algo.pkl','rb'))
data = pd.read_csv('final_df.csv')
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
    recommend_id = st.text_input("Please enter the user id you want to recommend for", default_value_goes_here)
    number_of_books = st.text_input("How many books do you want to predict", default_value_goes_here)
    result = ""

    # Display Books
    if st.button("Predict"):
        all_book_id = data.book_id.unique()
        top_n = []
        for book_id in all_book_id:
            top_n.append(model.predict(uid=int(recommend_id), iid=book_id))
            top_n.sort(key=lambda x: x.est, reverse=True)
        list = [id_to_name[pred.iid] for pred in top_n[:int(number_of_books)]]

        st.write(list)


if __name__ == '__main__':
    main()
