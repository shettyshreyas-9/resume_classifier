import pathlib
import sys
import os
import joblib

import pandas as pd
import numpy as np
import re
import string

import nltk

import streamlit as st



def get_latest_best_model_path(evaluate_log_file):
    with open(evaluate_log_file,'r') as log_file:
        lines= log_file.readlines()
        for line in reversed(lines):
            if 'Best model path' in line:
                return line.split('=')[-1].strip()
            else:
                return 'unavailable'
            
def clean_data(data_df):
    # resume_text= data_df['Resume']

    # remove urls
    cleaned_df= re.sub('http\S+','',data_df)
    # remove RT & cc
    cleaned_df= re.sub('http\S+','',cleaned_df)
    # remove hashtags
    cleaned_df= re.sub('#\S+','',cleaned_df)
    # remove mentions
    cleaned_df= re.sub('@\S+','',cleaned_df)
    # remove non-ASCII character
    cleaned_df= re.sub(r'[^\x00-\x7f]',r' ', cleaned_df)
    # remove extra whitespace
    cleaned_df = re.sub('\s+', ' ', cleaned_df)

    return cleaned_df


def preprocess_data(cleaned_df):

    # Lower case
    preprocessed_df= cleaned_df.lower()

    # Word tokenization
    preprocessed_df= nltk.word_tokenize(preprocessed_df)

    # Removing special characters
    j =[]
    for i in preprocessed_df:
        if i.isalnum():
            j.append(i)

    # Removing stopwords
    k= []
    from nltk.corpus import stopwords
    # stop_words= working_dir
    eng_stop_words= stopwords.words('english')
    punctuations= string.punctuation

    # print(eng_stop_words,'\n')
    # print(punctuations)

    for i in j:
        if i not in eng_stop_words and i not in punctuations:
            k.append(i)


    # Stemming  ** Problem in downloading wordnet so removing lemmetization **
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    l=[]
    for i in k:
        l.append(lemmatizer.lemmatize(i))

    # converting tokens into str 
    preprocessed_df = " ".join(l)
    return preprocessed_df


def main():

    curr_dir = pathlib.Path(__file__)
    working_dir = pathlib.Path(os.getcwd())
    model_path = working_dir.as_posix() + sys.argv[3]

    # Add the directory where you downloaded nltk data to paths for fetching
    nltk.data.path.append( working_dir.as_posix()+ sys.argv[2])
    print(nltk.data.path)


    evaluate_log_file= working_dir.as_posix()+sys.argv[1]
    print(evaluate_log_file)

    best_model_path= get_latest_best_model_path(evaluate_log_file)
    print(best_model_path)

    # Check if the model file exists
    if os.path.exists(best_model_path):

        #Load the best model
        best_model= joblib.load(best_model_path)

    st.title('Resume Classifier')
        
    # getting category & encoded values from train data
    train_df = pd.read_csv(working_dir.as_posix()+ sys.argv[4])

    unique_categories= train_df[['Category','Category_encoded']].drop_duplicates().sort_values('Category_encoded')

    category_dict= dict(zip(unique_categories['Category_encoded'],unique_categories['Category']))
        
    
    # Get user input text
    sample_text = st.text_input("Enter Resume text:")
    
    cleaned_data= clean_data(sample_text)

    preprocessed_data = preprocess_data(cleaned_data)

    # print(cleaned_data)
    # print(preprocessed_data)

    # Load the TF-IDF vectorizer
    tfidf_vectorizer = joblib.load(model_path+'/vectorizer/tfidf_vectorizer.joblib')

    vectorized_data = tfidf_vectorizer.transform([preprocessed_data])
    print(vectorized_data.shape)

    print(category_dict)

    prediction =best_model.predict(vectorized_data)
    # print(prediction)

    st.write("Resume belongs to category: ", category_dict[prediction[0]])

if __name__=='__main__':
    main()
