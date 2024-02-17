import pathlib
import os
import sys
import logging
import yaml

import re
import string
string.punctuation

import nltk

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def setup_logging():
    log_file_path = pathlib.Path(__file__).parent.as_posix()+ sys.argv[3]
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_data(data_path):
    df = pd.read_csv(data_path,encoding='utf-8')
    return df


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
    eng_stop_words= stopwords.words('english')
    punctuations= string.punctuation

    for i in j:
        if i not in eng_stop_words and i not in punctuations:
            k.append(i)


    # Stemming  ** Problem in downloading wordnet so removing lemmetization **
    # from nltk.stem import WordNetLemmatizer
    # lemmatizer = WordNetLemmatizer()

    # l=[]
    # for i in k:
    #     l.append(lemmatizer.lemmatize(i))

    # converting tokens into str 
    preprocessed_df = " ".join(k)
    return preprocessed_df


def split_data(output_df,output_path,test_split,seed):
    train, test= train_test_split(output_df, test_size=test_split, random_state= seed)
    return train, test
    


def save_data(train_df, test_df,output_path):
    train_df.to_csv(output_path+'/train_df.csv',index=False)
    test_df.to_csv(output_path+'/test_df.csv',index=False)

    logging.info("Saved train and test data successfully")




def main():

    # calling the logging function
    setup_logging()

    # project setup via cookiecutter may give problem using (__file__) so used (os.getcwd())
    curr_dir = pathlib.Path(__file__)
    working_dir= pathlib.Path(os.getcwd())
    home_dir = curr_dir.parent.parent.parent

    # input data
    input_file= sys.argv[1]
    data_path= working_dir.as_posix() + input_file

    # print ('\n',data_path,'\n')

    data_df= load_data(data_path)

    logging.info("Loaded data successfully")

    # clean data
    cleaned_df= data_df['Resume'].apply(lambda x: clean_data(x))
    logging.info("Cleaned data successfully")

    # pre-process data

        # NLTK dowload - # custom data path for downloading nltk
    nltk_path = working_dir.as_posix()+ sys.argv[4] 
    pathlib.Path(nltk_path).mkdir(parents=True, exist_ok=True)
        
    nltk.download('punkt', download_dir=nltk_path)
    nltk.download('stopwords', download_dir=nltk_path)
    nltk.download('wordnet', download_dir=nltk_path)
    
    preprocessed_df= cleaned_df.apply(lambda x: preprocess_data(x)) 
    logging.info("Preprocessed data successfully")

    # Label Encode data
    output_df = pd.concat([data_df['Category'], preprocessed_df.reset_index()['Resume']], axis=1)
    le= LabelEncoder()
    output_df['Category_encoded']= le.fit_transform(output_df['Category'])
    logging.info("Label encoded categorical data")


    # Split data
    params_file = working_dir.as_posix()+ sys.argv[5]
    params= yaml.safe_load(open(params_file))['make_dataset']

    output_path= working_dir.as_posix() + sys.argv[2]
    

    train_df, test_df= split_data(output_df,output_path,params['test_split'],params['seed'])
    logging.info("Split train and test data")


    # Save data

    save_data(train_df, test_df,output_path)


    print(nltk.data.path)



if __name__ == '__main__':
    main()
