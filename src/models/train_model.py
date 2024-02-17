import pathlib
import os
import yaml
import joblib
import logging
import sys

from itertools import product

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier



def setup_logging():
    # configure logging to a file and also print to the console
    log_file_path = pathlib.Path(__file__).parent.as_posix()+ sys.argv[4]    # Specify the path to your log file
    logging.basicConfig(
        filename= log_file_path,
        level= logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def train_model(X_train_tfidf, y_train, model_type, hyperparameters, cv_search, cv_values, scoring, output_path):

    if model_type == 'logistic_regression':
        model= LogisticRegression()
    elif model_type == 'random_forest':
        model= RandomForestClassifier()
    elif model_type == 'svm':
        model= SVC()
    elif model_type == 'knc':
        model= KNeighborsClassifier()
    elif model_type == 'multinomial_nb':
        model= MultinomialNB()
    elif model_type == 'xgb_class':
        model= XGBClassifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    param_grid= hyperparameters

    # print('\n',cv_search,'\n',cv_values,'\n',scoring,'\n', type(cv_search))

    if cv_search == 'grid':
        search= GridSearchCV(model, param_grid, cv=cv_values, scoring=scoring)
    # if cv_search == 'random':
    #     search= RandomizedSearchCV(model,param_grid,n_iter=10,cv=cv_values, scoring=scoring)
    else:
        raise ValueError(f"Unsupported cv_type: {cv_search}")
    

    search.fit(X_train_tfidf,y_train)
    best_model= search.best_estimator_

    logging.info(f"Training {model_type} model with best hyperparameters based on {cv_search} {scoring}: {search.best_estimator_}")

    # model_folder = '_'.join(f"{key}_{val}" for key,val in search.best_estimator_.items())
    # os.makedirs(os.path.join(output_path,model_folder,'model.joblib'))
    # print(output_path,model_folder,'model.joblib')

    model_folder = '_'.join(f"{key}_{val}" for key, val in search.best_params_.items())
    os.makedirs(os.path.join(output_path, model_folder), exist_ok=True)
    joblib.dump(best_model, os.path.join(output_path, model_folder, 'model.joblib'))

def main():

    setup_logging()

    # project setup via cookiecutter may give problem using (__file__) so used (os.getcwd())
    curr_dir = pathlib.Path(__file__)
    working_dir= pathlib.Path(os.getcwd())
                # print(working_dir)

    # parameters
    params_file= working_dir.as_posix() + sys.argv[2]
    params= yaml.safe_load(open(params_file))

    # data path
    data_path = working_dir.as_posix()+ sys.argv[1]
    # print(data_path)

    # model path
    model_path = working_dir.as_posix()+sys.argv[3]
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_path+'/train_df.csv')
    X_train= train_df['Resume']
    y_train= train_df['Category_encoded']

    # Add TF-IDF transformation
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

        # Save the fitted vectorizer
    joblib.dump(tfidf_vectorizer, model_path+'/vectorizer/tfidf_vectorizer.joblib')

    logging.info(f"Saved the vectorizer model")

    # Hyperparameter tuning
    cv_search = params['hyperparameter_tuning']['cv_type']
    cv_search= cv_search.strip()
    cv_values= params['hyperparameter_tuning']['cv_values']
    scoring=  params['hyperparameter_tuning']['scoring']


    for model_config in params['models']:
        model_type = model_config['model_type']
        hyperparameters = model_config['hyperparameters']
        train_model(X_train_tfidf,y_train,model_type,hyperparameters,cv_search,cv_values,scoring, f"models/{model_type}/")

    

if __name__ == '__main__':
    main()