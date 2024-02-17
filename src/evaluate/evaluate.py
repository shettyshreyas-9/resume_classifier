import pathlib
import os
import sys
import joblib
import logging
import yaml
from itertools import product

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score


import mlflow


def setup_logging():
    # configure logging and print to console
    log_file_path = pathlib.Path(__file__).parent.as_posix() + sys.argv[4]
    logging.basicConfig(
        filename= log_file_path,
        level= logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# Set the MLflow tracking URI
mlflow.set_tracking_uri('http://localhost:5000')


# Check if a run is active
if mlflow.active_run():
    mlflow.end_run()

# MLflow experiment name
experiment_name= "aikido"

# Check if the experiment exists, if not, create it
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(experiment_name)

mlflow.start_run()


def main():

    setup_logging()

    curr_dir= pathlib.Path(__file__)
    working_dir = pathlib.Path(os.getcwd())

    #parameters
    params_file= working_dir.as_posix()+ sys.argv[3]
    params= yaml.safe_load(open(params_file))

    # data path
    data_path= working_dir.as_posix()+ sys.argv[1]

    # model path
    model_path = working_dir.as_posix() + sys.argv[2]

    # load test data
    test_df= pd.read_csv(data_path+'/test_df.csv')
    X_test = test_df.drop(columns=['Category','Category_encoded'])
    y_test= test_df['Resume']


    # Transform the test data

    # Load the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    # X_test_tfidf = tfidf_vectorizer.transform(X_test)










if __name__ == '__main__':
    main()