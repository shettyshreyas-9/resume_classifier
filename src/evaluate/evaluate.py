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
    tfidf_vectorizer = joblib.load(model_path+'/vectorizer/tfidf_vectorizer.joblib')

    X_test_tfidf = tfidf_vectorizer.transform(X_test)


    # Evaluate models based on configurations from params.yaml
    for model_config in params['models']:
        model_type= model_config['model_type']
        hyperparameters_list = model_config['hyperparameters']

        # for key,values in hyperparameters_list.items():
        #     print(key)
        #     print(values)

        for hyperparameters in product(*hyperparameters_list.values()):
            # Load the trained model
            model_folder = '_'.join(str(val) for val in hyperparameters)
            model_file_path = os.path.join(model_path, f"{model_type}/{model_folder}/model.joblib")

            print(model_file_path)











if __name__ == '__main__':
    main()