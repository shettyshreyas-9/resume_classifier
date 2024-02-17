import pathlib
import os
import sys
import joblib
import logging
import yaml
from itertools import product
import subprocess

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score


import mlflow


def start_mlflow_ui():
    mlflow_ui_command = "mlflow ui"
    subprocess.Popen(mlflow_ui_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

start_mlflow_ui()


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
mlflow.set_tracking_uri('http://127.0.0.1:5000')


# Check if a run is active
if mlflow.active_run():
    mlflow.end_run()

# MLflow experiment name
my_experiment_name= "aikido"

# Check if the experiment exists, if not, create it
experiment = mlflow.get_experiment_by_name(my_experiment_name)
if experiment is None:
    mlflow.create_experiment(my_experiment_name)

mlflow.set_experiment(my_experiment_name)


# Function for evaluating model

def evaluate_model_and_log(X_test_tfidf,y_test,model,evaluate_type,metric_type,cv_values,model_type,params_and_values):

    if evaluate_type == 'cross_val_score':
        cv_scores= cross_val_score(model,X_test_tfidf, y_test, cv=cv_values,scoring=metric_type)

    # cv_scores=  cross_val_score(model,X_test_tfidf, y_test, cv=cv_values,scoring=metric_type)

    # Log the cross-validated scores
    for metric, score in zip([metric_type], [cv_scores.mean()]):
        mlflow.log_metric(metric, score)

    logging.info(f"The results for model: {model_type}, params: {params_and_values}, evaluation: {evaluate_type}, metric type:{metric_type}, CV: {cv_values} are = Mean scores: {cv_scores.mean()}")

    # Return the cross-validated scores
    return cv_scores
    
    # results = {}
    # cv_scores = {metric['metric_type']: 





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
    # X_test = test_df.drop(columns=['Category','Category_encoded'])
    X_test = test_df['Resume']
    y_test= test_df['Category_encoded']


    # Transform the test data

    # Load the TF-IDF vectorizer
    tfidf_vectorizer = joblib.load(model_path+'/vectorizer/tfidf_vectorizer.joblib')

    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print('\n')
    print(X_test.shape)
    print(type(X_test))
    
    print(X_test_tfidf.shape)
    print(type(X_test_tfidf))
    print(y_test.shape)
    print(type(y_test))
    print('\n')

    # Evaluate models based on configurations from params.yaml
    for model_config in params['models']:
        model_type= model_config['model_type']
        hyperparameters_list = model_config['hyperparameters']


        for hyperparameters in product(*hyperparameters_list.values()):
            # Load the trained model
            model_folder = '_'.join(f"{key}_{val}" for key, val in zip(hyperparameters_list.keys(), hyperparameters))
            model_file_path = os.path.join(model_path, f"{model_type}/{model_folder}/model.joblib")

            # print(model_folder)

            # Check if the model file exists (to only log the params for models that are saved)
            if os.path.exists(model_file_path):
                model = joblib.load(model_file_path)

                # Log the evaluation metrics using MLflow
                with mlflow.start_run():
                    mlflow.log_param("model_name", model_type)

                    # Log hyperparameters
                    params_and_values= dict(zip(hyperparameters_list.keys(), hyperparameters))
                    mlflow.log_params(params_and_values)

                    evaluate_type= params['evaluation']['evaluate_type']
                    metric_type= params['evaluation']['metric_type']
                    cv_values= params['evaluation']['cv']

                    scores = evaluate_model_and_log(X_test_tfidf, y_test, model, evaluate_type,metric_type, cv_values,model_type,params_and_values)


            else: 
                pass

            











if __name__ == '__main__':
    main()