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

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB



def setup_logging():
    # configure logging to a file and also print to the console
    log_file_path = pathlib.Path(__file__).parent.as_posix()+ sys.argv[4]    # Specify the path to your log file
    logging.basicConfig(
        filename= log_file_path,
        level= logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


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
    data_file = working_dir.as_posix()+ sys.argv[1]
    print(data_file)

if __name__ == '__main__':
    main()