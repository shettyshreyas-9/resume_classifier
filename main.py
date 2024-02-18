import pathlib
import sys
import os
import joblib

import pandas as pd
import numpy as np

import streamlit as st



def get_latest_best_model_path(evaluate_log_file):
    with open(evaluate_log_file,'r') as log_file:
        lines= log_file.readlines()
        for line in reversed(lines):
            if 'Best model path' in line:
                return line.split('=')[-1].strip()
            else:
                return 'unavailable'


def main():

    curr_dir = pathlib.Path(__file__)
    working_dir = pathlib.Path(os.getcwd())

    evaluate_log_file= working_dir.as_posix()+sys.argv[1]
    print(evaluate_log_file)

    best_model_path= get_latest_best_model_path(evaluate_log_file)
    print(best_model_path)

    # Check if the model file exists
    if os.path.exists(best_model_path):

        #Load the best model
        best_model= joblib.load(best_model_path)

    st.title('Resume Classifier')


if __name__=='__main__':
    main()
