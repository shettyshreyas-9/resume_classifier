import pathlib
import os
import sys
import logging
import yaml

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


def save_data(data_df,output_path):
    data_df.to_csv(output_path+'/processed_data.csv')

    logging.info("Saved data successfully")




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

    # Save data
    output_path= working_dir.as_posix() + sys.argv[2]
    save_data(data_df=data_df,output_path=output_path)






if __name__ == '__main__':
    main()
