from src.logger import logging
from src.exception import CustomException
import os
import sys
import pandas as pd


class DataValidation:
    def __init__(self,train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        
    def initiate_data_valiation(self):
        try:
            train_df = pd.read_csv(self.train_data)
            test_df = pd.read_csv(self.test_data)
            
            logging.info(f"train data shape: \n{train_df.shape}")
            logging.info(f"test data shape:\n{test_df.shape}")
            
            # data information
            logging.info(f"train data information:\n{train_df.info()}")
            logging.info(f"test data information:\n{test_df.info()}")
            
            ## checking null value
            logging.info(f"train data missing value:\n{train_df.isnull()}")
            logging.info(f"test data missing value:\n{test_df.isnull()}")
            
            # checking duplicated value
            logging.info(f"train data duplicate value:\n{train_df.duplicated().sum()}")
            logging.info(f"test data duplicate value:\n{test_df.duplicated().sum()}")
            
            # statatics test
            logging.info(f"train data statatics info:\n{train_df.describe()}")
            logging.info(f"test data statatics info:\n{test_df.describe()}")
            
            excepted_columns = ['gender','parental_level_of_education','race_ethnicity',
                    'test_preparation_course','lunch','reading_score','writing_score']
            
            missing_columns = [col for col in excepted_columns if col not in train_df.columns]
            
            if missing_columns:
                raise CustomException(f"missing columns in train data:{missing_columns}", sys)
            
        except Exception as e:
            raise CustomException(e,sys)