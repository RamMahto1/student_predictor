from src.logger import  logging
from src.exception import CustomException
import os
import sys
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import pandas as pd 
import numpy as np
from src.utils import save_obj

@dataclass

class DataTransformationConfig:
    preprocessor_file_path_obj:str = os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        try:
            numerical_features = ['reading_score','writing_score']
            categorical_features = ['gender','parental_level_of_education',
                                    'race_ethnicity','test_preparation_course','lunch']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ("num",num_pipeline,numerical_features),
                    ("cat",cat_pipeline,categorical_features)
                ]
            )
            
            logging.info(f"numerical feature:{numerical_features}")
            logging.info(f"categorical feature:{categorical_features}")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read data as data frame")
            logging.info("obtaining preprocessor obj")
            preprocessor_obj=self.get_data_transformer_obj()
            
            target_columns= ['math_score']
            
            input_feature_train_df = train_df.drop(columns=target_columns,axis=1)
            target_feature_train_df = train_df[target_columns]
            
            input_feature_test_df = test_df.drop(columns=target_columns,axis=1)
            target_feature_test_df = test_df[target_columns]
            
            logging.info(f"applying preprocessor obj on training and testing data")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr,target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_df.to_numpy()]
            
            
            save_obj(
                file_path=self.data_transformation_config.preprocessor_file_path_obj,
                obj=preprocessor_obj
            )
            
            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_file_path_obj
            )
            
            
            
            
            
            
        except Exception as e:
            raise CustomException(e,sys)