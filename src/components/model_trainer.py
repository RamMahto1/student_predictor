from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import(
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
    
)
from sklearn.svm import SVR
from dataclasses import dataclass
import os
import sys
from src.utils import save_obj, load_obj, evaluate_metrics


class ModelTrainerConfig:
    model_trainer_file_obj:str = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            
            # initiate the model
            models ={
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomforestRegressor":RandomForestRegressor(),
                "AdaboostingRegressor":AdaBoostRegressor(),
                "GrandientDecentRegressor":GradientBoostingRegressor(),
                "supportVectorMachine":SVR()
                
            }
            params = {
                "LinearRegression": {},
                "Ridge": {'alpha': [0.1, 1, 0.2]},
                "Lasso": {'alpha': [0.1, 1, 0.2]},
                "DecisionTreeRegressor": {'max_features': [3, 5, None], 
                                          'max_depth': [3, 5, 10]},
                "RandomForestRegressor": {"n_estimators": [3, 5, 10]},
                "AdaBoostRegressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 1, 0.2],
                    'base_estimator': [None],
                    'random_state': [None]
                    },
                "GradientBoostingRegressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 1, 0.2],
                    'loss': ['squared_error'],
                    'alpha': [0.9]
                    },
                "SVR": {}
                }

            
            # evaluta all the model
            report,best_model_name, best_model,best_score = evaluate_metrics(X_train,y_train,X_test,y_test,models,params)
            logging.info(f"Best model found:{best_model_name}: best score: {best_score}")
            
            # saved the model
            save_obj(file_path=self.model_trainer.model_trainer_file_obj, obj=best_model)

            return report,best_model_name,best_model,best_score
        
        except Exception as e:
            raise CustomException(e,sys)

