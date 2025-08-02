from src.logger import logging
from src.exception import CustomException
import os
import pickle
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_metrics(X_train,y_train,X_test,y_test,models,params):
    try:
        report = []
        best_score =float('-inf')
        best_model_name = None
        best_model = None
        
        for model_name,model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            param = params.get(model_name, {})
            if param:
                gs = GridSearchCV(model,param_grid=param,cv=5,scoring="r2",n_jobs=-1)
                gs.fit(X_train,y_train)
                current_model = gs.best_estimator_
                logging.info(f"Best parameter:{best_model_name}: {gs.best_params_}")
            else:
                model.fit(X_train,y_train)
                current_model = model
            y_pred = current_model.predict(X_test)
            
            ## evaluate the model 
            r2score = r2_score(y_test,y_pred)
            mse = mean_squared_error(y_test,y_pred)
            mae = mean_absolute_error(y_test,y_pred)
            rmse = np.sqrt(mse)
            if r2score > best_score:
                best_score = r2score
                best_model_name = model_name
                best_model = current_model
            # appending the report
            report.append({
                "Model Name": model_name,
                "r2_score":r2score,
                "mean_squared_error":mse,
                "mean_absolute_error":mae,
                "root_mean_squared_error":rmse
            })
        return report,best_model_name,best_model,best_score
    except Exception as e:
        raise CustomException(e,sys)
    
    
def load_obj(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)