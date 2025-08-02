from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import save_obj, load_obj, evaluate_metrics
from src.components.model_trainer import ModelTrainer

def main():
    try:
        ## step: 1 data ingestion
        data_ingestion = DataIngestion()
        train_data,test_data = data_ingestion.initiate_data_ingestion()
        
        ## step: 2 data validation
        data_validation =DataValidation(train_data,test_data)
        data_validation.initiate_data_valiation()
        
        # step: 3 Data Transforamtion
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr,test_arr)
        logging.info("model trained sucessfully")
        
        
        
    

        
    except Exception as e:
        raise CustomException(e, sys)   

if __name__=="__main__":
    main()



# try:
#     a = 2/0
# except Exception as e:
#     raise CustomException(e,sys)

# logging.info("Zero divide by 2")
# logging.info("Logging has started")

