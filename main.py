from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation


def main():
    try:
        ## step: 1 data ingestion
        data_ingestion = DataIngestion()
        train_data,test_data = data_ingestion.initiate_data_ingestion()
        
        ## step: 2 data validation
        data_validation =DataValidation(train_data,test_data)
        data_validation.initiate_data_valiation()
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

