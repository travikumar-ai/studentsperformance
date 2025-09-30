import os
import sys
import logging

from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.utills import logger
from src.utills.exception_handling import CustomException

import pandas as pd


data_path = os.path.join(
    os.getcwd(),
    'data',
    'data.csv'
)

artifacts_path = os.path.join(
    os.getcwd(),
    'src',
    'artifacts'
)

# os.makedirs(artifacts_path, exist_ok=True)

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join(artifacts_path, 'raw_data.csv')
    train_data_path = os.path.join(artifacts_path, 'train_data.csv')
    test_data_path = os.path.join(artifacts_path, 'test.csv')
    

class DataIngestion:
    def __init__(self) -> None:
        self.data_saving_paths = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info(f"Data Ingestion Started")

        try:
            data = pd.read_csv(data_path)

            train_data, test_data = train_test_split(data, 
                                                    random_state=42,
                                                    test_size=0.2)
            
            logging.info(f"Data Splitting done!")
            
            artifacts_dir_name = os.path.dirname(self.data_saving_paths.raw_data_path)
            os.makedirs(artifacts_dir_name, exist_ok=True)
            
            logging.info(f"Data saving started!")

            data.to_csv(self.data_saving_paths.raw_data_path, header=True, index=False)
            train_data.to_csv(self.data_saving_paths.train_data_path, header=True, index=False)
            test_data.to_csv(self.data_saving_paths.test_data_path, header=True, index=False)
            logging.info(f"Data saving done!")

            return [self.data_saving_paths.train_data_path,
                    self.data_saving_paths.test_data_path]
        
        except Exception as e:
            error_msg  = CustomException(e, sys)
            logging.info(f"Error occurred and error: [{error_msg}]")
            raise CustomException(e, sys)
        
    
