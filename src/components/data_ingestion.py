from dataclasses import dataclass
import logging
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

from src.utills import logger
from src.utills.exception_handling import CustomException

config = yaml.safe_load(open('config/params.yaml'))
data_config = config['data']

src_config = config['src']

data_path = os.path.join(
    os.getcwd(),
    data_config['data_dir'],
    data_config['data_file_name']
)

artifacts_path = os.path.join(
    os.getcwd(),
    src_config['src_path'],
    src_config['artifacts']['artifacts_path']
)

# os.makedirs(artifacts_path, exist_ok=True)

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join(artifacts_path, data_config['raw_data_file_name'])
    train_data_path = os.path.join(artifacts_path, data_config['train_data_file_name'])
    test_data_path = os.path.join(artifacts_path, data_config['test_data_file_name'])


class DataIngestion:
    def __init__(self) -> None:
        self.data_saving_paths = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")

        try:
            data = pd.read_csv(data_path)

            train_data, test_data = train_test_split(data, 
                                                    random_state=42,
                                                    test_size=0.2)
            
            logging.info("Data Splitting done!")
            
            artifacts_dir_name = os.path.dirname(self.data_saving_paths.raw_data_path)
            os.makedirs(artifacts_dir_name, exist_ok=True)
            
            logging.info("Data saving started!")

            data.to_csv(self.data_saving_paths.raw_data_path, header=True, index=False)
            train_data.to_csv(self.data_saving_paths.train_data_path, header=True, index=False)
            test_data.to_csv(self.data_saving_paths.test_data_path, header=True, index=False)
            logging.info("Data saving done!")

            return [self.data_saving_paths.train_data_path,
                    self.data_saving_paths.test_data_path]
        
        except Exception as e:
            error_msg  = CustomException(e, sys)
            logging.info(f"Error occurred and error: [{error_msg}]")
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
