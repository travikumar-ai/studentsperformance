
from src.components import data_transformation
from src.components.data_ingestion import DataIngestion
from src.components.model_training import Modeltrainer

import logging
from src.utills import logger


def model_training():
    
    data_ingestion = DataIngestion()
    
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    # print(train_data_path, test_data_path, sep='\n')
    
    data_transform = data_transformation.DataTransform()
    train_arr, test_arr, preprocessor_path = data_transform.initiate_data_transfer(train_data_path, test_data_path)
    # print(preprocessor_path)
    
    model_trainer = Modeltrainer()
    best_model_name, model_params = model_trainer.initiate_model_trainer(
                                                                    train_arr, test_arr
                                                                    )
    
    print(f"Best Model: {best_model_name}")
    print(f"Model Parameters: {model_params}")


if __name__ == '__main__':
    model_training()