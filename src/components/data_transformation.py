from dataclasses import dataclass
import os
import sys

import logging
from src.utills import logger
from src.utills.exception_handling import CustomException

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.utills.utilities import save_object, evaluate_model



@dataclass
class data_transformation_config:
    processor_obj_file_path = os.path.join(
        os.getcwd(),
        'src/artifacts',
        'processor.pkl'
    )
    
class DataTransform:
    def __init__(self) -> None:
        self.data_transformation_config = data_transformation_config()
        
    def get_data_tranformation_object(self):
        try:
            numerical_columns = ['reading_score',
                                 'writing_score']
            categorical_columns = ['gender', 
                                   'race_ethnicity', 
                                   'parental_level_of_education', 
                                   'lunch',
                                   'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps= [
                    ('imputr', SimpleImputer(strategy='mean') ),
                    ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ( 'impute', SimpleImputer(strategy='most_frequent') ),
                    ( 'oneHotEncoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            
            logging.info(f'Categorical Columns: {categorical_columns}')
            logging.info(f'Categorical Columns: {numerical_columns}')
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ( 'num_pipeline', num_pipeline, numerical_columns ),
                    ( 'cat_pipeline', cat_pipeline, categorical_columns)
                ], remainder='passthrough'
            )
            
            logging.info(f'Preprocessor created')
            return preprocessor
        except Exception as e:
            error_msg = CustomException(e, sys)
            logging.error('Error occurred while making transformation object')
            logging.error(f'Transformation error: {error_msg}')
            raise CustomException(e, sys)
    
    def initiate_data_transfer(self, train_data_path, test_data_path):
        
        logging.info('Data Transformation initiated')
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            
            preprocessor = self.get_data_tranformation_object()
            
            target_name = 'math_score'
            
            numerical_columns = ['reading_score',
                                 'writing_score']
            
            train_input_features_df = train_data.drop(columns= [target_name],
                                                      axis=1)
            train_target_feature = train_data[target_name]
            
            test_input_features_df = test_data.drop(columns=[target_name], axis=1)
            test_target_feature_df = test_data[target_name]
            
            logging.info(
                'Applying preprocessing object on training and testing dataframe')
            
            input_features_train_arr = preprocessor.fit_transform(train_input_features_df)
            input_features_test_arr = preprocessor.fit_transform(test_input_features_df)

            train_arr = np.c_[ input_features_train_arr, np.array(train_target_feature) ]
            test_arr = np.c_[ input_features_test_arr, np.array(test_target_feature_df) ]
            
            save_object( self.data_transformation_config.processor_obj_file_path, preprocessor )
            
            logging.info(f'saved the preprocessor with file_name:\
                [{self.data_transformation_config.processor_obj_file_path}]')
            return (train_arr,
                    test_arr,
                    self.data_transformation_config.processor_obj_file_path)
        except Exception as e:
            error_msg = CustomException(e, sys)
            logging.error(f'Error occurred at: [{error_msg}]')
            raise CustomException(e, sys)
