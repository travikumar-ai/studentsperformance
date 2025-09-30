import os
import sys

import logging
from typing import Tuple
from src.utills import logger

import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.utills.exception_handling import CustomException


def save_object(file_path, obj):
    logging.info(f"saving object [{file_path}] started")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file=file_obj)
            
        logging.info(f"Object dumping done with file name [{file_obj}]")
    except Exception as e:
        error_msg =  CustomException(e, sys)
        logging.info(f"Error occurred while dumping the [{file_obj}] object ")
        raise CustomException(e, sys)

def evaluate_model(X_train, 
                   y_train, 
                   X_test, 
                   y_test, 
                   models:dict, 
                   params:dict) -> Tuple[dict, dict, dict]:
    logging.info(f"Model evaluation started")

    try:
        train_report = {}
        test_report = {}
        models_best_params = {}
        
        for model_name, model in models.items():
            model_param = params[model_name]
            
            gs = GridSearchCV(model , model_param, cv=3)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train) # Train model with best parameters

            best_params = gs.best_params_

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            model_train_accuracy = r2_score(y_train, y_train_pred)
            model_test_accuracy  = r2_score(y_test, y_test_pred)
            
            train_report[model_name] = model_train_accuracy
            test_report[model_name] = model_test_accuracy
            models_best_params[model_name] = best_params
        logging.info(f"Model evaluation done!")
        return (train_report,
                test_report,
                models_best_params)
        
    except Exception as e:
        error_msg = CustomException(e, sys)
        logging.info(f"While evaluating Model error occurred at: {error_msg} ")
        raise CustomException(e, sys)
