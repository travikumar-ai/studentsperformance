from dataclasses import dataclass
import os
import sys

import logging


from src.utills import logger
from src.utills.exception_handling import CustomException

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              AdaBoostRegressor,
                              GradientBoostingRegressor)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


from sklearn.metrics import (mean_squared_error,
                             root_mean_squared_error,
                             r2_score)


from src.utills.utilities import save_object, evaluate_model

import mlflow
from mlflow.models import infer_signature


@dataclass
class ModelTrainingConfig:
    model_save_path = os.path.join(
        os.getcwd(),
        'src/artifacts',
    )
    
class Modeltrainer:
    def __init__(self) -> None:
        self.model_save_path = ModelTrainingConfig()
        
    def initiate_model_trainer(self, train_data, test_data):
        try:
            logging.info('model training started')
            
            X_train, y_train, X_test, y_test = ( train_data[:, :-1],
                                                train_data[:, -1], 
                                                test_data[:, :-1],
                                                test_data[:, -1]
                                                )
            
            models = {
                LinearRegression.__name__ : LinearRegression(),
                SVR.__name__ : SVR(),
                KNeighborsRegressor.__name__ : KNeighborsRegressor(),
                DecisionTreeRegressor.__name__ : DecisionTreeRegressor(),
                RandomForestRegressor.__name__ : RandomForestRegressor(),
                AdaBoostRegressor.__name__ : AdaBoostRegressor(),
                GradientBoostingRegressor.__name__ : GradientBoostingRegressor(),
                XGBRegressor.__name__ : XGBRegressor(),
                CatBoostRegressor.__name__ : CatBoostRegressor()
            }
            
            params = {
                LinearRegression.__name__ : {},
                SVR.__name__: {
                    'kernel': ['linear', 'rbf'], 
                    },
                KNeighborsRegressor.__name__:{
                    'n_neighbors': [3,5,7],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree',
                                       'kd_tree', 'brute'] ,
                    'leaf_size':  [10, 20, 30]
                    },
                DecisionTreeRegressor.__name__:{
                    'criterion': ['squared_error',
                                  'friedman_mse', 'absolute_error'],
                    'splitter':['best', 'random'],
                    'max_depth':[10, 20, 30]
                    
                    },
                RandomForestRegressor.__name__:{
                    'criterion': ['squared_error',
                                  'friedman_mse', 'absolute_error']
                    },
                AdaBoostRegressor.__name__: {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                    }, 
                GradientBoostingRegressor.__name__:{},
                XGBRegressor.__name__:  {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                CatBoostRegressor.__name__: {
                    'depth': [6, 8, 10],
                    'iterations': [30, 50, 100]
                    }
                }
            
            models_training_scores, models_test_scores, models_best_params = evaluate_model(X_train,
                                                                                            y_train,
                                                                                            X_test,
                                                                                            y_test,
                                                                                            models,
                                                                                            params
                                                                            )
            
            best_model_score = max(sorted(models_test_scores.values()))
            
            ## To get best model name from dict
            best_model_name = max(models_test_scores, key=models_test_scores.get)
            
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            best_model = models[best_model_name]
            
            mlflow.set_experiment('Student Performance prediction')
            with mlflow.start_run(run_name='Students Performance Prediction'):
                
                # Get best model and parameters
                best_params = models_best_params[best_model_name]
                
                # Set parameters and fit the model
                best_model.set_params(**best_params)
                best_model.fit(X_train, y_train)
                
                # Log parameters
                mlflow.log_params(best_params)
                
                predicted = best_model.predict(X_test)
                
                # Calculate and log metrics
                r2score = r2_score(y_test, predicted)
                mse = mean_squared_error(y_test, predicted)
                rmse = root_mean_squared_error(y_test, predicted)
                
                mlflow.log_metric("r2_score", r2score)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)
                
                mlflow.set_tag('model', best_model_name)
                
                # Infer signature correctly and log the model
                signature = infer_signature(X_train, best_model.predict(X_train))
                
                mlflow.sklearn.log_model(
                    sk_model= best_model,
                    artifact_path='model',
                    signature=signature,
                    registered_model_name=f'{best_model_name}'
                )
                
                if best_model_score < 0.6:
                    raise CustomException("No best model found", sys)
                logging.info(
                    f"Best found model on both training and testing dataset")
                
                save_object(
                    file_path=os.path.join(self.model_save_path.model_save_path,
                                           best_model_name + '.pkl'),
                    obj=best_model
                )
            return best_model_name, models_best_params[best_model_name]
        except Exception as e:
            raise CustomException(e, sys)
            
            
            
            
            
