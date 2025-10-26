import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgbm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import load_data
from scipy.stats import randint

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        
        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f"Loading training and testing data from {self.train_path}")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            target = 'booking_status'
            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]

            X_test = test_df.drop(columns=[target])
            y_test = test_df[target]

            logger.info("Data loading completed.")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error in loading and splitting data: {e}")
            raise CustomException(f"Error in loading and splitting data: {e}")

    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Starting model training...")
            lgbm_model = lgbm.LGBMClassifier(random_state=self.random_search_params['random_state'])
            
            logger.info("Performing hyperparameter tuning using RandomizedSearchCV...")
            random_search = RandomizedSearchCV(
                estimator           =   lgbm_model,
                param_distributions =   self.params_dist,
                n_iter              =   self.random_search_params['n_iter'],
                cv                  =   self.random_search_params['cv'],
                n_jobs              =   self.random_search_params['n_jobs'],
                verbose             =   self.random_search_params['verbose'],
                random_state        =   self.random_search_params['random_state'],
                scoring             =   self.random_search_params['scoring']
            )

            logger.info("Starting our model training...")
            random_search.fit(X_train, y_train)

            logger.info("Model training completed.")
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best hyperparameters: {best_params}")
            return best_lgbm_model
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise CustomException(f"Error in model training: {e}")

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating our model...")
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            logger.info(f"Model evaluation completed. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise CustomException(f"Error in model evaluation: {e}")

    def save_model(self, model):
        try:
            logger.info(f"Saving model to {self.model_output_path}")
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved at {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error in saving model: {e}")
            raise CustomException(f"Error in saving model: {e}")
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting the model training pipeline...")
                logger.info("Starting MLFLOW experimentation.")
                logger.info("Logging the trainging and testing data set to MLFLOW.")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)

                logger.info("Logging model to MLFLOW.")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging parameters and metrics to MLFLOW.")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model training pipeline completed successfully.")
                return metrics
        except Exception as e:
            logger.error(f"Error in running model training pipeline: {e}")
            raise CustomException(f"Error in running model training pipeline: {e}")
      
if __name__ == "__main__":
    trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_FILE_PATH,
        test_path=PROCESSED_TEST_FILE_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    trainer.run()