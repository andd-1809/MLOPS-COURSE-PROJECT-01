import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, train_file_path, test_file_path, processed_dir, config_path):
        self.train_path = train_file_path
        self.test_path = test_file_path
        self.processed_dir = processed_dir


        self.config = read_yaml(config_path)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)

    def preprocess_data(self, df):
        try:
            logger.info("Starting data preprocessing steps")
            logger.info(f"Dropping the columns")

            df.drop(columns=['Booking_ID'], inplace=True)
            df.drop_duplicates(inplace=True)

            #TODO: check lai doan nay
            #cat_cols = self.config.get('categorical_columns', [])
            #num_cols = self.config.get('numerical_columns', [])

            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']

            logger.info(f"Encoding categorical columns: {cat_cols}")
            label_encoder = LabelEncoder()
            mappings = {}
            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label:code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
            
            logger.info(f"Label Mappings: {mappings}")
            for col, mapping in mappings.items():
                logger.info(f"Column: {col}, Mapping: {mapping}")
            
            logger.info("Doing Skiewness Handling")
            skewness_threshold = self.config.get('skewness_threshold', 5)
            skewness = df[num_cols].apply(lambda x: x.skew())
            for column in skewness[skewness>skewness_threshold].index:
                df[column] = np.log1p(df[column])
            return df
        except Exception as e:
            logger.error("Error during data preprocessing")
            raise CustomException("Failed to preprocess data", e)

    def balance_data(self, df):
        try:
            logger.info("Starting handle imbalanced data")
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']

            smote = SMOTE(random_state=369)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['booking_status'] = y_resampled

            logger.info("Data balancing completed")
            return balanced_df
        except Exception as e:
            logger.error("Error during data balancing")
            raise CustomException("Failed to balance data", e)

    def select_features(self, df):
        try:
            logger.info("Starting feature selection using RandomForestClassifier")
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']

            model = RandomForestClassifier(random_state=369)
            model.fit(X, y)

            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
            top_features_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

            num_features_to_select = self.config["data_processing"]['no_of_features']
            top_10_features = top_features_importance_df["feature"].head(num_features_to_select).values

            logger.info(f"Top {num_features_to_select} features based on importance: {top_10_features}")

            top_10_df = df[top_10_features.tolist() + ["booking_status"]]
            return top_10_df
        except Exception as e:
            logger.error("Error during feature selection")
            raise CustomException("Failed to select features", e)

    def save_processed_data(self, df, file_path):
        try:
            logger.info(f"Saving processed data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info(f"Processed data saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data to {file_path}")
            raise CustomException("Failed to save processed data", e)

    def process(self):
        try:
            logger.info("Loading data from RAW directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)
            
            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]  # Ensure test set has same features as train set

            self.save_processed_data(train_df, PROCESSED_TRAIN_FILE_PATH)
            self.save_processed_data(test_df, PROCESSED_TEST_FILE_PATH)

            logger.info("Data processing completed successfully")
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise CustomException("Data processing failed", e)
      
if __name__ == "__main__":
    try:
        data_processor = DataProcessor(
            train_file_path=TRAIN_FILE_PATH,
            test_file_path=TEST_FILE_PATH,
            processed_dir=PROCESSED_DIR,
            config_path=CONFIG_PATH
        )
        data_processor.process()
    except CustomException as e:
        logger.error(f"Error in main execution: {str(e)}")