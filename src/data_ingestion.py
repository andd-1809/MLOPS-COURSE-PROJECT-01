import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config.get('bucket_name')
        self.bucket_file_name = self.config.get('bucket_file_name')
        self.train_ratio = float(self.config.get('train_ratio', 0.8))


        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info("DataIngestion initialized with {self.bucket_name} and file is {self.bucket_file_name}, train_ratio={self.train_ratio}")

    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"Data downloaded from bucket {self.bucket_name} to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error("Error downloading data from GCS")
            raise CustomException("Failed to download data from GCS", e)

    def split_data(self):
        try:
            logger.info("Starting data split into train and test sets")
            df = pd.read_csv(RAW_FILE_PATH)
            train_df, test_df = train_test_split(df, train_size=self.train_ratio, random_state=369)
            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Data split into train and test sets with ratio {self.train_ratio}")
            logger.info(f"Train data saved to {TRAIN_FILE_PATH}, Test data saved to {TEST_FILE_PATH}")
        except Exception as e:
            logger.error("Error splitting data into train and test sets")
            raise CustomException("Failed to split data", e)

    def run(self):
        try:
            logger.info("Starting data ingestion process")

            self.download_csv_from_gcp()
            self.split_data()

            logger.info("Data ingestion process completed successfully")
            return TRAIN_FILE_PATH, TEST_FILE_PATH
        except CustomException as e:
            logger.error(f"Error in data ingestion process: {str(e)}")
        finally:
            logger.info("Data ingestion process finished")

if __name__ == "__main__":
    try:
        config = read_yaml(CONFIG_PATH)
        data_ingestion = DataIngestion(config)
        train_path, test_path = data_ingestion.run()
        logger.info(f"Data ingestion completed. Train path: {train_path}, Test path: {test_path}")
    except CustomException as ce:
        logger.error(f"CustomException caught in main: {ce}")