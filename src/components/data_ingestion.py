import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


# Configuration class for Data Ingestion
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')  # Path to the raw dataset
    preprocessed_data_path: str = os.path.join('artifacts', 'processed_data.csv')  # Path to save the preprocessed dataset

# DataIngestion Class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Started data ingestion method")
        try:
            # Load the dataset from raw data path
            df = pd.read_csv('notebook/data/heart_disease_data.csv')
            logging.info(f"Loaded dataset from {self.ingestion_config.raw_data_path}. Shape: {df.shape}")

            # Drop duplicate rows
            df.drop_duplicates(inplace=True)
            logging.info(f"Dropped duplicates. New shape: {df.shape}")

            # Replace empty strings with pd.NA
            df.replace("", pd.NA, inplace=True)
            logging.info("Replaced empty strings with pd.NA for missing values")

            # Convert categorical columns to 'category' data type
            categorical_columns = [
                'HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke',
                'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic',
                'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease',
                'SkinCancer'
            ]
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].astype(object)
            logging.info("Converted categorical columns to object type")

            # Convert numeric columns to numeric type
            numeric_columns = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            logging.info("Converted numeric columns to numeric type")

            # Save the preprocessed data
            df.to_csv(self.ingestion_config.preprocessed_data_path, index=False)
            logging.info(f"Preprocessed data saved to {self.ingestion_config.preprocessed_data_path}")

            return df  # Returning the dataframe for further use
        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)



if __name__ == "__main__":
    try:
        # Data Ingestion
        print("Starting data ingestion...")
        obj = DataIngestion()
        train_data = obj.initiate_data_ingestion()
        print("Data ingestion completed.")

        # Data Transformation (consider running this in batches or using GPU-accelerated methods)
        print("Starting data transformation...")
        data_transformation = DataTransformation()
        X,y, _ = data_transformation.initiate_data_transformation(train_data)

        print("Data transformation completed.")
        
        # Model Training
        print("Starting model training...")

        # Initialize ModelTrainer class
        modeltrainer = ModelTrainer()

        # Since we are using the entire dataset without a train-test split
        # we will only pass the complete `train_arr` (as a numpy array) to `initiate_model_trainer`
        recall = modeltrainer.initiate_model_trainer(X,y)


        # Print out the final recall score
        print(f"Model training completed with Recall score: {recall}")

    
    except Exception as e:
        print(f"An error occurred: {e}")