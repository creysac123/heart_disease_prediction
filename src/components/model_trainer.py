import os
import sys
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from imblearn.over_sampling import SMOTE

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, df):
        try:
            logging.info("Starting model training process...")

            # Check for NaN or empty string values
            nan_counts = df.isna().sum()
            empty_counts = (df == '').sum()
            
            if nan_counts.any() or empty_counts.any():
                logging.info("Data contains NaN or empty string values:")
                if nan_counts.any():
                    logging.info(f"NaN values:\n{nan_counts[nan_counts > 0]}")
                if empty_counts.any():
                    logging.info(f"Empty string values:\n{empty_counts[empty_counts > 0]}")

            # Split the DataFrame into features and target
            X = df.drop(columns='HeartDisease')
            y = df['HeartDisease']

            logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

            # Apply SMOTE to the entire dataset
            logging.info("Applying SMOTE to handle class imbalance")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logging.info(f"SMOTE resampling completed. Resampled data shape: {X_resampled.shape}")

            # Initialize the XGBoost model with best parameters
            best_params = {
                'learning_rate': 0.1,
                'max_depth': 7,
                'n_estimators': 300,
                'subsample': 0.8
            }
            xgb_model = xgb.XGBClassifier(random_state=42, **best_params)

            # Use cross_val_predict to get predictions across all cross-validation folds
            y_pred = cross_val_predict(xgb_model, X_resampled, y_resampled, cv=5, method='predict')

            # Classification report
            logging.info("Classification Report:")
            report = classification_report(y_resampled, y_pred)
            logging.info(report)

            # Confusion Matrix
            logging.info("Confusion Matrix:")
            cm = confusion_matrix(y_resampled, y_pred)
            logging.info(cm)

            # Calculate recall for the positive class (class 1)
            recall = recall_score(y_resampled, y_pred, pos_label=1)
            logging.info(f"Recall for class 1: {recall:.4f}")

            # Save the best model
            logging.info("Saving the best model to the specified file path")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=xgb_model
            )

            return recall

        except Exception as e:
            logging.error(f"An error occurred during model training: {str(e)}")
            raise CustomException(e, sys)
