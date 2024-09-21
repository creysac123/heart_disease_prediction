import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, df):
        try:
            logging.info(f"Initial dataframe shape: {df.shape}")

            columns_to_keep = ['BMI', 'AgeCategory', 'SleepTime', 'PhysicalHealth', 'MentalHealth',
                               'GenHealth', 'Diabetic', 'Sex', 'Smoking', 'Stroke', 'PhysicalActivity', 
                               'DiffWalking', 'HeartDisease']
            df = df[columns_to_keep]
            logging.info(f"Dataframe shape after dropping columns: {df.shape}")
            logging.info(f"Data types:\n{df.dtypes}")
            
            
            df = self.impute_missing_values(df)
            logging.info(f"Missing value imputation completed.\nMissing values after imputation:\n{df.isnull().sum()}")

            df_encoded = self.binary_encode_columns(df)
            logging.info(f"Data types after binary encoding:\n{df_encoded.dtypes}")

            df_encoded = self.ordinal_encode_columns(df_encoded)
            logging.info(f"Data types after ordinal encoding:\n{df_encoded.dtypes}")

            # Convert all columns to numeric types before scaling
            df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')

            # Apply preprocessing and scale features
            
            X = df_encoded.drop('HeartDisease', axis=1)
            y = df_encoded['HeartDisease']
            preprocessing_obj = self.get_data_transformer_object()
            X = pd.DataFrame(preprocessing_obj.fit_transform(X), 
                                                  columns=X.columns)

            logging.info(f"Dataframe shape after transformation: {X.shape}")

            # Save the preprocessor object
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)
            logging.info(f"Preprocessor object saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            return X,y, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise CustomException(e, sys)

    def impute_missing_values(self, df):
        try:
            numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
            categorical_features = df.select_dtypes(include=[object, 'category']).columns
            
            # Impute numeric features
            numeric_imputer = SimpleImputer(strategy='median')
            df.loc[:, numeric_features] = numeric_imputer.fit_transform(df[numeric_features])
            
            # Impute categorical features
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df.loc[:, categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
            
            return df
        
        except Exception as e:
            logging.error(f"Error during missing value imputation: {e}")
            raise CustomException(e, sys)

    def binary_encode_columns(self, df):
        try:
            logging.info(f"Data types before binary encoding:\n{df.dtypes}")
            binary_columns = ['Smoking', 'Stroke', 'PhysicalActivity', 'DiffWalking', 'HeartDisease']
            
            # Convert to string type first using .loc
            df.loc[:, binary_columns] = df.loc[:, binary_columns].astype(str)
            
            # Apply mapping and convert to float
            df.loc[:, binary_columns] = df.loc[:, binary_columns].apply(lambda col: col.map({'Yes': 1, 'No': 0}).astype(float))
            
            return df
        except Exception as e:
            logging.error(f"Error during binary encoding: {e}")
            raise CustomException(e, sys)

    def binary_encode_columns(self, df):
        try:
            binary_columns = ['Smoking', 'Stroke', 'PhysicalActivity', 'DiffWalking', 'HeartDisease']
            df.loc[:, binary_columns] = df[binary_columns].apply(lambda col: col.map({'Yes': 1, 'No': 0}).astype(float))
            return df
        except Exception as e:
            logging.error(f"Error during binary encoding: {e}")
            raise e

    def ordinal_encode_columns(self, df):
        try:
            # Define ordinal mappings for specific columns
            age_mapping = {'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4,
                           '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9,
                           '70-74': 10, '75-79': 11, '80 or older': 12}
            diabetic_mapping = {'No': 0, 'No, borderline diabetes': 1, 'Yes (during pregnancy)': 2, 'Yes': 3}
            genhealth_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}
            sex_mapping = {'Female': 0, 'Male': 1}
            # Apply mappings
            df.loc[:, 'AgeCategory'] = df['AgeCategory'].map(age_mapping).astype(float)
            df.loc[:, 'Diabetic'] = df['Diabetic'].map(diabetic_mapping).astype(float)
            df.loc[:, 'GenHealth'] = df['GenHealth'].map(genhealth_mapping).astype(float)
            df.loc[:, 'Sex'] = df['Sex'].map(sex_mapping).astype(float)
            return df
        except Exception as e:
            logging.error(f"Error during ordinal encoding: {e}")
            raise e



    def get_data_transformer_object(self):
        try:
            column_transformer = ColumnTransformer(
                transformers=[('scaler', StandardScaler(), 
                               ['BMI', 'PhysicalHealth', 'MentalHealth', 'AgeCategory', 
                                'Diabetic', 'GenHealth', 'SleepTime'])],
                remainder='passthrough'  # Keep other columns unchanged
            )
            return column_transformer
        except Exception as e:
            logging.error(f"Error creating transformer object: {e}")
            raise CustomException(e, sys)
