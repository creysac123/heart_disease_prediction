import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation  # Import DataTransformation for transformations

import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation  # Import the DataTransformation class

class PredictPipeline:
    def __init__(self):
        self.data_transformation = DataTransformation()  # Instantiate DataTransformation class

    def predict(self, features):
        try:
            # Load the saved model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Check if model is loaded
            if model is None:
                raise CustomException("Model loading failed. Model is None.", sys)

            # Manually encode binary features
            binary_columns = ['Smoking', 'Stroke', 'PhysicalActivity', 'DiffWalking']
            for col in binary_columns:
                features[col] = features[col].map({'Yes': 1, 'No': 0})

            # Call the ordinal encoding function from the DataTransformation class
            features = self.data_transformation.ordinal_encode_columns(features)

            # Preprocess the input features
            data_scaled = preprocessor.transform(features)
            

            # Make predictions
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, 
                 bmi: float,
                 age_category: str,
                 sleep_time: float,
                 physical_health: float,
                 mental_health: float,
                 gen_health: str,
                 diabetic: str,
                 sex: str,
                 smoking: str,
                 stroke: str,
                 physical_activity: str,
                 diff_walking: str):
        
        # Initialize with the columns required for prediction
        self.bmi = bmi
        self.age_category = age_category
        self.sleep_time = sleep_time
        self.physical_health = physical_health
        self.mental_health = mental_health
        self.gen_health = gen_health
        self.diabetic = diabetic
        self.sex = sex
        self.smoking = smoking
        self.stroke = stroke
        self.physical_activity = physical_activity
        self.diff_walking = diff_walking

    def get_data_as_data_frame(self):
        try:
            # Convert input data to a DataFrame with the appropriate structure
            custom_data_input_dict = {
                "BMI": [self.bmi],
                "AgeCategory": [self.age_category],
                "SleepTime": [self.sleep_time],
                "PhysicalHealth": [self.physical_health],
                "MentalHealth": [self.mental_health],
                "GenHealth": [self.gen_health],
                "Diabetic": [self.diabetic],
                "Sex": [self.sex],
                "Smoking": [self.smoking],
                "Stroke": [self.stroke],
                "PhysicalActivity": [self.physical_activity],
                "DiffWalking": [self.diff_walking]
            }

            # Convert the dictionary to a DataFrame for preprocessing and prediction
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
