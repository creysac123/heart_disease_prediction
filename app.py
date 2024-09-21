from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Debugging: Print the form data received
        print("Form Data:", request.form)

        # Collect form data
        data = CustomData(
            bmi=float(request.form.get('bmi')),  # Ensure 'bmi' matches the expected parameter name
            age_category=request.form.get('age_category'),  # Ensure 'age_category' matches
            sleep_time=float(request.form.get('sleep_time')),  # Ensure 'sleep_time' matches
            physical_health=float(request.form.get('physical_health')),  # Ensure 'physical_health' matches
            mental_health=float(request.form.get('mental_health')),  # Ensure 'mental_health' matches
            gen_health=request.form.get('gen_health'),  # Ensure 'gen_health' matches
            diabetic=request.form.get('diabetic'),  # Ensure 'diabetic' matches
            sex=request.form.get('sex'),  # Ensure 'sex' matches
            smoking=request.form.get('smoking'),  # Ensure 'smoking' matches
            stroke=request.form.get('stroke'),  # Ensure 'stroke' matches
            physical_activity=request.form.get('physical_activity'),  # Ensure 'physical_activity' matches
            diff_walking=request.form.get('diff_walking')  # Ensure 'diff_walking' matches
        )

        # Convert form data to DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Create a PredictPipeline instance and make a prediction
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Return the prediction result
        return render_template('home.html', results= {"Yes" if results[0] == 1 else "No"})



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
