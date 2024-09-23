from flask import Flask, request, render_template, flash, redirect, url_for
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Set a secret key for session management
app.secret_key = 'your_secret_key'

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collect form data
        data = CustomData(
            bmi=float(request.form.get('bmi')),
            age_category=request.form.get('age_category'),
            sleep_time=float(request.form.get('sleep_time')),
            physical_health=float(request.form.get('physical_health')),
            mental_health=float(request.form.get('mental_health')),
            gen_health=request.form.get('gen_health'),
            diabetic=request.form.get('diabetic'),
            sex=request.form.get('sex'),
            smoking=request.form.get('smoking'),
            stroke=request.form.get('stroke'),
            physical_activity=request.form.get('physical_activity'),
            diff_walking=request.form.get('diff_walking')
        )

        # Convert form data to DataFrame
        pred_df = data.get_data_as_data_frame()

        # Create a PredictPipeline instance and make a prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Prepare the prediction result
        prediction_result = 'Yes' if results[0] == 1 else 'No'
        
        # Flash the prediction result
        flash(f'The predicted heart disease outcome is: {prediction_result}')
        return redirect(url_for('predict_datapoint'))  # Redirect to the same route

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)
