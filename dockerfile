# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /heart_disease

# Copy only the necessary files
COPY artifacts/model.pkl /heart_disease/artifacts/
COPY artifacts/preprocessor.pkl /heart_disease/artifacts/


# Copy app.py (or your entry point file)
COPY app.py /heart_disease/

# Copy the templates folder
COPY templates /heart_disease/templates

# Copy utils.py, logger.py, and exception.py from src folder
COPY src /heart_disease/src

# Copy requirements.txt
COPY requirements.txt /heart_disease/

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]
