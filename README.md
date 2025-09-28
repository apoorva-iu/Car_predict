Used Car Price Predictor 🚗

This project is a web application that predicts the selling price of used cars using a machine learning model deployed with Flask.

🚀 Key Features
Interactive Web Form: Users input key car features (Company, Model, Year, Kms Driven, Fuel Type).

Dynamic UI: Car Models are dynamically filtered based on the selected Company using JavaScript.

ML Prediction: Price estimation is powered by a pre-trained regression model.

Project Structure
File/Folder	Purpose
app.py	Flask App: Core application logic, routing, and model serving.
templates/	Frontend: Contains index.html (the prediction form).
static/	Web Assets: Contains css/ for styling.
model.joblib	Prediction Model: The primary trained model file used by app.py.
LinearRegressionModel.pkl	Alternative format for the trained model.
Quikr Analysis.ipynb	Analysis: Jupyter Notebook detailing EDA, feature engineering, and model training.
quikr_car.csv	Raw Data: The original, uncleaned dataset.
Cleaned Car.csv	Cleaned Data: The processed dataset used for model training.
requirements.txt	Dependencies: Lists all necessary Python packages (Flask, scikit-learn, etc.).

Setup and Installation
1. Clone the Repository
Clone your project using your GitHub URL and navigate into the project folder (car_predict):

git clone https://github.com/apoorva-iu/car_predict
cd car_predict

Configure Environment
Create and activate a Python virtual environment:

python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

Install Dependencies
Install all necessary packages:

pip install -r requirements.txt

How to Run
Ensure your virtual environment is active.

Run the Flask application from within the car_predict folder:

python app.py

Open your web browser and navigate to the local server address:

http://127.0.0.1:5000/
