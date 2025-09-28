Car Price Predictor 🚗

This is a web application that predicts the resale price of a used car. The project utilizes a Flask backend to serve predictions from a machine learning model and a dynamic frontend built with HTML/JavaScript for user interaction.

📂 Project Structure
File/Folder	Purpose
app.py	Flask Application: Core logic for routing, model loading, and handling price prediction requests.
templates/	Frontend: Contains index.html, the main prediction form.
static/	Web Assets: Contains css/ for all project styling.
model.joblib	Trained Model: The primary saved machine learning regression model.
Quikr Analysis.ipynb	Analysis: Jupyter Notebook documenting the entire process: EDA, cleaning, feature engineering, and model training.
requirements.txt	Lists all necessary Python dependencies (Flask, scikit-learn, pandas, etc.).

Export to Sheets
⚙️ Setup and Installation
1. Clone the Repository
Clone your project from GitHub and navigate into the main project folder (car_predict):

Bash

git clone https://github.com/apoorva-iu/car_predict
cd car_predict 
2. Configure Environment (Recommended)
It's best practice to use a virtual environment to isolate project dependencies:

Bash

# Create and activate the environment
python3 -m venv venv
source venv/bin/activate  # Use 'venv\Scripts\activate' on Windows
3. Install Dependencies
Install all required Python packages:

Bash

pip install -r requirements.txt
▶️ How to Run the Application
Ensure your virtual environment is active (if you used one).

Run the Flask application:

Bash

python app.py
The application will be available at:

http://127.0.0.1:5000/






