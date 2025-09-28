import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from urllib.parse import unquote

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Cleaned Car.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
CURRENT_YEAR = 2025

app = Flask(__name__)

# Load data
def load_data(path=DATA_PATH):
    df = pd.read_csv(r"C:\Users\Admin\OneDrive\Documents\Projects\car_predict\Cleaned Car.csv")
    for col in ['company', 'name', 'fuel_type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df.dropna(subset=['Price', 'kms_driven', 'year'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if 'car_age' not in df.columns:
        df['car_age'] = CURRENT_YEAR - df['year']
    return df

# Train model if not exists
def train_and_save_model(df, model_path=MODEL_PATH):
    X = df[['company', 'name', 'year', 'kms_driven', 'fuel_type', 'car_age']].copy()
    y = df['Price'].copy()

    cat_features = ['company', 'name', 'fuel_type']
    num_features = ['year', 'kms_driven', 'car_age']

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
        ("num", StandardScaler(), num_features)
    ])

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_path)
    return pipeline

# Load data and model
df = load_data()
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
    except:
        model = train_and_save_model(df)
else:
    model = train_and_save_model(df)

# Prepare dropdown data
companies = sorted(df['company'].dropna().unique().tolist())
company_to_models = {c: sorted(df[df['company'] == c]['name'].unique().tolist()) for c in companies}
years = sorted(df['year'].dropna().unique().astype(int).tolist(), reverse=True)
fuels = sorted(df['fuel_type'].dropna().unique().tolist())

# Routes
@app.route("/")
def index():
    return render_template("index.html",
                           companies=companies,
                           years=years,
                           fuels=fuels)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.form
        company = data.get('company')
        name = data.get('name')
        year = int(data.get('year'))
        fuel = data.get('fuel_type')
        kms = float(data.get('kms_driven'))
        car_age = CURRENT_YEAR - year

        X_new = pd.DataFrame([{
            'company': company,
            'name': name,
            'year': year,
            'kms_driven': kms,
            'fuel_type': fuel,
            'car_age': car_age
        }])

        pred = float(model.predict(X_new)[0])
        pred_rounded = int(round(pred / 1000.0) * 1000)
        return jsonify({
            'success': True,
            'predicted_price': pred_rounded,
            'predicted_price_display': f"₹ {pred_rounded:,}"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route("/get_models/<path:company>")
def get_models(company):
    # Decode URL to match original key
    company_name = unquote(company)
    models = company_to_models.get(company_name, [])
    return jsonify(models)

if __name__ == "__main__":
    app.run(debug=True)
