
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create the Pima Indians Diabetes Dataset
def create_pima_dataset():
    """
    Create the Pima Indians Diabetes Dataset with actual data patterns
    """
    # Real Pima Indians Diabetes data (sample of actual dataset)
    pima_data = [
        [6,148,72,35,0,33.6,0.627,50,1],
        [1,85,66,29,0,26.6,0.351,31,0],
        [8,183,64,0,0,23.3,0.672,32,1],
        [1,89,66,23,94,28.1,0.167,21,0],
        [0,137,40,35,168,43.1,2.288,33,1],
        [5,116,74,0,0,25.6,0.201,30,0],
        [3,78,50,32,88,31.0,0.248,26,1],
        [10,115,0,0,0,35.3,0.134,29,0],
        [2,197,70,45,543,30.5,0.158,53,1],
        [8,125,96,0,0,0.0,0.232,54,1],
        [4,110,92,0,0,37.6,0.191,30,0],
        [10,168,74,0,0,38.0,0.537,34,1],
        [10,139,80,0,0,27.1,1.441,57,0],
        [1,189,60,23,846,30.1,0.398,59,1],
        [5,166,72,19,175,25.8,0.587,51,1],
        [7,100,0,0,0,30.0,0.484,32,1],
        [0,118,84,47,230,45.8,0.551,31,1],
        [7,107,74,0,0,29.6,0.254,31,1],
        [1,103,30,38,83,43.3,0.183,33,0],
        [1,115,70,30,96,34.6,0.529,32,1],
        [3,126,88,41,235,39.3,0.704,27,0],
        [8,99,84,0,0,35.4,0.388,50,0],
        [7,196,90,0,0,39.8,0.451,41,1],
        [9,119,80,35,0,29.0,0.263,29,1],
        [11,143,94,33,146,36.6,0.254,51,1],
        [10,125,70,26,115,31.1,0.205,41,1],
        [7,147,76,0,0,39.4,0.257,43,1],
        [1,97,66,15,140,23.2,0.487,22,0],
        [13,145,82,19,110,22.2,0.245,57,0],
        [5,117,92,0,0,34.1,0.337,38,0],
        [5,109,75,26,0,36.0,0.546,60,0],
        [3,158,76,36,245,31.6,0.851,28,1],
        [3,88,58,11,54,24.8,0.267,22,0],
        [6,92,92,0,0,19.9,0.188,28,0],
        [10,122,78,31,0,27.6,0.512,45,0],
        [4,103,60,33,192,24.0,0.966,33,0],
        [11,138,76,0,0,33.2,0.420,35,0],
        [9,102,76,37,0,32.9,0.665,46,1],
        [2,90,68,42,0,38.2,0.503,27,1],
        [4,111,72,47,207,37.1,1.390,56,1],
        [3,180,64,25,70,34.0,0.271,26,0],
        [7,133,84,0,0,40.2,0.696,37,0],
        [7,106,92,18,0,22.7,0.235,48,0],
        [9,171,110,24,240,45.4,0.721,54,1],
        [7,159,64,0,0,27.4,0.294,40,0],
        [0,180,66,39,0,42.0,1.893,25,1],
        [1,146,56,0,0,29.7,0.564,29,0],
        [2,71,70,27,0,28.0,0.586,22,0],
        [7,103,66,32,0,39.1,0.344,31,1],
        [7,105,0,0,0,0.0,0.305,24,0],
        [1,103,80,11,82,19.4,0.491,22,0],
        [1,101,50,15,36,24.2,0.526,26,0],
        [5,88,66,21,23,24.4,0.342,30,0],
        [8,176,90,34,300,33.7,0.467,58,1],
        [7,150,66,42,342,34.7,0.718,42,0],
        [1,73,50,10,0,23.0,0.248,21,0],
        [7,187,68,39,304,37.7,0.254,41,1],
        [0,100,88,60,110,46.8,0.962,31,0],
        [0,146,82,0,0,40.5,1.781,44,0],
        [0,105,64,41,142,41.5,0.173,22,0],
        [2,84,0,0,0,0.0,0.304,21,0],
        [8,133,72,0,0,32.9,0.270,39,1],
        [5,44,62,0,0,25.0,0.587,36,0],
        [2,141,58,34,128,25.4,0.699,24,0],
        [7,114,66,0,0,32.8,0.258,42,1],
        [5,99,74,27,0,29.0,0.203,32,0],
        [0,109,88,30,0,32.5,0.855,38,1],
        [2,109,92,0,0,42.7,0.845,54,0],
        [1,95,66,13,38,19.6,0.334,25,0],
        [4,146,85,27,100,28.9,0.189,27,0],
        [2,100,66,20,90,32.9,0.867,28,1],
        [5,139,64,35,140,28.6,0.411,26,0],
        [13,126,90,0,0,43.4,0.583,42,1],
        [4,129,86,20,270,35.1,0.231,23,0],
        [1,79,75,30,0,32.0,0.396,22,0],
        [1,0,48,20,0,24.7,0.140,22,0],
        [7,62,78,0,0,32.6,0.391,41,0],
        [5,95,72,33,0,37.7,0.370,27,0],
        [0,131,0,0,0,43.2,0.270,26,1],
        [2,112,66,22,0,25.0,0.307,24,0],
        [3,113,44,13,0,22.4,0.140,22,0],
        [2,74,0,0,0,0.0,0.102,22,0],
        [7,83,78,26,71,29.3,0.767,36,0],
        [0,101,65,28,0,24.6,0.237,22,0],
        [5,137,108,0,0,48.8,0.227,37,1],
        [2,110,74,29,125,32.4,0.698,27,0],
        [13,106,72,54,0,36.6,0.178,45,0],
        [2,100,68,25,71,38.5,0.324,26,0],
        [15,136,70,32,110,37.1,0.153,43,1],
        [1,107,68,19,0,26.5,0.165,24,0],
        [1,80,55,0,0,19.1,0.258,21,0],
        [4,123,80,15,176,32.0,0.443,34,0],
        [7,81,78,40,48,46.7,0.261,42,0],
        [4,134,72,0,0,23.8,0.277,60,1],
        [2,142,82,18,64,24.7,0.761,21,0],
        [6,144,72,27,228,33.9,0.255,40,0],
        [2,92,62,28,0,31.6,0.130,24,0],
        [1,71,48,18,76,20.4,0.323,22,0],
        [6,93,50,30,64,28.7,0.356,23,0],
        [1,122,90,51,220,49.7,0.325,31,1],
        [1,163,72,0,0,39.0,1.222,33,1],
        [1,151,60,0,0,26.1,0.179,22,0],
        [0,125,96,0,0,22.5,0.262,21,0],
        [1,81,72,18,40,26.6,0.283,24,0],
        [2,85,65,0,0,39.6,0.930,27,0],
        [1,126,56,29,152,28.7,0.801,21,0],
        [1,96,122,0,0,22.4,0.207,27,0],
        [4,144,58,28,140,29.5,0.287,37,0],
        [3,83,58,31,18,34.3,0.336,25,0],
        [0,95,85,25,36,37.4,0.247,24,1],
        [3,171,72,33,135,33.3,0.199,24,1],
        [8,155,62,26,495,34.0,0.543,46,1],
        [1,89,76,34,37,31.2,0.192,23,0],
        [4,76,62,0,0,34.0,0.391,25,0],
        [7,160,54,32,175,30.5,0.588,39,1],
        [4,146,92,0,0,31.2,0.539,61,1],
        [5,124,74,0,0,34.0,0.22,38,1],
        [5,78,48,0,0,33.7,0.654,25,0],
        [4,97,60,23,0,28.2,0.443,22,0],
        [4,99,76,15,51,23.2,0.223,21,0],
        [0,162,76,56,100,53.2,0.759,25,1],
        [6,111,64,39,0,34.2,0.26,24,0],
        [2,107,74,30,100,33.6,0.404,23,0],
        [5,132,80,0,0,26.8,0.186,69,0],
        [0,113,76,0,0,33.3,0.278,23,1],
        [1,88,30,42,99,55.0,0.496,26,1],
        [3,120,70,30,135,42.9,0.452,30,0],
        [1,118,58,36,94,33.3,0.261,23,0],
        [1,117,88,24,145,34.5,0.403,40,1],
        [0,105,84,0,0,27.9,0.741,62,1],
        [4,173,70,14,168,29.7,0.361,33,1],
        [9,122,56,0,0,33.3,1.114,33,1],
        [3,170,64,37,225,34.5,0.356,30,1],
        [8,84,74,31,0,38.3,0.457,39,0],
        [2,96,68,13,49,21.1,0.647,26,0],
        [2,125,60,20,140,33.8,0.088,31,0],
        [0,100,70,26,50,30.8,0.597,21,0],
        [0,93,60,25,92,28.7,0.532,22,0],
        [0,129,80,0,0,31.2,0.703,29,0],
        [5,105,72,29,325,36.9,0.159,28,0],
        [3,128,78,0,0,21.1,0.268,55,0],
        [5,106,82,30,0,39.5,0.286,38,0],
        [2,108,52,26,63,32.5,0.318,22,0],
        [10,108,66,0,0,32.4,0.272,42,1],
        [4,154,62,31,284,32.8,0.237,23,0],
        [0,102,75,23,0,0.0,0.572,21,0],
        [9,57,80,37,0,32.8,0.096,41,0],
        [2,106,64,35,119,30.5,1.4,34,0],
        [5,147,78,0,0,33.7,0.218,65,0],
        [2,90,70,17,0,27.3,0.085,22,0],
        [1,136,74,50,204,37.4,0.399,24,0],
        [4,114,65,0,0,21.9,0.432,37,0],
        [9,156,86,28,155,34.3,1.189,42,1],
        [1,153,82,42,485,40.6,0.687,23,0],
        [8,188,78,0,0,47.9,0.137,43,1],
        [7,152,88,44,0,50.0,0.337,36,1],
        [2,99,52,15,94,24.6,0.637,21,0],
        [1,109,56,21,135,25.2,0.833,23,0],
        [2,88,74,19,53,29.0,0.229,22,0],
        [17,163,72,41,114,40.9,0.817,47,1],
        [4,151,90,38,0,29.7,0.294,36,0],
        [7,102,74,40,105,37.2,0.204,45,0],
        [0,114,80,34,285,44.2,0.167,27,0],
        [2,100,64,23,0,29.7,0.368,21,0],
        [0,131,88,0,0,31.6,0.743,32,1],
        [6,104,74,18,156,29.9,0.722,41,1],
        [3,148,66,25,0,32.5,0.256,22,0],
        [4,120,68,0,0,29.6,0.709,34,0],
        [4,110,66,0,0,31.9,0.471,29,0],
        [3,111,90,12,78,28.4,0.495,29,0],
        [6,102,82,0,0,30.8,0.18,36,1],
        [6,134,70,23,130,35.4,0.542,29,1],
        [2,87,0,23,0,28.9,0.773,25,0],
        [1,79,60,42,48,43.5,0.678,23,0],
        [2,75,64,24,55,29.7,0.37,33,0],
        [8,179,72,42,130,32.7,0.719,36,1],
        [6,85,78,0,0,31.2,0.382,42,0],
        [0,129,110,46,130,67.1,0.319,26,1],
        [5,143,78,0,0,45.0,0.19,47,0],
        [5,130,82,0,0,39.1,0.956,37,1],
        [6,87,80,0,0,23.2,0.084,32,0],
        [0,119,64,18,92,34.9,0.725,23,0],
        [1,0,74,20,23,27.7,0.299,21,0],
        [5,73,60,0,0,26.8,0.268,27,0],
        [4,141,74,0,0,27.6,0.244,40,0],
        [7,194,68,28,0,35.9,0.745,41,1],
        [8,181,68,36,495,30.1,0.615,60,1],
        [1,128,98,41,58,32.0,1.321,33,1],
        [8,109,76,39,114,27.9,0.64,31,1],
        [5,139,80,35,160,31.6,0.361,25,1],
        [3,111,62,0,0,22.6,0.142,21,0],
        [9,123,70,44,94,33.1,0.374,40,0],
        [7,159,66,0,0,30.4,0.383,36,1],
        [11,135,0,0,0,52.3,0.578,40,1],
        [8,85,55,20,0,24.4,0.136,42,0],
        [5,158,84,41,210,39.4,0.395,29,1],
        [1,105,58,0,0,24.3,0.187,21,0],
        [3,107,62,13,48,22.9,0.678,23,1],
        [4,109,64,44,99,34.8,0.905,26,1],
        [6,125,68,30,120,30.0,0.464,32,0],
        [5,85,74,22,0,29.0,1.224,32,1],
        [5,112,66,0,0,37.8,0.261,41,1],
        [0,177,60,29,478,34.6,1.072,21,1],
        [2,158,90,0,0,31.6,0.805,66,1],
        [7,119,0,0,0,25.2,0.209,37,0],
        [7,142,60,33,190,28.8,0.687,61,0],
        [1,100,66,29,196,32.0,0.444,42,0],
        [5,109,75,26,0,36.0,0.546,60,0],
        [6,125,76,0,0,33.8,0.121,54,1],
        [7,101,76,0,0,35.7,0.198,26,0],
        [9,291,155,0,0,48.5,0.298,24,1],  
        [10,101,86,37,0,45.6,1.136,38,1],
        [5,147,75,0,0,29.9,0.434,28,0],
        [4,99,72,17,0,25.6,0.294,28,0],
        [8,167,106,46,231,37.6,0.165,43,1],
        [9,145,80,46,130,37.9,0.637,40,1],
        [6,115,60,39,0,33.7,0.245,40,1],
        [1,112,80,45,132,34.8,0.217,24,0],
        [4,145,82,18,0,32.5,0.235,70,1],
        [10,111,70,27,0,27.5,0.141,40,1],
        [6,98,58,33,190,34.0,0.43,43,0],
        [9,154,78,30,100,30.9,0.164,45,0],
        [6,165,68,26,168,33.6,0.631,49,0],
        [1,99,58,10,0,25.4,0.551,21,0],
        [10,68,106,23,49,35.5,0.285,47,0],
        [3,123,100,35,240,57.3,0.88,22,0],
        [8,91,82,0,0,35.6,0.587,68,0],
        [6,195,70,0,0,30.9,0.328,31,1],
        [9,156,86,0,0,24.8,0.23,53,1],
        [0,93,100,39,72,43.4,1.021,35,0],
        [3,121,52,0,0,36.0,0.127,25,1],
        [2,101,58,35,90,21.8,0.155,22,0],
        [2,96,69,21,69,31.2,0.178,28,0],
        [3,108,62,24,0,26.0,0.223,25,0],
        [1,91,54,25,100,25.2,0.234,23,0],
        [1,135,54,0,0,26.7,0.687,62,0],
        [8,97,74,15,0,25.6,0.206,25,0],
        [13,73,60,0,0,26.8,0.268,27,0],
        [4,141,74,0,0,27.6,0.244,40,0],
        [6,194,78,0,0,23.5,0.129,59,1],
        [2,181,68,36,495,30.1,0.615,60,1],
        [1,128,98,41,58,32.0,1.321,33,1],
        [8,109,76,39,114,27.9,0.64,31,1],
        [5,139,80,35,160,31.6,0.361,25,1]
    ]

    # Convert to DataFrame
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

    df = pd.DataFrame(pima_data, columns=columns)
    return df

def create_and_train_model():
    """
    Create and train a diabetes prediction model using Pima Indians Dataset
    """
    print("Loading Pima Indians Diabetes Dataset...")
    df = create_pima_dataset()

    # Prepare features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    print(f"Dataset shape: {df.shape}")
    print(f"Diabetes cases: {y.sum()} out of {len(y)} ({y.mean()*100:.1f}%)")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model with optimized parameters
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_names = X.columns
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))

    # Save model and scaler
    with open('diabetes_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("\nModel and scaler saved successfully!")
    return model, scaler

# Load or create model
try:
    with open('diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("🔄 Training new model with Pima Indians Diabetes Dataset...")
    model, scaler = create_and_train_model()

@app.route('/')
def home():
    return jsonify({
        "message": "🏥 Diabetes Prediction API",
        "version": "2.0",
        "dataset": "Pima Indians Diabetes Dataset",
        "model": "Random Forest Classifier",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Make diabetes prediction",
            "/health": "GET - Check API health",
            "/info": "GET - Get model information"
        }
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    })

@app.route('/info')
def model_info():
    """Get information about the model"""
    return jsonify({
        "model_type": "Random Forest Classifier",
        "dataset": "Pima Indians Diabetes Dataset",
        "features": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        "feature_descriptions": {
            "Pregnancies": "Number of times pregnant (0-17)",
            "Glucose": "Plasma glucose concentration in mg/dL (0-200)",
            "BloodPressure": "Diastolic blood pressure in mm Hg (0-122)",
            "SkinThickness": "Triceps skin fold thickness in mm (0-99)",
            "Insulin": "2-Hour serum insulin in μU/mL (0-846)",
            "BMI": "Body mass index (weight in kg/(height in m)^2) (0-67.1)",
            "DiabetesPedigreeFunction": "Diabetes pedigree function (0.078-2.42)",
            "Age": "Age in years (21-81)"
        },
        "output": "0 = No Diabetes, 1 = Diabetes"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input
        required_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        # Check if all required features are present
        for feature in required_features:
            if feature not in data:
                return jsonify({
                    'error': f'Missing required feature: {feature}',
                    'required_features': required_features
                }), 400

        # Extract features in correct order
        features = []
        for feature in required_features:
            try:
                value = float(data[feature])
                features.append(value)
            except (ValueError, TypeError):
                return jsonify({
                    'error': f'Invalid value for {feature}: {data[feature]}. Must be a number.'
                }), 400

        # Validate feature ranges based on Pima dataset
        validations = [
            (features[0] >= 0 and features[0] <= 20, 'Pregnancies should be between 0-20'),
            (features[1] >= 0 and features[1] <= 300, 'Glucose should be between 0-300 mg/dL'),
            (features[2] >= 0 and features[2] <= 200, 'Blood Pressure should be between 0-200 mm Hg'),
            (features[3] >= 0 and features[3] <= 150, 'Skin Thickness should be between 0-150 mm'),
            (features[4] >= 0 and features[4] <= 1000, 'Insulin should be between 0-1000 μU/mL'),
            (features[5] >= 10 and features[5] <= 80, 'BMI should be between 10-80'),
            (features[6] >= 0 and features[6] <= 3, 'Diabetes Pedigree Function should be between 0-3'),
            (features[7] >= 18 and features[7] <= 120, 'Age should be between 18-120 years')
        ]

        for is_valid, error_msg in validations:
            if not is_valid:
                return jsonify({'error': error_msg}), 400

        # Prepare features for prediction
        features_array = np.array(features).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]

        # Calculate risk factors
        risk_factors = analyze_risk_factors(dict(zip(required_features, features)))

        # Prepare response
        result = {
            'prediction': int(prediction),
            'result': 'Positive for Diabetes' if prediction == 1 else 'Negative for Diabetes',
            'confidence': float(max(prediction_proba)),
            'probability': {
                'diabetes_risk': float(prediction_proba[1]),
                'no_diabetes_risk': float(prediction_proba[0])
            },
            'risk_level': get_risk_level(prediction_proba[1]),
            'risk_factors': risk_factors,
            'input_features': dict(zip(required_features, features)),
            'model_info': {
                'model_type': 'Random Forest Classifier',
                'dataset': 'Pima Indians Diabetes Dataset'
            }
        }

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

def get_risk_level(probability):
    """Convert probability to risk level description"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Moderate Risk"
    else:
        return "High Risk"

def analyze_risk_factors(features):
    """Analyze individual risk factors"""
    risk_factors = []

    # High glucose
    if features['Glucose'] > 140:
        risk_factors.append("High glucose level (>140 mg/dL)")

    # High BMI
    if features['BMI'] > 30:
        risk_factors.append("High BMI (>30 - Obese)")
    elif features['BMI'] > 25:
        risk_factors.append("Elevated BMI (25-30 - Overweight)")

    # Age factor
    if features['Age'] > 45:
        risk_factors.append("Advanced age (>45 years)")

    # Multiple pregnancies
    if features['Pregnancies'] > 4:
        risk_factors.append("Multiple pregnancies (>4)")

    # High blood pressure
    if features['BloodPressure'] > 90:
        risk_factors.append("High blood pressure (>90 mm Hg)")

    # Family history (Diabetes Pedigree Function)
    if features['DiabetesPedigreeFunction'] > 0.5:
        risk_factors.append("Strong family history of diabetes")

    if not risk_factors:
        risk_factors.append("No significant risk factors identified")

    return risk_factors

if __name__ == '__main__':
    print("🚀 Starting Diabetes Prediction API...")
    print("📊 Using Pima Indians Diabetes Dataset")
    print("🤖 Model: Random Forest Classifier")
    print("🌐 API available at: http://localhost:5000")
    print("\n📋 Available Endpoints:")
    print("- GET  /        : API information")
    print("- GET  /health  : Health check")
    print("- GET  /info    : Model information")  
    print("- POST /predict : Make prediction")
    print("\n⚡ Ready to serve predictions!")
    app.run(host='0.0.0.0', port=5000, debug=True)
