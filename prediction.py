import joblib
import pandas as pd

# Load your pre-trained model and label encoder
model = joblib.load('lightgbm_model.pkl')  # make sure this matches your GitHub file
label_encoder = joblib.load('label_encoder.joblib')  # keep if your model uses it

def preprocess_input(input_df):
    """
    Simplest preprocessing:
    - Only keep the columns collected from Streamlit sidebar
    - Encode categorical variables with one-hot encoding
    """
    df = input_df.copy()

    # List of categorical columns from your sidebar
    categorical_cols = [
        'foundation_type',
        'roof_type',
        'ground_floor_type',
        'position',
        'land_surface_condition'
    ]

    # Convert categorical columns to string (safety)
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    # One-hot encode categorical columns
    df_categorical = pd.get_dummies(df[categorical_cols], drop_first=True)

    # Keep numerical columns
    numerical_cols = ['age', 'count_floors_pre_eq']
    df_numerical = df[numerical_cols].copy()

    # Combine numerical and categorical
    df_processed = pd.concat([df_numerical, df_categorical], axis=1)

    return df_processed

def predict_damage(input_df):
    # Preprocess input
    processed_df = preprocess_input(input_df)

    # Predict
    pred = model.predict(processed_df)
    
    # Predict probability (confidence)
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(processed_df).max()
    else:
        confidence = 1.0  # fallback if model has no predict_proba

    # Decode label if label_encoder exists
    if label_encoder:
        label = label_encoder.inverse_transform(pred)[0]
    else:
        label = pred[0]

    return label, confidence

