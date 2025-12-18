import joblib
import pandas as pd

# Load model and label encoder
model = joblib.load('lightgbm_model.pkl')        # Your trained LightGBM model
label_encoder = joblib.load('label_encoder.joblib')  # Only if your model uses label encoding
feature_names = joblib.load('features.joblib')  # List of features used during training

def preprocess_input(input_df):
    """
    Preprocess input from Streamlit sidebar:
    - Keep numerical columns
    - One-hot encode categorical columns
    - Align with model features
    """
    df = input_df.copy()

    # Categorical columns from your sidebar
    categorical_cols = [
        'foundation_type',
        'roof_type',
        'ground_floor_type',
        'position',
        'land_surface_condition'
    ]

    # Convert to string and one-hot encode
    for col in categorical_cols:
        df[col] = df[col].astype(str)
    df_cat = pd.get_dummies(df[categorical_cols], drop_first=True)

    # Numerical columns
    numerical_cols = ['age', 'count_floors_pre_eq']
    df_num = df[numerical_cols].copy()

    # Combine numerical and categorical
    processed_df = pd.concat([df_num, df_cat], axis=1)

    # Align columns with model features
    final_df = pd.DataFrame(0, index=processed_df.index, columns=feature_names)
    for col in processed_df.columns:
        if col in final_df.columns:
            final_df[col] = processed_df[col]

    return final_df

def predict_damage(input_df):
    """
    Predict building damage level and confidence
    """
    processed_df = preprocess_input(input_df)

    # Predict
    pred = model.predict(processed_df)

    # Predict probability (confidence)
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(processed_df).max()
    else:
        confidence = 1.0

    # Decode label if label_encoder exists
    if label_encoder:
        label = label_encoder.inverse_transform(pred)[0]
    else:
        label = pred[0]

    return label, confidence
