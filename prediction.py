import joblib
import pandas as pd

# --- Load essential files ---
model = joblib.load('lightgbm_model.pkl')       # Your trained LightGBM model
label_encoder = joblib.load('label_encoder.joblib')  # Label encoder
feature_names = joblib.load('features.joblib')  # List of all features used in training

def preprocess_input(input_df):
    """
    Preprocess the sidebar input for prediction:
    - Handles numerical columns
    - Handles categorical columns with one-hot encoding
    - Aligns columns with model's training features
    """
    df = input_df.copy()

    # --- Step 1: Identify columns ---
    numerical_cols = ['age', 'count_floors_pre_eq']
    categorical_cols = [
        'foundation_type',
        'roof_type',
        'ground_floor_type',
        'position',
        'land_surface_condition'
    ]

    # Ensure numerical columns exist
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0

    # Ensure categorical columns exist
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = 'other'

    # --- Step 2: One-hot encode categorical columns ---
    df_cat = pd.get_dummies(df[categorical_cols], drop_first=True)

    # --- Step 3: Extract numerical columns ---
    df_num = df[numerical_cols].copy()

    # --- Step 4: Combine numerical + categorical ---
    processed_df = pd.concat([df_num, df_cat], axis=1)

    # --- Step 5: Align with model features ---
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

    # --- Step 1: Predict ---
    pred = model.predict(processed_df)

    # --- Step 2: Predict confidence ---
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(processed_df).max()
    else:
        confidence = 1.0

    # --- Step 3: Decode label ---
    if label_encoder:
        label = label_encoder.inverse_transform(pred)[0]
    else:
        label = pred[0]

    return label, confidence
