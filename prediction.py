import joblib
import pandas as pd

# Load model, label encoder, and training feature names
model = joblib.load('lightgbm_model.pkl')
label_encoder = joblib.load('label_encoder.joblib')
feature_names = joblib.load('features.joblib')  # all 42 columns

def preprocess_input(input_df):
    """
    Prepares input for LightGBM:
    - Ensures all 42 columns exist
    - One-hot encodes categorical variables to match training features
    """
    df = input_df.copy()

    # --- Numerical columns ---
    numerical_cols = ['age', 'count_floors_pre_eq']
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0
    df_num = df[numerical_cols]

    # --- Categorical columns ---
    categorical_cols = [
        'foundation_type',
        'roof_type',
        'ground_floor_type',
        'position',
        'land_surface_condition'
    ]

    # Initialize categorical DataFrame with 0s for all one-hot columns
    df_cat = pd.DataFrame(0, index=df.index, columns=[c for c in feature_names if any(c.startswith(cat+'_') for cat in categorical_cols)])

    for cat_col in categorical_cols:
        value = df[cat_col][0]
        # Find matching one-hot column
        matched_cols = [c for c in df_cat.columns if c.startswith(cat_col+'_') and value in c]
        for mc in matched_cols:
            df_cat.at[0, mc] = 1

    # --- Combine numerical + categorical ---
    processed_df = pd.concat([df_num, df_cat], axis=1)

    # --- Final alignment with model features ---
    final_df = pd.DataFrame(0, index=processed_df.index, columns=feature_names)
    for col in processed_df.columns:
        if col in final_df.columns:
            final_df[col] = processed_df[col]

    return final_df

def predict_damage(input_df):
    """
    Returns the predicted damage label and confidence
    """
    processed_df = preprocess_input(input_df)

    # Predict label
    pred = model.predict(processed_df)

    # Predict confidence
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(processed_df).max()
    else:
        confidence = 1.0

    # Decode label
    if label_encoder:
        label = label_encoder.inverse_transform(pred)[0]
    else:
        label = pred[0]

    return label, confidence
