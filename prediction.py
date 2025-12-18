import joblib
import pandas as pd

# Load model, label encoder, and feature names
model = joblib.load('lightgbm_model.pkl')
label_encoder = joblib.load('label_encoder.joblib')
feature_names = joblib.load('features.joblib')  # columns used during training

def preprocess_input(input_df):
    df = input_df.copy()

    # --- Step 1: numerical columns ---
    numerical_cols = ['age', 'count_floors_pre_eq']
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0
    df_num = df[numerical_cols].copy()

    # --- Step 2: categorical columns ---
    categorical_cols = [
        'foundation_type',
        'roof_type',
        'ground_floor_type',
        'position',
        'land_surface_condition'
    ]
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = 'other'
        df[col] = df[col].astype(str)
    df_cat = pd.get_dummies(df[categorical_cols])

    # --- Step 3: Combine numerical + categorical ---
    processed_df = pd.concat([df_num, df_cat], axis=1)

    # --- Step 4: Align with model features ---
    final_df = pd.DataFrame(0, index=processed_df.index, columns=feature_names)
    for col in processed_df.columns:
        if col in final_df.columns:
            final_df[col] = processed_df[col]

    return final_df

def predict_damage(input_df):
    processed_df = preprocess_input(input_df)

    # --- Predict ---
    pred = model.predict(processed_df)

    # --- Predict confidence ---
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(processed_df).max()
    else:
        confidence = 1.0

    # --- Decode label ---
    if label_encoder:
        label = label_encoder.inverse_transform(pred)[0]
    else:
        label = pred[0]

    return label, confidence
