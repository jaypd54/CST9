import joblib
import pandas as pd

# Load model, label encoder, and feature names
model = joblib.load('lightgbm_model.pkl')
label_encoder = joblib.load('label_encoder.joblib')
feature_names = joblib.load('features.joblib')  # all 42 columns used in training

def preprocess_input(input_df):
    """
    Preprocess sidebar input:
    - Fill numerical values
    - One-hot encode categorical values
    - Fill missing features so all model columns are present
    """
    df = input_df.copy()

    # --- Numerical columns ---
    numerical_cols = ['age', 'count_floors_pre_eq']
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0
    df_num = df[numerical_cols].copy()

    # --- Categorical columns ---
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

    # --- One-hot encoding for categorical columns based on feature_names ---
    df_cat = pd.DataFrame(0, index=df.index, columns=[c for c in feature_names if any(c.startswith(cat+'_') for cat in categorical_cols)])
    for cat_col in categorical_cols:
        value = df[cat_col][0]
        matched_cols = [c for c in df_cat.columns if c.startswith(cat_col+'_') and value in c]
        for mc in matched_cols:
            df_cat.at[0, mc] = 1

    # --- Combine numerical + categorical ---
    processed_df = pd.concat([df_num, df_cat], axis=1)

    # --- Ensure all feature_names are present and in correct order ---
    final_df = pd.DataFrame(0, index=processed_df.index, columns=feature_names)
    for col in processed_df.columns:
        if col in final_df.columns:
            final_df[col] = processed_df[col]

    return final_df

def predict_damage(input_df):
    """
    Predict damage label and confidence
    """
    processed_df = preprocess_input(input_df)

    # Predict
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
