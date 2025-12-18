import joblib
import pandas as pd

# Load model and label encoder
model = joblib.load('lightgbm_model.pkl')
label_encoder = joblib.load('label_encoder.joblib')
feature_names = joblib.load('features.joblib')  # all 42 columns your model expects

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
    df_cat = pd.get_dummies(df[categorical_cols])

    # --- Combine numerical + categorical ---
    processed_df = pd.concat([df_num, df_cat], axis=1)

    # --- Align with model features ---
    # Fill missing columns with 0, keep order same as model
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
