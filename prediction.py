import joblib
import pandas as pd

# Load model, label encoder, and training features
model = joblib.load('lightgbm_model.pkl')
label_encoder = joblib.load('label_encoder.joblib')
feature_names = joblib.load('features.joblib')  # 42 columns

# Default values for optional features
DEFAULTS = {
    'age': 0,
    'count_floors_pre_eq': 1,
    'foundation_type': 'other',
    'roof_type': 'other',
    'ground_floor_type': 'other',
    'position': 'not_attached',
    'land_surface_condition': 'other'
}

def preprocess_input_ui(input_df):
    """
    Convert sidebar inputs into a DataFrame that matches all 42 model features.
    Missing features are filled with 0.
    """
    df = input_df.copy()

    # Fill missing values from sidebar with defaults
    for col, default in DEFAULTS.items():
        if col not in df.columns or df[col].isnull().any():
            df[col] = default

    # Numerical features
    numerical_cols = ['age', 'count_floors_pre_eq']
    df_num = df[numerical_cols]

    # Categorical features
    categorical_cols = ['foundation_type', 'roof_type', 'ground_floor_type', 'position', 'land_surface_condition']

    # Initialize one-hot DataFrame with 0 for all categorical features
    cat_columns = [c for c in feature_names if any(c.startswith(cat+'_') for cat in categorical_cols)]
    df_cat = pd.DataFrame(0, index=df.index, columns=cat_columns)

    for cat_col in categorical_cols:
        value = df[cat_col][0]
        matched_cols = [c for c in df_cat.columns if c.startswith(cat_col+'_') and value in c]
        for mc in matched_cols:
            df_cat.at[0, mc] = 1

    # Combine numerical + categorical
    df_combined = pd.concat([df_num, df_cat], axis=1)

    # Final alignment with all 42 features
    final_df = pd.DataFrame(0, index=df_combined.index, columns=feature_names)
    for col in df_combined.columns:
        if col in final_df.columns:
            final_df[col] = df_combined[col]

    return final_df

def predict_damage(input_df):
    """
    Returns damage label and confidence from sidebar input
    """
    processed_df = preprocess_input_ui(input_df)
    pred = model.predict(processed_df)

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(processed_df).max()
    else:
        confidence = 1.0

    if label_encoder:
        label = label_encoder.inverse_transform(pred)[0]
    else:
        label = pred[0]

    return label, confidence
