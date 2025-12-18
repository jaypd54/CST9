# prediction.py
import pandas as pd
import joblib

# Load model and features (once)
MODEL_FILE = 'ui_safe_lgb_model.pkl'
FEATURES_FILE = 'ui_safe_features.joblib'

model = joblib.load(MODEL_FILE)
model_features = joblib.load(FEATURES_FILE)

def preprocess_input(user_input: dict) -> pd.DataFrame:
    """
    Converts user input dictionary to a DataFrame aligned with model features.
    Missing features are filled with 0.
    """
    df = pd.DataFrame([user_input])

    # One-hot encode categorical features manually
    df_encoded = pd.get_dummies(df)

    # Align columns with model's training features
    final_df = pd.DataFrame(0, index=[0], columns=model_features)
    for col in df_encoded.columns:
        if col in final_df.columns:
            final_df[col] = df_encoded[col]

    return final_df

def predict_damage(user_input: dict):
    """
    Predicts damage level and confidence from user input dict
    """
    processed_df = preprocess_input(user_input)
    pred = model.predict(processed_df)
    prob = model.predict_proba(processed_df).max()
    return pred[0], prob
