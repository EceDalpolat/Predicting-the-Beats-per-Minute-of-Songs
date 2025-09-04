import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import pandas as pd
import joblib
from src.utils.logger import get_logger
from src.utils.config import RAW_DATA_PATH, FEATURES_DATA_PATH, MODEL_PATH
from src.preprocessing.feature_engineering import add_features, encode_features

logger = get_logger()
# === Test Verisini Hazırla ===
def process_features(input_file: str = os.path.join(RAW_DATA_PATH, "test.csv"),
                     save: bool = True) -> pd.DataFrame:
    logger.info(f"test verisi yükleniyor: {input_file}")
    df = pd.read_csv(input_file)
    if "BeatsPerMinute" not in df.columns:
        df["BeatsPerMinute"] = 120

    # Yeni özellikler ekle
    df = add_features(df)

    # Encoding işlemi
    df = encode_features(df)

    logger.info("Yeni özellikler ve encoding tamamlandı ✅")

    if save:
        feature_file = os.path.join(FEATURES_DATA_PATH, "music_features_test.csv")
        df.to_csv(feature_file, index=False)
        logger.info(f"Feature engineered + encoded veri kaydedildi: {feature_file}")

    return df
    
    return df
def make_submission(model_name: str = "Ridge"):
    # Test verisini hazırla
    test_df = process_features()
    X_test = test_df.drop(columns=["id", "BeatsPerMinute"])
    
    # Modeli yükle
    model_file = os.path.join(MODEL_PATH, f"{model_name}.pkl")
    logger.info(f"Model yükleniyor: {model_file}")
    model = joblib.load(model_file)
    
    # Tahmin üret
    preds = model.predict(X_test)
    
    # Submission dataframe
    submission = pd.DataFrame({
        "id": test_df["id"],
        "BeatsPerMinute": preds
    })
    
    # Kaydet
    submission_file = os.path.join(FEATURES_DATA_PATH, "submission.csv")
    submission.to_csv(submission_file, index=False)
    logger.info(f"Submission kaydedildi: {submission_file}")
    print("✅ Submission hazır!")

make_submission()