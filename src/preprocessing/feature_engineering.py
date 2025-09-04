
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config import PROCESSED_DATA_PATH

logger = get_logger()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yeni özellikler ekler:
    - EnergyLevel
    - MoodCategory
    - TempoCategory
    - Danceability
    """
    # Energy kategorisi
    df["EnergyLevel"] = pd.cut(df["Energy"], bins=[0, 0.33, 0.66, 1],
                               labels=["Low", "Medium", "High"])

    # Mood kategorisi
    df["MoodCategory"] = pd.cut(df["MoodScore"], bins=[0, 0.33, 0.66, 1],
                                labels=["Sad", "Neutral", "Happy"])

    # Tempo kategorisi
    df["TempoCategory"] = pd.cut(df["BeatsPerMinute"], bins=[0, 90, 130, 300],
                                 labels=["Slow", "Medium", "Fast"])

    # Danceability skoru
    df["Danceability"] = df["RhythmScore"] * df["Energy"] * (df["BeatsPerMinute"] / 200)

    return df
from sklearn.preprocessing import LabelEncoder

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kategorik değişkenleri encode eder.
    - EnergyLevel, TempoCategory → Label Encoding
    - MoodCategory → One-Hot Encoding
    """
    # Label Encoding
    label_cols = ["EnergyLevel", "TempoCategory"]
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # One-Hot Encoding (MoodCategory)
    df = pd.get_dummies(df, columns=["MoodCategory"], prefix="Mood")
    
    return df


def process_features(input_file: str = os.path.join(PROCESSED_DATA_PATH, "music_processed.csv"),
                     save: bool = True) -> pd.DataFrame:
    logger.info(f"İşlenmiş veri yükleniyor: {input_file}")
    df = pd.read_csv(input_file)

    # Yeni özellikler ekle
    df = add_features(df)

    # Encoding işlemi
    df = encode_features(df)

    logger.info("Yeni özellikler ve encoding tamamlandı ✅")

    if save:
        feature_file = os.path.join(PROCESSED_DATA_PATH, "music_features.csv")
        df.to_csv(feature_file, index=False)
        logger.info(f"Feature engineered + encoded veri kaydedildi: {feature_file}")

    return df


process_features()