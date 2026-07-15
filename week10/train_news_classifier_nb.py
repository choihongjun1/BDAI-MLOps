"""
아주 쉬운 버전: 나이브 베이즈(MultinomialNB)로 뉴스 카테고리 분류.

실행:
    python train_news_classifier_nb.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DATA_PATH = "news.csv"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/news_category_pipeline_nb.joblib"


def main() -> None:
    # 1) 데이터 로드
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["headline", "category"]).copy()
    df["headline"] = df["headline"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[(df["headline"] != "") & (df["category"] != "")]

    # 2) 학습/평가 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        df["headline"],
        df["category"],
        test_size=0.2,
        random_state=42,
        stratify=df["category"],
    )

    # 3) 파이프라인: TF-IDF + Naive Bayes
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("nb", MultinomialNB()),
        ]
    )

    # 4) 학습
    model.fit(X_train, y_train)

    # 5) 평가
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Naive Bayes 테스트 정확도: {acc:.4f}")

    # 6) 저장
    joblib.dump(model, MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")

    # 7) 간단 예측 예시
    sample = "World leaders meet to discuss global economy"
    sample_pred = model.predict([sample])[0]
    print(f"샘플 헤드라인: {sample}")
    print(f"예측 카테고리: {sample_pred}")


if __name__ == "__main__":
    main()
