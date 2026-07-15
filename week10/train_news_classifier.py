"""
뉴스 헤드라인 카테고리 분류 모델 학습 스크립트.

실행 예시:
    python train_news_classifier.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# 현재 파일 기준으로 경로를 계산하므로, 어느 위치에서 실행해도 동작합니다.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent.parent.parent / "ML1/04.Classification/dataset/news.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "news_category_pipeline.joblib"


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"headline", "category"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"데이터 컬럼이 올바르지 않습니다. 필요한 컬럼: {required_columns}, 실제 컬럼: {set(df.columns)}"
        )

    # 결측/중복 제거: 실습 환경에서 가장 단순하고 안정적인 정제
    df = df.dropna(subset=["headline", "category"]).copy()
    df["headline"] = df["headline"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[(df["headline"] != "") & (df["category"] != "")]
    df = df.drop_duplicates(subset=["headline", "category"])
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),  # unigram + bigram
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LinearSVC(random_state=42)),
        ]
    )


def main() -> None:
    print("1) 데이터 로드 및 정제")
    df = load_and_clean_data(DATA_PATH)
    print(f"   - 샘플 수: {len(df):,}")
    print(f"   - 카테고리 수: {df['category'].nunique()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["headline"],
        df["category"],
        test_size=0.2,
        random_state=42,
        stratify=df["category"],
    )

    print("2) 모델 학습")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("3) 성능 평가")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   - 테스트 정확도(Accuracy): {acc:.4f}")
    print("\n[카테고리별 성능 리포트]")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("4) 모델 저장")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"   - 저장 완료: {MODEL_PATH}")

    print("\n완료: FastAPI에서 이 모델 파일을 로드해 예측 API를 실행할 수 있습니다.")


if __name__ == "__main__":
    main()
