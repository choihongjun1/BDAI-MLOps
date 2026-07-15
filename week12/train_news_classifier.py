"""
뉴스 헤드라인 카테고리 분류 모델 학습 스크립트.
Pipeline 없이 TfidfVectorizer와 LinearSVC를 단계별로 직접 사용합니다.

실행 예시:
    python train_news_classifier.py
"""

import joblib, os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


# 현재 파일 기준으로 경로를 계산하므로, 어느 위치에서 실행해도 동작합니다.
DATA_PATH = "news.csv"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/news_category_pipeline.joblib"


def load_and_clean_data(csv_path) -> pd.DataFrame:
    """
    CSV 파일을 읽어 정제된 DataFrame을 반환합니다.

    - headline, category 컬럼이 없으면 ValueError를 발생시킵니다.
    - 결측값, 빈 문자열, 중복 행을 순서대로 제거합니다.
    """
    df = pd.read_csv(csv_path)

    # headline 또는 category 가 비어 있는(NaN) 행 제거
    df = df.dropna(subset=["headline", "category"]).copy()

    # 문자열로 변환 후 앞뒤 공백 제거
    df["headline"] = df["headline"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    # 공백만 남은 빈 문자열 행 제거
    df = df[(df["headline"] != "") & (df["category"] != "")]

    # 완전히 동일한 (헤드라인, 카테고리) 쌍 중복 제거
    df = df.drop_duplicates(subset=["headline", "category"])

    return df


def main() -> None:

    # ── 1단계: 데이터 로드 및 정제 ──────────────────────────────────────
    print("1) 데이터 로드 및 정제")
    df = load_and_clean_data(DATA_PATH)
    print(f"   - 전체 샘플 수: {len(df):,}")
    print(f"   - 카테고리 수:  {df['category'].nunique()}")

    # ── 2단계: 학습 / 테스트 데이터 분할 ────────────────────────────────
    print("\n2) 학습/테스트 데이터 분할")

    # stratify=y: 카테고리 비율이 학습/테스트 세트에서 균등하게 유지되도록 설정
    X_train, X_test, y_train, y_test = train_test_split(
        df["headline"],
        df["category"],
        test_size=0.2,
        random_state=42,
        stratify=df["category"],
    )
    print(f"   - 학습 샘플 수: {len(X_train):,}")
    print(f"   - 테스트 샘플 수: {len(X_test):,}")

    # ── 3단계: TF-IDF 벡터화 ────────────────────────────────────────────
    # 텍스트(문자열)를 머신러닝 모델이 처리할 수 있는 숫자 행렬로 변환합니다.
    # TF-IDF = 단어 빈도(TF) × 역문서 빈도(IDF)
    #   - TF : 해당 문서에서 단어가 얼마나 자주 등장하는가
    #   - IDF: 전체 문서에서 얼마나 드물게 등장하는가 (희귀할수록 중요)
    print("\n3) TF-IDF 벡터화")

    vectorizer = TfidfVectorizer(
        lowercase=True,       # 모든 텍스트를 소문자로 통일 (Apple = apple)
        stop_words="english", # a, the, is 같은 영어 불용어 제거
        ngram_range=(1, 2),   # 단어 1개(unigram)와 2개 연속(bigram) 모두 사용
                              # 예: "stock" + "stock market" 둘 다 특성으로 활용
        min_df=2,             # 전체 문서 중 2개 미만에서만 등장하는 단어는 제외 (오타 등 노이즈 방지)
        max_df=0.95,          # 전체 문서의 95% 이상에서 등장하는 너무 흔한 단어는 제외 
        )

    # fit_transform: 학습 데이터로 어휘 사전을 구축하고, 동시에 행렬로 변환
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # transform: 테스트 데이터는 학습 데이터로 만든 어휘 사전을 그대로 사용
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"   - 어휘 사전 크기: {len(vectorizer.vocabulary_):,}개 단어")
    print(f"   - 학습 행렬 크기: {X_train_tfidf.shape}")

    # ── 4단계: 분류 모델 학습 ────────────────────────────────────────────
    # LinearSVC: 선형 서포트 벡터 분류기
    print("\n4) LinearSVC 모델 학습")
    clf = LinearSVC(random_state=42)

    # 벡터화된 학습 데이터와 정답 레이블(카테고리)로 모델 학습
    clf.fit(X_train_tfidf, y_train)
    print("   - 학습 완료")

    # ── 5단계: 성능 평가 ─────────────────────────────────────────────────
    print("\n5) 성능 평가")

    # 벡터화된 테스트 데이터로 카테고리 예측
    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"   - 테스트 정확도(Accuracy): {acc:.4f}")

    # ── 6단계: 모델 저장 ─────────────────────────────────────────────────
    # 예측 시에는 벡터화기(vectorizer)와 분류기(clf) 를 순서대로 모두 사용해야 합니다.
    # 두 객체를 딕셔너리로 묶어 파일 하나에 저장합니다.
    print("6) 모델 저장")

    os.mkdir(MODEL_DIR)

    model_bundle = {
        "vectorizer": vectorizer,  # 새 텍스트를 숫자로 변환하는 벡터화기
        "classifier": clf,         # 숫자 행렬로 카테고리를 예측하는 분류기
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"   - 저장 완료: {MODEL_PATH}")

if __name__ == "__main__":
    main()
