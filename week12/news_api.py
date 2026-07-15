"""
학습된 뉴스 카테고리 분류 모델을 FastAPI로 서빙하는 스크립트.

실행 예시:
    uvicorn news_api:app --reload
"""

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

"""
ML 모델 불러오기
"""
MODEL_PATH = "models/news_category_pipeline.joblib"

def load_model():
    """
    저장된 모델 파일을 불러와 반환합니다.

    MODEL_PATH 경로에 모델 파일(.joblib)이 없으면 FileNotFoundError를 발생시킵니다.
    모델이 없을 경우 train_news_classifier.py를 먼저 실행해 모델을 생성해야 합니다.

    Returns:
        dict: {"vectorizer": TfidfVectorizer, "classifier": LinearSVC} 형태의 딕셔너리.
    """
    return joblib.load(MODEL_PATH)

try:
    model_bundle = load_model()
except Exception as e:
    model_bundle = None
    model_load_error = str(e)

"""
FastAPI로 API 개발
"""

app = FastAPI(
    title="뉴스 헤드라인 카테고리 분류 API",
    description="헤드라인을 입력하면 카테고리를 예측합니다.",
    version="1.0.0",
)

@app.get("/")
def root():
    """
    API 루트 엔드포인트입니다.

    API 이름, 문서 경로(/docs), 모델 로드 상태를 반환합니다.
    서버가 정상 실행 중인지 빠르게 확인할 때 사용합니다.
    """
    return {
        "message": "뉴스 헤드라인 카테고리 분류 API",
        "docs": "/docs",
        "model_loaded": model_bundle is not None,
    }

"""
예측 요청(Request)용 데이터 모델 정의
"""
class PredictRequest(BaseModel):
    headline: str = Field(
        ..., # 필수 입력값임 (ellipsis는 필수 입력임을 의미)
        min_length=3,  # 최소 길이 3자
        description="분류할 뉴스 헤드라인",  # 필드 설명
        examples=["Biden says U.S. forces would defend Taiwan if China invaded"],  # 예시 입력
   
    )

# 예측 응답(Response)용 데이터 모델 정의
class PredictResponse(BaseModel):
    headline: str
    predicted_category: str

@app.get("/health")
def health():
    """
    모델 로드 상태를 확인하는 헬스체크 엔드포인트입니다.

    모델이 정상적으로 로드되었으면 {"status": "ok"}를 반환하고,
    로드에 실패했으면 {"status": "error"}와 함께 오류 내용을 반환합니다.
    """
    if model_bundle is None:
        return {"status": "error", "detail": model_load_error}
    return {"status": "ok", "detail": "model loaded"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    뉴스 헤드라인을 받아 카테고리를 예측하는 엔드포인트입니다.

    학습 때와 동일한 순서로 처리합니다:
      1) 헤드라인 텍스트를 TF-IDF 벡터로 변환 (vectorizer.transform)
      2) 변환된 벡터를 분류기에 입력해 카테고리 예측 (classifier.predict)

    Args:
        payload (PredictRequest): 분류할 뉴스 헤드라인 문자열을 담은 요청 객체.

    Returns:
        PredictResponse: 입력 헤드라인과 예측된 카테고리를 담은 응답 객체.

    Raises:
        HTTPException 503: 모델이 로드되지 않은 경우.
        HTTPException 400: headline 값이 공백으로만 이루어진 경우.
    """
    if model_bundle is None:
        raise HTTPException(status_code=503, detail=model_load_error)

    text = payload.headline.strip()
    if not text:
        raise HTTPException(status_code=400, detail="headline 값이 비어 있습니다.")

    # 저장된 딕셔너리에서 벡터화기와 분류기를 각각 꺼냅니다.
    vectorizer = model_bundle["vectorizer"]
    clf = model_bundle["classifier"]

    # 1) 입력 텍스트를 TF-IDF 벡터로 변환 (학습 때 구축된 어휘 사전 기준)
    X = vectorizer.transform([text])

    # 2) 변환된 벡터로 카테고리 예측
    pred = clf.predict(X)[0]

    return PredictResponse(headline=text, predicted_category=str(pred))
