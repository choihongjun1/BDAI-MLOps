"""
학습된 뉴스 카테고리 분류 모델을 FastAPI로 서빙하는 스크립트.

실행 예시:
    uvicorn news_api:app --reload
"""

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models/news_category_pipeline.joblib"

app = FastAPI(
    title="뉴스 헤드라인 카테고리 분류 API",
    description="헤드라인을 입력하면 카테고리를 예측합니다.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    headline: str = Field(
        ...,
        min_length=3,
        description="분류할 뉴스 헤드라인",
        examples=["Biden says U.S. forces would defend Taiwan if China invaded"],
    )


class PredictResponse(BaseModel):
    headline: str
    predicted_category: str


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {MODEL_PATH}\n"
            "먼저 train_news_classifier.py를 실행해 모델을 생성하세요."
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except (FileNotFoundError, OSError, ValueError) as e:
    model = None
    model_load_error = str(e)
else:
    model_load_error = None


@app.get("/")
def root():
    return {
        "message": "뉴스 헤드라인 카테고리 분류 API",
        "docs": "/docs",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    if model is None:
        return {"status": "error", "detail": model_load_error}
    return {"status": "ok", "detail": "model loaded"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail=model_load_error)

    text = payload.headline.strip()
    if not text:
        raise HTTPException(status_code=400, detail="headline 값이 비어 있습니다.")

    pred = model.predict([text])[0]
    print(model.predict([text]))
    return PredictResponse(headline=text, predicted_category=str(pred))
