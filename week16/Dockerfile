# FastAPI와 Streamlit 서비스가 공통으로 사용하는 실행 이미지
FROM python:3.11-slim

WORKDIR /app

COPY news_app/ .
RUN pip install --no-cache-dir -r requirements.txt


# 이미지 빌드 중 뉴스 분류 모델을 학습해 API 컨테이너에 포함합니다.
RUN python train_news_classifier.py

EXPOSE 8000
EXPOSE 8501

# 단일 컨테이너에서 FastAPI와 Streamlit을 함께 실행합니다.
CMD ["sh", "-c", "uvicorn news_api:app --host 0.0.0.0 --port 8000 & streamlit run news_streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true"]
