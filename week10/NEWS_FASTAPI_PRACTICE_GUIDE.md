# 뉴스 헤드라인 분류 + FastAPI 실습 가이드

- 데이터: `news.csv`
- 목표: 헤드라인 입력 -> 카테고리 예측
- 구성 파일:
  - `train_news_classifier.py` (모델 학습/저장)
  - `news_api.py` (FastAPI 예측 API)
  - `requirements.txt` (필요 라이브러리)

---

## 1) 가상환경(venv) 생성 및 활성화 (권장)

---

## 2) 라이브러리 설치

```bash
pip install -r requirements.txt
```

설치되는 주요 패키지:

- `scikit-learn`: 머신러닝 모델 학습
- `pandas`: CSV 데이터 로드/정제
- `joblib`: 학습된 모델 저장/불러오기
- `fastapi`, `uvicorn`: API 서버 실행

---

## 3) 모델 학습 및 저장

```bash
python train_news_classifier.py
```

실행 시 하는 일:

1. `news.csv` 로드
2. `headline`, `category` 컬럼 정제(결측/중복 제거)
3. `TF-IDF + LinearSVC` 분류 모델 학습
4. 테스트 정확도 및 리포트 출력
5. 모델 파일 저장

저장 결과:

- `models/news_category_pipeline.joblib`

> 참고: `TF-IDF + LinearSVC`는 뉴스 헤드라인 같은 텍스트 분류에서 성능이 잘 나오는 대표적인 기본 조합입니다.

---

## 4) FastAPI 서버 실행

```bash
uvicorn news_api:app --reload
```

실행 후 브라우저에서 접속:

- Swagger 문서: `http://127.0.0.1:8000/docs`
- 기본 엔드포인트: `http://127.0.0.1:8000/`
- 헬스체크: `http://127.0.0.1:8000/health`

특정 포트 사용시
```bash
uvicorn news_api:app --reload --port 8001
```

---

## 5) Swagger에서 예측 테스트

1. `http://127.0.0.1:8000/docs` 접속
2. `POST /predict` 펼치기
3. `Try it out` 클릭
4. 예시 JSON 입력:

```json
{
  "headline": "World leaders meet at the UN to discuss the ongoing war"
}
```

5. `Execute` 클릭
6. 응답 예시:

```json
{
  "headline": "World leaders meet at the UN to discuss the ongoing war",
  "predicted_category": "WORLD NEWS"
}
```