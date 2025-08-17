# 🌐 FastAPI Translate Service

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

FastAPI 기반 번역 서비스 API입니다.  
영어 ↔ 한국어 번역, TTS(Text-to-Speech), 그리고 비동기 작업 처리(Celery)를 지원합니다.

---

## 🚀 Features
- 🔄 **Translation API** : 영어 ↔ 한국어 자동 번역 (Model : madlad-400)
- ⚡ **Async Processing with Celery** : 대용량 요청도 안정적으로 처리
- 🐳 **Docker Compose 지원** : 로컬/서버 어디서든 손쉽게 실행 가능
- 📡 **RESTful API + Swagger UI** : 직관적인 테스트 및 문서화

---

## 📂 Project Structure
```bash
fastapi-translate-service/
├── fastapi/             # FastAPI 앱 코드
│   ├── router/          # API 라우터
│   │   ├── router.py
│   │   └── task.py
│   ├── main.py          
│   └── celeryconfig.py  # Celery 설정
├── docker-compose.yml
├── .env
└── README.md
```

## ⚙️ Installation & Usage

1️⃣ Clone Repository
```bash
git clone https://github.com/Jang-YoonSung/fastapi-translate-service.git
cd fastapi-translate-service
```
2️⃣ Setup Environment

```.env``` 파일을 생성하고 필요한 환경변수를 작성합니다:
```bash
# RabbitMQ 접속 정보
RABBITMQ_USER=your_rabbitmq_username
RABBITMQ_PASS=your_rabbitmq_password
```
3️⃣ Run with Docker Compose
```bash
docker-compose up --build
```
서비스 실행 후, 아래 주소에서 API 문서를 확인할 수 있습니다:

👉 http://localhost:8000/docs

## 🛠 Tech Stack
FastAPI - Python Web Framework

Celery - Distributed Task Worker

Redis - Message Broker, Database

Rabbit MQ - Task Queue

Docker - Containerization

## 📄 License
MIT License

✨ 만든 사람: @Jang-YoonSung
