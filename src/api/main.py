from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '../../')
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

app = FastAPI(title="Keiiba-AI API", version="1.0.0")

# CORS setup for Frontend (localhost:3000)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Keiiba-AI API v1"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
