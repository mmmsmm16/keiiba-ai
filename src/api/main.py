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

# Include Routers
from src.api.routers import races, predictions, simulation
app.include_router(races.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(simulation.router, prefix="/api/simulation")

# CORS setup for Frontend (localhost:3000)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins (dev mode)
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
