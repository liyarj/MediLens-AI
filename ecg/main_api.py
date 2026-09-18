from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ecg_router import router as ecg_router

app = FastAPI(
    title="MediLens Diagnostic AI Backend API",
    description="Multi-modal medical AI diagnostic microservices API (ECG, X-Ray, CT, Risk Assessment)",
    version="1.0.0"
)

# Enable CORS for MediLens Web Frontend Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount ECG Prediction Router under /predict/ecg
app.include_router(ecg_router, prefix="/predict/ecg")

@app.get("/")
def root():
    return {
        "service": "MediLens Medical AI API",
        "status": "online",
        "active_endpoints": [
            "/predict/ecg"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
