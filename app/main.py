from fastapi import FastAPI

app = FastAPI(
    title="AI Traffic Prediction System",
    description="Machine Learning based traffic prediction API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Traffic Prediction System API is running",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }