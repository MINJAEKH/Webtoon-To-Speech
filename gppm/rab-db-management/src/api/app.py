from fastapi import FastAPI
from src.api.routes.vectorize_gppm import router as vectorize_router

app = FastAPI(title='Test Airflow Params Mock API')

app.include_router(vectorize_router)

@app.get("/health")
def health() :
    return {'status' : 'ok'}