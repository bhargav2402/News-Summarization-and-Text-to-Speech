from fastapi import FastAPI
from utils import generate_report  # Your existing function
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow Streamlit frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:8501"] etc.
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "API is running. Use /generate_report?company_name=Apple"}

@app.get("/generate_report")
def report(company_name: str):
    return generate_report(company_name)
