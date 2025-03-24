from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from utils import generate_company_report
from pydantic import BaseModel
import os

app = FastAPI(
    title="News Summarization & Sentiment API",
    description="API for extracting news, analyzing sentiment, and generating Hindi TTS",
    version="1.0"
)

# Allow frontend to access the API (CORS config)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request Body Model (optional for POST)
# -------------------------
class CompanyRequest(BaseModel):
    company_name: str
    max_articles: int = 10

# -------------------------
# GET Endpoint – Generate Report
# -------------------------
@app.get("/generate_report")
def get_company_report(company_name: str = Query(..., description="Company name to search")):
    """
    Generate news summary, sentiment analysis, and Hindi TTS report.
    """
    try:
        report = generate_company_report(company_name)
        return {"status": "success", "data": report}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------
# Health Check Endpoint
# -------------------------
@app.get("/")
def root():
    return {"message": "API is running. Use /generate_report?company_name=Apple"}
