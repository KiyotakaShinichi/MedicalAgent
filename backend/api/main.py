from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI(title="AI Oncology Monitoring System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "MedicalAgent API is running"}

@app.get("/patient-report")
def get_patient_report():
    with open("Data/patient_report.json", "r") as f:
        report = json.load(f)

    return report
