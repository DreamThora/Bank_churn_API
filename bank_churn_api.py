"""
Bank churn prediction API — Tier 1 production-ready version.

Adds on top of the minimal version:
    1. Logging               — every request + result is written to a log
    2. Error handling         — failures return a clean HTTP error, not HTML
    3. Model version          — response tells the client which model produced it
    4. Prediction ID          — every prediction gets a unique ID for tracing
    5. Startup event          — model is loaded when the server starts, not at import

Run with:
    uvicorn bank_churn_api:app --reload
"""

import logging
import uuid
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# import needed so joblib.load can find the feature-engineering functions
import bank_churn_pipeline  # noqa: F401


# --- Tier 1, item 1: logging setup -----------------------------------------
# Configure once at module load. Every log line will include a timestamp,
# log level, and the module name — enough to trace what happened when.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --- constants -------------------------------------------------------------
# Tier 1, item 3: bump this whenever you retrain and redeploy. Downstream
# systems (dashboards, A/B tests, incident reports) rely on this to tell
# predictions apart across versions.
MODEL_VERSION = "v1.0.0"

MODEL_PATH = "churn_model_pipeline.joblib"


# --- app + model state -----------------------------------------------------
app = FastAPI(title="Bank Churn Prediction API", version=MODEL_VERSION)

# Tier 1, item 5: declared here, but NOT loaded here. Loading happens in the
# startup event below so a broken .joblib file surfaces as a clean startup
# error we can log, not a silent crash before the app even starts.
pipeline = None


@app.on_event("startup")
def load_model():
    """Load the trained pipeline into memory once, when the server starts."""
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH} (version {MODEL_VERSION})")
    except Exception:
        # exception() logs the full stack trace — critical for debugging why
        # a deploy failed to come up
        logger.exception(f"Failed to load model from {MODEL_PATH}")
        raise


# --- request / response schemas --------------------------------------------
class CustomerRecord(BaseModel):
    CustomerId: int
    CreditScore: int
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]
    EstimatedSalary: float


# --- endpoints -------------------------------------------------------------
@app.get("/health")
def health():
    """Load balancers and orchestrators poll this to know the service is alive."""
    # If pipeline is None, the model failed to load — report unhealthy so the
    # orchestrator can stop sending traffic to this instance.
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(customer: CustomerRecord):
    # Tier 1, item 4: generate a unique ID for THIS prediction. Log it and
    # return it — if the customer later complains "why was I flagged", the
    # ops team can grep the logs for this ID and see exactly what happened.
    prediction_id = str(uuid.uuid4())

    logger.info(
        f"predict request "
        f"prediction_id={prediction_id} "
        f"customer_id={customer.CustomerId}"
    )

    # Tier 1, item 2: wrap the actual prediction in try/except. Any failure
    # (bad data, memory error, corrupted model) becomes a clean HTTP 500 with
    # a message the client can understand — not a stack trace leaking out.
    try:
        feature = customer.dict()
        # placeholder values for the columns drop_useless_feature expects to
        # exist but whose values are never read — see the pipeline's docstring
        feature["id"] = 0
        feature["Surname"] = ""

        df = pd.DataFrame([feature])
        churn_probability = float(pipeline.predict_proba(df)[0][1])

    except Exception:
        # log the full stack trace under this prediction_id so we can find it
        logger.exception(
            f"predict failed "
            f"prediction_id={prediction_id} "
            f"customer_id={customer.CustomerId}"
        )
        # never leak the internal exception message to the client
        raise HTTPException(status_code=500, detail="Prediction failed")

    logger.info(
        f"predict result "
        f"prediction_id={prediction_id} "
        f"customer_id={customer.CustomerId} "
        f"probability={churn_probability:.4f}"
    )

    # Tier 1, item 3+4: include prediction_id and model_version in EVERY
    # response. Costs almost nothing, saves hours of debugging later.
    return {
        "prediction_id": prediction_id,
        "customer_id": customer.CustomerId,
        "churn_probability": churn_probability,
        "model_version": MODEL_VERSION,
    }
