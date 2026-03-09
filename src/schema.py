from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    contract_type: str = Field(..., description="month-to-month | one-year | two-year")
    internet_service: str = Field(..., description="dsl | fiber | none")
    paperless_billing: bool
    payment_method: str = Field(..., description="credit-card | bank-transfer | mailed-check | electronic-check")

class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
