from fastapi import FastAPI, status, HTTPException
import uvicorn
from schemas.InsurancePremiumUserInput import UserInputInsurancePremium
from schemas.InsurancePremiumPrediction import PredictionResponse
from model.InsurancePremium import InsurancePremium
from config.config import MODEL_VERSION

app = FastAPI()

# model version


@app.get("/")
def home():
    """this is route is home url route"""
    return {"message": "Welcome to insurance premium prediction"}


@app.get("/health")
def health():
    """this route is use to not any type problem run webserver and api"""
    return {"status": "OK", "message": "Api is run continue", "version": MODEL_VERSION}


@app.post("/predict",response_model=PredictionResponse)
def predict_premium(data: UserInputInsurancePremium):
    """this route is use to call and input provide and prediction base on user input"""
    try:
        model = InsurancePremium()

        prediction = model.prediction(data)

        return {
            "message": "prediction successfully",
            "status_code": 200,
            "prediction": prediction,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Internal Server Error", "error": str(e)},
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
