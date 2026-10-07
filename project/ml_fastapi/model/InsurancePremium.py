import pickle
from config.config import MODEL_PATH
from pathlib import Path
from schemas.InsurancePremiumUserInput import UserInputInsurancePremium
from schemas.InsurancePremiumPrediction import PredictionResult
import pandas as pd


class InsurancePremium:

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = None
        self.model_path = model_path

    def load_model(self):
        """this method is used to load ml model"""
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

    def prediction(self, data: UserInputInsurancePremium) -> PredictionResult:
        """this function is use to get input from user and predict output base on input"""

        # load model
        if self.model is None:
            self.load_model()

        # predict output and creat DataFrame
        input_df = pd.DataFrame(
            [
                {
                    "bmi": data.bmi,
                    "age_group": data.age_group,
                    "lifestyle_risk": data.lifestyle_risk,
                    "city_tier": data.city_tier,
                    "income_lpa": data.income_lpa,
                    "occupation": data.occupation,
                }
            ]
        )

        # get all class lable for the model
        class_labels = self.model.classes_.tolist()

        # find prediction class
        prediction = self.model.predict(input_df).tolist()[0]

        # probabilities calculate
        probabilities = self.model.predict_proba(input_df)[0]
        confidence = max(probabilities)
        # print(prediction)
        # print(probabilities)

        ## combine the all genrated output
        class_probs = dict(zip(class_labels, map(lambda p: round(p, 4), probabilities)))

        # return prediction output
        return PredictionResult(
            predicted_category=prediction,
            confidence=round(confidence, 4),
            class_probabilities=class_probs,
        )
