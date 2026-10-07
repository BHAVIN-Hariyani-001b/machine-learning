from pydantic import BaseModel, Field, computed_field,field_validator
from typing import Annotated, Literal
from config.config import TIER_1_CITIES,TIER_2_CITIES


# difine the schemas for user input validate

class UserInputInsurancePremium(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120, description="Age Of The User")]
    weight: Annotated[float, Field(..., gt=0.0, description="Weight Of The User")]
    height: Annotated[
        float, Field(..., gt=0.0, lt=2.5, description="Height Of The User")
    ]
    income_lpa: Annotated[
        float, Field(..., gt=0.0, description="Annual Salary Of The User")
    ]
    smoker: Annotated[bool, Field(..., description="Is User A Smoker")]
    city: Annotated[str, Field(..., description="The City That The User Belongs To")]
    occupation: Annotated[
        Literal[
            "retired",
            "freelancer",
            "student",
            "government_job",
            "business_owner",
            "unemployed",
            "private_job",
        ],
        Field(..., description="Occupation Of The User "),
    ]

    @field_validator('city')
    @classmethod
    def normalize_city(cls,v : str) -> str:
        return v.strip().title()

    @computed_field
    @property
    def bmi(self) -> float:
        return self.weight / (self.height ** 2)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return 'high'
        elif self.smoker or self.bmi > 27:
            return 'medium'
        else:
            return 'low'

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return 'young'
        elif self.age < 45:
            return 'adult'
        elif self.age < 60:
            return 'middle_aged'
        else:
            return 'senior'

    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in TIER_1_CITIES:
            return 1
        elif self.city in TIER_2_CITIES:
            return 2
        else:
            return 3

    
