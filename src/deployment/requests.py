from typing import Optional

from pydantic import BaseModel


class InferenceRequest(BaseModel):

    Pclass: int
    Name: str
    Sex: str
    Age: Optional[float]  # Age can be missing
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: Optional[str]  # Cabin can be missing
    Embarked: str
