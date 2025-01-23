from pydantic import BaseModel
class Dish(BaseModel):
    idx: int
    text: str
    tokens: str = ''
    labels: str = ''
