from pydantic import BaseModel, Field

class Dish(BaseModel):
    idx: int
    text: str
    tokens: list[str] = []
    labels: list[str] = []

class DishTokens(BaseModel):
    """记录食材属性的数据结构，包括分词和每个部分的性质。"""

    text: str = Field(description="原始的食物名称")
    tokens: list[str] = Field(description="分词的结果，用|分隔")
    properties: list[str] = Field(description="每个部分的性质，用|分隔，和tokens一一对应")
