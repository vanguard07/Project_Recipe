from pydantic import BaseModel
from typing import List, Optional

class RecipeCreate(BaseModel):
    title: str
    ingredients: List[str]
    instructions: List[str]
    cuisine: Optional[str] = None
    meal_type: Optional[str] = None
    prep_time: Optional[str] = None
    cook_time: Optional[str] = None
    tags: Optional[List[str]] = []
    source_url: Optional[str] = None