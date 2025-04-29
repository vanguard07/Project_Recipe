from pydantic import BaseModel
from typing import List, Optional

class RecipeCreate(BaseModel):
    title: str
    ingredients: List[str]
    instructions: List[str]
    cuisine: Optional[str] = None
    meal_type: Optional[str] = None
    prep_time_in_mins: Optional[int] = None
    cook_time_in_mins: Optional[int] = None
    total_time_in_mins: Optional[int] = None
    tags: Optional[List[str]] = []
    source_url: Optional[str] = None
    estimated_calories: Optional[float] = None
    protein_grams: Optional[float] = None
    fat_grams: Optional[float] = None
    nutrients_present: Optional[List[str]] = []

class Prompt(BaseModel):
    prompt: str

class ChatHistory(BaseModel):
    id: str
    messages: List[dict]
