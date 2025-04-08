from fastapi import APIRouter, HTTPException
from app.schemas.recipe import RecipeCreate
from app.models.recipe import Recipe
from app.db.database import recipe_collection
from typing import List

router = APIRouter()

@router.post("/", response_model=Recipe)
async def create_recipe(recipe: RecipeCreate):
    recipe_dict = recipe.dict()
    result = await recipe_collection.insert_one(recipe_dict)
    created_recipe = await recipe_collection.find_one({"_id": result.inserted_id})
    return created_recipe

@router.get("/", response_model=List[Recipe])
async def get_recipes(limit: int = 10):
    recipes = await recipe_collection.find().to_list(limit)
    return recipes