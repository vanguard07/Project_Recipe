from fastapi import APIRouter, HTTPException, Request, Depends
from app.schemas.recipe import RecipeCreate, Prompt
from app.models.recipe import Recipe
from app.db.database import recipe_collection
from typing import List, Optional
from openai import OpenAI
import os
import json
import requests
# Initialize OpenAI client
oai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # Replace with your actual OpenAI API key

router = APIRouter()

@router.post("/", response_model=Recipe)
async def create_recipe(recipe: RecipeCreate):
    recipe_dict = recipe.model_dump()
    result = await recipe_collection.insert_one(recipe_dict)
    created_recipe = await recipe_collection.find_one({"_id": result.inserted_id})
    return created_recipe

@router.get("/", response_model=List[Recipe])
async def get_recipes(limit: int = 10):
    recipes = await recipe_collection.find().to_list(limit)
    return recipes

@router.post("/search", response_model=dict)
async def search_recipes(prompt: Prompt, request: Request):
    try:
        # Provide MongoDB schema context to help OpenAI generate relevant queries
        schema_context = RecipeCreate.model_json_schema()
        
        # Request MongoDB query from OpenAI
        openai_response = oai_client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": "You are a MongoDB expert with great expertise in writing MongoDB queries for any given data to produce an expected output."},
                {"role": "user", "content": f"Given the following MongoDB document schema for recipes: {schema_context}, The MongoDB aggregation pipeline to produce the expected output for a given prompy. The response corresponds to just the list of stages in the aggregation pipeline and shouldn't contain the db.collection.aggregate prefix. Ensure that he returned query is a fully formed MongoDB pipeline syntax. If the user query prompt doesn't feel related to related to recipes, return empty list. The user prompt is: {prompt.prompt}"},
                ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "recipe_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string"
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        
        try:
            # Extract query from the OpenAI response
            response = json.loads(openai_response.output_text)
            print(f"AI-generated query: {response}, {type(response)}")
        except json.JSONDecodeError:
            # Handle case where response isn't valid JSON
            raise HTTPException(status_code=422, detail="Failed to parse query from AI response.")
        
        # Execute the response aggregation query in the recipe collection
        # Assuming the response is a valid MongoDB aggregation pipeline
        # You may need to adjust this part based on your actual MongoDB driver and query execution method
        # For example, using pymongo:
        pipeline = json.loads(response.get('query'))
        recipes = await recipe_collection.aggregate(pipeline).to_list(length=None)
        if not recipes:
            return {"message": "Cannot help with this, let's try something else."}

        # Format results into text output (you can customize as needed)
        result_texts = []
        for rec in recipes:
            text = f"Title: {rec.get('title')}\nIngredients: {', '.join(rec.get('ingredients', []))}\nInstructions: {', '.join(rec.get('instructions', []))}"
            result_texts.append(text)

        return {"results": "\n\n".join(result_texts)}
    
    except Exception as e:
        print(f"Error generating query: {e}")
        raise HTTPException(status_code=500, detail="Error generating query from prompt.")


@router.post("/store", response_model=Recipe)
async def store_recipe_from_url(prompt: Prompt):
    try:
        # Extract URL from the user prompt
        url = prompt.prompt.strip()
        
        # Validate URL format (basic check)
        if not url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid URL format. Please provide a valid recipe website URL.")
        # Define the schema context for OpenAI
        schema_context = RecipeCreate.model_json_schema()
        
        # Request OpenAI to extract recipe data from the webpage
        openai_response = oai_client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": "You are a recipe extraction expert. Your task is to analyze recipe webpages and extract structured recipe information according to a specific schema."},
                {"role": "user", "content": f"Extract the recipe information from this webpage and format it according to the following schema: {schema_context}. Here's the webpage link: {url}. The expected output should include the title, ingredients, instructions, cuisine, meal type, prep time, cook time, and tags. If any of these fields are not available on the webpage, return null for those fields. If you're unable to extract the recipe or if the webpage contains data related to anything other than recipe, return an empty object. DO NOT HALLUCINATE OR MAKE UP ANY DATA!"},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "recipe_extraction",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "ingredients": {"type": "array", "items": {"type": "string"}},
                            "instructions": {"type": "array", "items": {"type": "string"}},
                            "cuisine": {"type": ["string", "null"]},
                            "meal_type": {"type": ["string", "null"]},
                            "prep_time": {"type": ["string", "null"]},
                            "cook_time": {"type": ["string", "null"]},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["title", "ingredients", "instructions", "cuisine", "meal_type", "prep_time", "cook_time", "tags"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        
        try:
            # Extract recipe data from the OpenAI response
            recipe_data = json.loads(openai_response.output_text)
            print(f"AI-extracted recipe data: {recipe_data}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="Failed to parse recipe data from AI response.")
        
        # Create a RecipeCreate model instance
        recipe = RecipeCreate(
            title=recipe_data.get("title"),
            ingredients=recipe_data.get("ingredients", []),
            instructions=recipe_data.get("instructions", []),
            cuisine=recipe_data.get("cuisine"),
            meal_type=recipe_data.get("meal_type"),
            prep_time=recipe_data.get("prep_time"),
            cook_time=recipe_data.get("cook_time"),
            tags=recipe_data.get("tags", []),
            source_url=url  # Store the original URL
        )
        
        # Store the recipe in the database
        recipe_dict = recipe.model_dump()
        result = await recipe_collection.insert_one(recipe_dict)
        created_recipe = await recipe_collection.find_one({"_id": result.inserted_id})
        
        return created_recipe
        
    except Exception as e:
        print(f"Error storing recipe from URL: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing recipe from URL: {str(e)}")