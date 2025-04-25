from fastapi import APIRouter, HTTPException, Request, Depends, Body
from app.schemas.recipe import RecipeCreate, Prompt
from app.models.recipe import Recipe
from app.db.database import recipe_collection, chat_history_collection
from typing import List, Optional
from openai import OpenAI
import os
import json
from bson.objectid import ObjectId
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
async def search_recipes(request: Request):
    try:
        request_body = await request.json()

        prompt = request_body.get("prompt", "")
        chat_id = request_body.get("chat_id", None)

        # Provide MongoDB schema context to help OpenAI generate relevant queries
        schema_context = RecipeCreate.model_json_schema()
        
        # Request MongoDB query from OpenAI
        openai_response = oai_client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": f"""Given the following MongoDB schema context for recipes: {schema_context}.
                Generate a dictionary containing key value pairs of the components which can be used in MongoDB search query to search recipes in the collection based on the user's natural language input query.
                Return ONLY the dictionary in valid JSON format without any formatting, explanations or comments."""
                },
                {"role": "user", "content": f"{prompt}"},
                ],
        )
        
        try:
            # Extract query from the OpenAI response
            print(f"OpenAI response: {openai_response.output_text}")
            response = json.loads(openai_response.output_text)
        except json.JSONDecodeError:
            # Handle case where response isn't valid JSON
            raise HTTPException(status_code=422, detail="Failed to parse query from AI response.")
        
        # Execute the response aggregation query in the recipe collection
        # pipeline = json.loads(response.get('query'))
        recipes = await recipe_collection.find(response).to_list(100)
        if not recipes:
            return {"message": "Cannot help with this, let's try something else."}

        # Format results into text output (you can customize as needed)
        result_texts = []
        for rec in recipes:
            text = f"Title: {rec.get('title')}\nIngredients: {', '.join(rec.get('ingredients', []))}\nInstructions: {', '.join(rec.get('instructions', []))}"
            result_texts.append(text)
        
        # If chat_id is provided, store the conversation history else create a new one
        if chat_id:
            chat_history = await chat_history_collection.find_one({"_id": ObjectId(chat_id)})
            print(f"Chat history: {chat_history}")
            if chat_history:
                # Append the new message to the existing conversation history
                chat_history["messages"].append({"role": "user", "content": prompt})
                chat_history["messages"].append({"role": "assistant", "content": json.dumps(result_texts)})
                await chat_history_collection.update_one({"_id": ObjectId(chat_id)}, {"$set": chat_history})
        else:
            # Create a new conversation history
            new_chat_history = {
                "messages": [{"role": "user", "content": f"{prompt}"},
                {"role": "assistant", "content": json.dumps(result_texts)}]
            }
            chat_history_doc = await chat_history_collection.insert_one(new_chat_history)
            chat_id = str(chat_history_doc.inserted_id)
            print(f"New chat history created with ID: {chat_id}")

        return {"results": result_texts, "chat_id": chat_id}
    
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
            model="gpt-4.1-nano",
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
            title=recipe_data.get("title").lower(),
            ingredients=recipe_data.get("ingredients", []),
            instructions=recipe_data.get("instructions", []),
            cuisine=recipe_data.get("cuisine").lower() if recipe_data.get("cuisine") else None,
            meal_type=recipe_data.get("meal_type").lower() if recipe_data.get("meal_type") else None,
            prep_time=recipe_data.get("prep_time"). lower() if recipe_data.get("prep_time") else None,
            cook_time=recipe_data.get("cook_time").lower() if recipe_data.get("cook_time") else None,
            tags=[tag.lower() for tag in recipe_data.get("tags", [])],
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

# Add a new schema for storing conversation history
@router.post("/customize", response_model=dict)
async def customize_recipe(request: Request):
    try:
        request_body = await request.json()
        prompt = request_body.get("prompt", "")
        chat_id = request_body.get("chat_id", None)
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # If chat_id is provided, retrieve the conversation history
        if not chat_id:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_history = await chat_history_collection.find_one({"_id": ObjectId(chat_id)})

        if not chat_history:
            raise HTTPException(status_code=404, detail="Chat history not found")

        # Extract the conversation history
        conversation_history = chat_history.get("messages", [])

        # Prepare the messages for OpenAI
        messages = [{"role": "system", "content": "You are a culinary expert that helps with recipe customization. You can suggest ingredient substitutions, adjust proportions based on servings, provide variations for dietary restrictions, and give cooking tips. Your responses should be practical, specific, and maintain the original recipe's flavor profile when possible."}]
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add the current prompt
        messages.append({"role": "user", "content": prompt})
        print(f"Messages for OpenAI: {messages}")
        # Request customization from OpenAI
        openai_response = oai_client.responses.create(
            model="gpt-4.1-nano",
            input=messages
        )
        
        try:
            # Parse the OpenAI response
            customization_data = openai_response.output_text
            print(f"AI-generated customization data: {customization_data}")

            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": customization_data})

            # update the conversation history
            await chat_history_collection.update_one({"_id": ObjectId(chat_id)}, {"$set": {"messages": conversation_history}})

            return {
                "result": customization_data,
                "chat_id": chat_id,
            }
            
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="Failed to parse customization from AI response.")
    
    except HTTPException as http_ex:
        # Handle HTTP exceptions separately
        raise http_ex
    except Exception as e:
        print(f"Error customizing recipe: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing customization request: {str(e)}")

@router.post("/classify", response_model=dict)
async def classify_prompt(request: Request):
    try:
        request_body = await request.json()
        prompt = request_body.get("prompt", "")
        chat_id = request_body.get("chat_id", None)
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Provide MongoDB schema context to help OpenAI generate relevant queries
        schema_context = RecipeCreate.model_json_schema()

        system_prompt = f"""You are an expert at classifying user queries about recipes. Your task is to determine whether a user's input is:

        1. SEARCH: User wants to find or discover recipes matching certain criteria (ingredients, cuisine, meal type, etc.)
        2. CUSTOMIZE: User wants to modify, adapt, or get advice about existing recipes (substitutions, portion changes, etc.)
        3. OTHER: Query is unrelated to recipes or cannot be categorized as search or customize

        Classification guidelines:

        SEARCH indicators:
        - Asking for recipes with specific ingredients ("pasta recipes with tomatoes")
        - Requests for dishes from specific cuisines ("Thai dishes")
        - Meal-type queries ("breakfast ideas", "dinner recipes")
        - Diet-specific recipe requests ("keto recipes", "vegan meals")
        - Using terms like "find", "show me", "what are", "recipes for", "how to make"

        CUSTOMIZE indicators:
        - Mentions of substitutions ("What can I use instead of butter?")
        - Portion adjustments ("How do I adjust this for 8 people?")
        - Dietary adaptations ("Make this recipe gluten-free")
        - Cooking technique modifications ("Can I bake instead of fry?")
        - Using terms like "change", "replace", "substitute", "adjust", "adapt", "modify", "convert"

        If you feel that questions about a specific recipe that's being discussed or follow-up questions about recipe details can be found in the mongo collection based on the schema {schema_context}, you can classify it as SEARCH.

        Return ONLY the classification type with no additional explanation."""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history for context
         # If chat_id is provided, retrieve the conversation history
        if chat_id:
            chat_history = await chat_history_collection.find_one({"_id": ObjectId(chat_id)})
            if chat_history:
                # Only include the last 5 messages for relevant context
                conversation_history = chat_history.get("messages", [])[-10:] if chat_history.get("messages") else []
                messages.extend(conversation_history)
        
        # Add the current prompt
        messages.append({"role": "user", "content": f"Classify this query: '{prompt}'"})
        
        openai_response = oai_client.responses.create(
            model="gpt-4.1-mini",
            input=messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "query_classification",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["search", "customize"],
                                "description": "The classification of the user query"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score for the classification (0-1)"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of why this classification was chosen"
                            }
                        },
                        "required": ["type", "confidence", "reasoning"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        
        classification = json.loads(openai_response.output_text)
        print(f"AI classification: {classification}")
        
        # You could add a confidence threshold here if needed
        if classification.get("confidence", 0) < 0.6:
            print(f"Low confidence classification: {classification.get('confidence')}")
            
        # Check the classification result
        print(f"Classification result: {classification.get('type')}")
        if classification.get("type") == "customize":
            return await customize_recipe(request)
        elif classification.get("type") == "search":
            return await search_recipes(request)
        else:
            return {
                "message": "I'm not sure how to help with that query. Could you try asking about a recipe search or customization?",
                "chat_id": chat_id
            }

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(f"Error classifying prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")