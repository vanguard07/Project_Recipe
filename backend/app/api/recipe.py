import os # Add os import
from pathlib import Path # Add pathlib import

from fastapi import APIRouter, HTTPException, Request
from app.schemas.recipe import RecipeCreate, Prompt, LangchainChatInput
from app.models.recipe import Recipe
from app.db.database import recipe_collection, chat_history_collection
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json
from bson.objectid import ObjectId
# Langchain/Chroma imports
from langchain.docstore.document import Document
import chromadb
from app.core.config import settings

# Langchain imports
from langchain.docstore.document import Document # Import Document for Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage # Import message types

# Initialize OpenAI client
oai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Initialize ChromaDB client and embeddings
embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
# Ensure the collection exists or create it
vector_collection_name = "recipes"
vector_store = chroma_client.get_or_create_collection(name=vector_collection_name, embedding_function=chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(api_key=settings.OPENAI_API_KEY, model_name="text-embedding-ada-002"))

# Initialize Embeddings and LLM
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini", openai_api_key=settings.OPENAI_API_KEY)

# Initialize Chroma Vector Store connection (using the same persistent path)
vector_store_client = Chroma(
    collection_name=vector_collection_name,
    embedding_function=embeddings,
    persist_directory=settings.CHROMA_PERSIST_DIR,
)
retriever = vector_store_client.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant recipes

# 1. Contextualize Question Prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 2. History-Aware Retriever Chain
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 3. Question Answering Prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks about recipes. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 4. Document Chain (Stuffing context into prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# 5. Combine Retriever and QA Chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


router = APIRouter()

# --- Define path to prompts directory relative to this file ---
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# --- Helper function for storing and vectorizing ---
async def _store_recipe_and_vectorize(recipe: RecipeCreate) -> Recipe:
    """Stores a recipe in MongoDB and adds it to the Chroma vector store."""
    # Store the recipe in the database
    recipe_dict = recipe.model_dump()
    result = await recipe_collection.insert_one(recipe_dict)
    created_recipe = await recipe_collection.find_one({"_id": result.inserted_id})

    if not created_recipe:
        # This should ideally not happen if insert_one succeeds
        raise Exception("Failed to retrieve recipe after insertion.")

    # --- Add to Chroma Vector Store ---
    try:
        # Format recipe data for Langchain Document
        page_content = f"Title: {created_recipe['title']}\n"
        if created_recipe.get('cuisine'): page_content += f"Cuisine: {created_recipe['cuisine']}\n"
        if created_recipe.get('meal_type'): page_content += f"Meal Type: {created_recipe['meal_type']}\n"
        page_content += f"Ingredients: {'; '.join(created_recipe['ingredients'])}\n"
        page_content += f"Instructions: {' '.join(created_recipe['instructions'])}\n"
        if created_recipe.get('tags'): page_content += f"Tags: {', '.join(created_recipe['tags'])}\n"
        if created_recipe.get('estimated_calories'): page_content += f"Estimated Calories: {created_recipe['estimated_calories']}\n"
        if created_recipe.get('protein_grams'): page_content += f"Protein: {created_recipe['protein_grams']}g\n"
        if created_recipe.get('fat_grams'): page_content += f"Fat: {created_recipe['fat_grams']}g\n"
        if created_recipe.get('nutrients_present'): page_content += f"Nutrients Present: {', '.join(created_recipe['nutrients_present'])}\n"
        if created_recipe.get('prep_time_in_mins'): page_content += f"Prep Time: {created_recipe['prep_time_in_mins']} mins\n"
        if created_recipe.get('cook_time_in_mins'): page_content += f"Cook Time: {created_recipe['cook_time_in_mins']} mins\n"
        if created_recipe.get('total_time_in_mins'): page_content += f"Total Time: {created_recipe['total_time_in_mins']} mins\n"

        metadata = {
            "source": created_recipe.get('source_url', 'unknown'),
            "title": created_recipe['title'],
            "mongo_id": str(created_recipe['_id']) # Store MongoDB ID for reference
        }
        if created_recipe.get('cuisine'): metadata['cuisine'] = created_recipe['cuisine']
        if created_recipe.get('meal_type'): metadata['meal_type'] = created_recipe['meal_type']
        if created_recipe.get('total_time_in_mins') is not None: metadata['total_time_in_mins'] = created_recipe['total_time_in_mins']
        if created_recipe.get('protein_grams') is not None: metadata['protein_grams'] = created_recipe['protein_grams']
        if created_recipe.get('fat_grams') is not None: metadata['fat_grams'] = created_recipe['fat_grams']

        # Create Langchain Document
        doc = Document(page_content=page_content, metadata=metadata)

        # Add document to Chroma collection
        vector_store.add(
            documents=[doc.page_content],
            metadatas=[doc.metadata],
            ids=[str(created_recipe['_id'])] # Use MongoDB ID as Chroma ID
        )
        print(f"Recipe {created_recipe['_id']} added/updated in Chroma vector store.")

    except Exception as e:
        # Log the error but don't fail the whole request
        # If vector store update fails, the recipe is still in MongoDB
        print(f"Error adding/updating recipe {created_recipe['_id']} in Chroma: {e}")
    # --- End Chroma Add ---

    return created_recipe
# --- End Helper function ---

def prettify_response_with_openai(raw_response, prompt=None, response_type="general"):
    # Prepare the system prompt
    system_prompt = (
        "You are an expert at formatting recipe chatbot responses for end users. "
        "Given the following raw data from a database or retrieval-augmented generation (RAG) system, convert it into a clear, concise, and visually organized format suitable for display in a chat UI. Use lists, bullet points, and step numbers where appropriate. Highlight key information such as recipe titles, ingredients, instructions, and nutrition. Do not add any extra commentary or explanation. Only return the formatted content based on the user's query. "
        f"The response type is: {response_type}."
    )

    if prompt:
        system_prompt += f" The user prompt was: {prompt}"

    # Convert raw_response to string if needed
    if not isinstance(raw_response, str):
        try:
            pretty_input = json.dumps(raw_response, indent=2, ensure_ascii=False)
        except Exception:
            pretty_input = str(raw_response)
    else:
        pretty_input = raw_response

    openai_response = oai_client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pretty_input}
        ],
    )
    return openai_response.output_text

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

        with open(PROMPTS_DIR / "search.txt", "r") as f:
            SEARCH_SYSTEM_PROMPT_TEMPLATE = f.read()

        # Provide MongoDB schema context to help OpenAI generate relevant queries
        schema_context = RecipeCreate.model_json_schema()

        # Format the system prompt with the schema context
        system_prompt_content = SEARCH_SYSTEM_PROMPT_TEMPLATE.format(schema_context=schema_context)

        messages = [{"role": "system", "content": system_prompt_content}]

        chat_history = await chat_history_collection.find_one({"_id": ObjectId(chat_id)})
        if chat_history:
            # Only include the last 10 messages for relevant context
            conversation_history = chat_history.get("messages", [])[-10:] if chat_history.get("messages") else []
            messages.extend(conversation_history)

        # Add the current prompt
        messages.append({"role": "user", "content": f"Prepare the filter based on the ongoing conversation: {prompt}"})

        # Request MongoDB query from OpenAI
        openai_response = oai_client.responses.create(
            model="o3-mini",
            input=messages,
        )
        
        try:
            # Extract query from the OpenAI response
            print(f"OpenAI response: {openai_response.output_text}")
            response = json.loads(openai_response.output_text)
        except json.JSONDecodeError:
            # Handle case where response isn't valid JSON
            raise HTTPException(status_code=422, detail="Failed to parse query from AI response.")

        # Execute the response aggregation query in the recipe collection
        if isinstance(response, dict) and len(response) == 0:
            return {"result": "Cannot help with this, let's try something else.", chat_id: chat_id}

        print(f"Response for MongoDB query: {response}")
        recipes = await recipe_collection.find(response).to_list(100) # Limit results
        if not recipes:
            # Convert the response to a $or query by pushing all conditions into an array
            or_query = {"$or": []}
            for key, value in response.items():
                if isinstance(value, list):
                    or_query["$or"].append({key: {"$in": value}})
                else:
                    or_query["$or"].append({key: value})
            print(f"OR query: {or_query}")
            recipes = await recipe_collection.find(or_query).to_list(100) # Limit results
        if not recipes:
            return {"result": "Cannot help with this, let's try something else."}

        # Format results into text output (you can customize as needed)
        result_texts = []
        for rec in recipes:
            text = f"Title: {rec.get('title')}\nIngredients: {', '.join(rec.get('ingredients', []))}\nInstructions: {', '.join(rec.get('instructions', []))}"
            result_texts.append(text)
        
        prettified_response = prettify_response_with_openai(recipes, prompt, response_type='search')
        
        # If chat_id is provided, store the conversation history else create a new one
        if chat_id:
            chat_history = await chat_history_collection.find_one({"_id": ObjectId(chat_id)})
            print(f"Chat history: {chat_history}")
            if chat_history:
                # Append the new message to the existing conversation history
                chat_history["messages"].append({"role": "user", "content": prompt})
                chat_history["messages"].append({"role": "assistant", "content": prettified_response})
                # Ensure type is set or updated
                await chat_history_collection.update_one({"_id": ObjectId(chat_id)}, {"$set": {"messages": chat_history["messages"], "type": "gpt"}})
        else:
            # Create a new conversation history
            new_chat_history = {
                "messages": [{"role": "user", "content": f"{prompt}"},
                {"role": "assistant", "content": prettified_response}],
                "type": "gpt"
            }
            chat_history_doc = await chat_history_collection.insert_one(new_chat_history)
            chat_id = str(chat_history_doc.inserted_id)
            print(f"New chat history created with ID: {chat_id}")

        print(prettified_response)
        return {"result": prettified_response, "chat_id": chat_id}
    
    except Exception as e:
        print(f"Error generating query: {e}")
        raise HTTPException(status_code=500, detail="Error generating query from prompt.")
    except FileNotFoundError:
        print(f"ERROR: Prompt file not found at {PROMPTS_DIR / 'search.txt'}")
        raise HTTPException(status_code=500, detail="Error generating query from prompt.")


@router.post("/store", response_model=Recipe)
async def store_recipe_from_url(request: Request):
    try:
        # Extract URL from the user prompt
        request_body = await request.json()
        url = request_body.get("url", "")
        print(f"Received URL: {url}", type(url))
        # Validate URL format (basic check)
        if not url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid URL format. Please provide a valid recipe website URL.")
        
        with open(PROMPTS_DIR / "store.txt", "r") as f:
            STORE_SYSTEM_PROMPT_TEMPLATE = f.read()

        # Define the schema context for OpenAI
        schema_context = RecipeCreate.model_json_schema()

        # Format the prompt with the URL and schema context in one step
        system_prompt_content = STORE_SYSTEM_PROMPT_TEMPLATE.format(url=url, schema_context=json.dumps(schema_context))
        
        # Request OpenAI to extract recipe data from the webpage
        openai_response = oai_client.responses.create(
            model="gpt-4.1-nano",
            input=[
                {"role": "system", "content": "You are a recipe extraction expert. Your task is to analyze recipe webpages and extract structured recipe information according to a specific schema."},
                {"role": "user", "content": system_prompt_content},
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
                            "prep_time_in_mins": {"type": ["number", "null"]},
                            "cook_time_in_mins": {"type": ["number", "null"]},
                            "total_time_in_mins": {"type": ["number", "null"]}, 
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "estimated_calories": {"type": ["number", "null"]},
                            "protein_grams": {"type": ["number", "null"]}, 
                            "fat_grams": {"type": ["number", "null"]}, 
                            "nutrients_present": {"type": "array", "items": {"type": "string"}},
                            "image_url": {"type": ["string", "null"]}
                        },
                        "required": ["title", "ingredients", "instructions", "cuisine", "meal_type", "prep_time_in_mins", "cook_time_in_mins", "total_time_in_mins", "tags", "estimated_calories", "protein_grams", "fat_grams", "nutrients_present", "image_url"],
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
            
            # Basic validation: Check if essential fields are present
            if not recipe_data.get("title") or not recipe_data.get("ingredients") or not recipe_data.get("instructions"):
                 raise HTTPException(status_code=422, detail="AI failed to extract essential recipe data (title, ingredients, instructions). The URL might not contain a valid recipe.")

        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="Failed to parse recipe data from AI response.")
        except Exception as e: # Catch potential validation errors
             raise HTTPException(status_code=422, detail=str(e))

        # Create a RecipeCreate model instance
        recipe_to_store = RecipeCreate(
            title=recipe_data.get("title", "Untitled Recipe").lower(), # Provide default title
            ingredients=recipe_data.get("ingredients", []),
            instructions=recipe_data.get("instructions", []),
            cuisine=recipe_data.get("cuisine").lower() if recipe_data.get("cuisine") else None,
            meal_type=recipe_data.get("meal_type").lower() if recipe_data.get("meal_type") else None,
            prep_time_in_mins=recipe_data.get("prep_time_in_mins"),
            cook_time_in_mins=recipe_data.get("cook_time_in_mins"),
            total_time_in_mins=recipe_data.get("total_time_in_mins"), # Added total time
            tags=[tag.lower() for tag in recipe_data.get("tags", [])],
            source_url=url,  # Store the original URL,
            estimated_calories=recipe_data.get("estimated_calories"),
            protein_grams=recipe_data.get("protein_grams"), # Added protein
            fat_grams=recipe_data.get("fat_grams"), # Added fat
            nutrients_present=[nutrient.lower() for nutrient in recipe_data.get("nutrients_present", [])],
            image_url=recipe_data.get("image_url") # Added image URL
        )

        # Use the helper function to store and vectorize
        created_recipe = await _store_recipe_and_vectorize(recipe_to_store)

        return created_recipe
        
    except HTTPException as http_ex:
        # Re-raise HTTP exceptions directly
        raise http_ex
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

        with open(PROMPTS_DIR / "customize.txt", "r") as f:
            CUSTOMIZE_SYSTEM_PROMPT_TEMPLATE = f.read()

        # Prepare the messages for OpenAI
        messages = [{"role": "system", "content": CUSTOMIZE_SYSTEM_PROMPT_TEMPLATE}]
        
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
            await chat_history_collection.update_one(
                {"_id": ObjectId(chat_id)}, 
                {"$set": {"messages": conversation_history, "type": "gpt"}}, # Ensure type is set/updated
                upsert=True # Create if it somehow doesn't exist, though unlikely here
            )

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

        with open(PROMPTS_DIR / "classify.txt", "r") as f:
            raw_prompt = f.read()

        # Format with schema context
        system_prompt = raw_prompt.format(schema_context=schema_context)

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
                                "enum": ["search", "customize", "other"],
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
                "result": "I'm not sure how to help with that query. Could you try asking about a recipe search or customization?",
                "chat_id": chat_id
            }

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(f"Error classifying prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    

@router.post("/langchain", response_model=Dict[str, Any])
async def langchain_chat(request: Request):
    """
    Handle chat requests using Langchain chains, manually managing history
    in a single MongoDB document per chat session.
    """
    request_body = await request.json()
    user_input = request_body.get("prompt", "")
    chat_id = request_body.get("chat_id", None)
    messages_for_db = [] # To store history in simple dict format for DB
    chat_history_for_chain = [] # To store history in Langchain BaseMessage format

    try:
        # Determine chat_id and load history
        if chat_id:
            if not ObjectId.is_valid(chat_id):
                 raise HTTPException(status_code=400, detail="Invalid chat ID format.")
            chat_doc = await chat_history_collection.find_one({"_id": ObjectId(chat_id)})
            if chat_doc:
                messages_for_db = chat_doc.get("messages", [])
                # Convert loaded history to Langchain message objects
                for msg in messages_for_db:
                    if msg.get("role") == "user":
                        chat_history_for_chain.append(HumanMessage(content=msg.get("content", "")))
                    elif msg.get("role") == "assistant":
                        chat_history_for_chain.append(AIMessage(content=msg.get("content", "")))
            else:
                 # If chat_id provided but not found, treat as new chat but use provided ID
                 print(f"Chat ID {chat_id} provided but not found. Starting new history with this ID.")
                 effective_chat_id = chat_id # Keep the provided ID
        else:
            # Generate a new chat ID if none provided
            effective_chat_id = str(ObjectId())
            print(f"No chat ID provided. Generated new ID: {effective_chat_id}")

        # Ensure effective_chat_id is set
        if not chat_id: # If chat_id was None initially
            chat_id = effective_chat_id

        # Invoke the RAG chain
        result = await rag_chain.ainvoke({
            "input": user_input,
            "chat_history": chat_history_for_chain
        })

        # Extract the answer
        answer = result.get("answer")
        final_answer_for_user = answer # Initialize final answer
        print(f"RAG chain answer: {answer}")

        # --- Fallback Logic ---
        if answer is None or answer.strip().lower() == "i don't know.":
            print("RAG chain couldn't answer. Attempting fallback and potential storage.")
            # Make call to OpenAI API to get a fallback answer
            fallback_response = await llm.ainvoke(user_input)
            fallback_content = fallback_response.content if hasattr(fallback_response, 'content') else None

            if fallback_content:
                print(f"Fallback raw content: {fallback_content[:100]}...")
                # Attempt to extract structured recipe data from the fallback content
                try:
                    schema_context = RecipeCreate.model_json_schema()
                    extraction_prompt = f"""Analyze the following text which might contain a recipe. Extract the recipe information and format it according to the following JSON schema: {schema_context}.
                    The expected output should include title, ingredients, instructions, cuisine, meal type, prep time, cook time, total time, tags, estimated calories per serving, protein grams per serving, fat grams per serving, and nutrients present.
                    Return times in minutes. If fields are missing, use null. If the text is not a recipe, return an empty JSON object {{}}.
                    DO NOT HALLUCINATE. Only extract data present in the text.

                    Text to analyze:
                    {fallback_content}
                    """
                    extraction_response = oai_client.responses.create(
                        model="gpt-4.1-nano",
                        input=[
                            {"role": "system", "content": "You are a recipe extraction expert."},
                            {"role": "user", "content": extraction_prompt},
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
                                        "prep_time_in_mins": {"type": ["number", "null"]},
                                        "cook_time_in_mins": {"type": ["number", "null"]},
                                        "total_time_in_mins": {"type": ["number", "null"]}, # Added total time
                                        "tags": {"type": "array", "items": {"type": "string"}},
                                        "estimated_calories": {"type": ["number", "null"]},
                                        "protein_grams": {"type": ["number", "null"]}, # Added protein
                                        "fat_grams": {"type": ["number", "null"]}, # Added fat
                                        "nutrients_present": {"type": "array", "items": {"type": "string"}}
                                    },
                                    # Updated required fields
                                    "required": ["title", "ingredients", "instructions", "cuisine", "meal_type", "prep_time_in_mins", "cook_time_in_mins", "total_time_in_mins", "tags", "estimated_calories", "protein_grams", "fat_grams", "nutrients_present"],
                                    "additionalProperties": False
                                },
                                "strict": True
                            }
                        }
                    )

                    recipe_data = json.loads(extraction_response.output_text)
                    print(f"Attempted extraction data: {recipe_data}")

                    # Validate extracted data (basic check)
                    if recipe_data and recipe_data.get("title") and recipe_data.get("ingredients") and recipe_data.get("instructions"):
                        print("Fallback content successfully parsed as a recipe.")
                        # Create RecipeCreate object
                        recipe_to_store = RecipeCreate(
                            title=recipe_data.get("title", "Untitled AI Recipe").lower(),
                            ingredients=recipe_data.get("ingredients", []),
                            instructions=recipe_data.get("instructions", []),
                            cuisine=recipe_data.get("cuisine", "").lower() if recipe_data.get("cuisine") else None,
                            meal_type=recipe_data.get("meal_type", "").lower() if recipe_data.get("meal_type") else None,
                            prep_time_in_mins=recipe_data.get("prep_time_in_mins"),
                            cook_time_in_mins=recipe_data.get("cook_time_in_mins"),
                            total_time_in_mins=recipe_data.get("total_time_in_mins"),
                            tags=[tag.lower() for tag in recipe_data.get("tags", [])],
                            source_url="ai_generated", # Mark as AI generated
                            estimated_calories=recipe_data.get("estimated_calories"),
                            protein_grams=recipe_data.get("protein_grams"),
                            fat_grams=recipe_data.get("fat_grams"),
                            nutrients_present=[n.lower() for n in recipe_data.get("nutrients_present", [])]
                        )

                        # Use the helper function to store and vectorize
                        created_recipe = await _store_recipe_and_vectorize(recipe_to_store)
                        print(f"Stored AI-generated recipe with ID: {created_recipe['_id']}")

                        # Set the user-facing answer
                        final_answer_for_user = prettify_response_with_openai((f"I couldn't find a matching recipe in the database, but I generated one using AI and have now added it for future reference.\n\n"
                                                 f"Title: {created_recipe['title']}\n"
                                                 f"Ingredients: {', '.join(created_recipe['ingredients'])}\n"
                                                f"Instructions: {', '.join(created_recipe['instructions'])}\n"
                                                 f"(You can ask me about '{created_recipe['title']}' now.)"))
                    else:
                        # Extraction failed or didn't yield a valid recipe
                        print("Fallback content could not be parsed as a recipe.")
                        final_answer_for_user = f"I couldn't find a matching recipe in the recipe collection."

                except json.JSONDecodeError:
                    print("Failed to decode JSON from recipe extraction response.")
                    final_answer_for_user = f"I couldn't find a matching recipe in the database."
                except Exception as e:
                    print(f"Error during fallback recipe extraction/storage: {e}")
                    # Fallback to just showing the AI content with a warning
                    final_answer_for_user = f"I couldn't find a matching recipe in the database."
            else:
                # Fallback response itself was empty
                print("Fallback generation failed.")
                final_answer_for_user = "Sorry, I couldn't find an answer in the database and failed to generate a fallback response."
        # --- End Fallback Logic ---

        # Append current interaction to the list for DB update using the final answer
        messages_for_db.append({"role": "user", "content": user_input})
        messages_for_db.append({"role": "assistant", "content": final_answer_for_user}) # Use final_answer_for_user

        # Update the single chat document in MongoDB
        await chat_history_collection.update_one(
            {"_id": ObjectId(chat_id)},
            {"$set": {"messages": messages_for_db, "type": "langchain"}},
            upsert=True
        )

        # Return the final answer to the user
        return {
            "answer": final_answer_for_user, # Return the potentially modified answer
            "chat_id": chat_id,
        }

    except HTTPException as http_ex:
         raise http_ex # Re-raise validation errors etc.
    except Exception as e:
        print(f"Error during Langchain chat: {e}")
        # Log error, potentially try to save partial history if desired
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")