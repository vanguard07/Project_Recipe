from fastapi import APIRouter, HTTPException, Body
from app.db.database import chat_history_collection, recipe_collection, db, client
from app.core.config import settings
from bson.objectid import ObjectId
from typing import List, Dict, Any, Optional
from app.schemas.recipe import RecipeCreate # Import RecipeCreate schema
from app.models.recipe import Recipe # Import Recipe model if needed for typing
import json # Import json for parsing
from langchain.docstore.document import Document # Import Document for Chroma
from openai import OpenAI
import chromadb

# Langchain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import MongoDBChatMessageHistory
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage # Import message types

router = APIRouter()

# --- Langchain Setup ---
# Initialize Embeddings and LLM
embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini", openai_api_key=settings.OPENAI_API_KEY)

# Initialize Chroma Vector Store connection (using the same persistent path)
vector_store_client = Chroma(
    collection_name="recipes", # Use the same collection name as in recipe.py
    embedding_function=embeddings,
    persist_directory=settings.CHROMA_PERSIST_DIR,
)
retriever = vector_store_client.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant recipes

# --- End Langchain Setup ---

# --- Langchain Chain Setup (No RunnableWithMessageHistory) ---

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

# --- End Langchain Chain Setup ---

# Initialize OpenAI client
oai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Initialize ChromaDB client and embeddings
chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
# Ensure the collection exists or create it
# Using a simple name, adjust metadata/embedding function if needed later
vector_collection_name = "recipes"
vector_store = chroma_client.get_or_create_collection(name=vector_collection_name, embedding_function=chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(api_key=settings.OPENAI_API_KEY, model_name="text-embedding-ada-002"))

class LangchainChatInput(BaseModel):
    prompt: str
    chat_id: Optional[str] = None

@router.post("/langchain", response_model=Dict[str, Any])
async def langchain_chat(chat_input: LangchainChatInput):
    """
    Handle chat requests using Langchain chains, manually managing history
    in a single MongoDB document per chat session.
    """
    user_input = chat_input.prompt
    chat_id = chat_input.chat_id
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
                    extraction_response = oai_client.responses.create( # Use async client if available, otherwise sync
                        model="gpt-4o-mini", # Or a model suitable for extraction
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
                        recipe = RecipeCreate(
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

                        # Store in MongoDB
                        recipe_dict = recipe.model_dump()
                        insert_result = await recipe_collection.insert_one(recipe_dict)
                        created_recipe_id = insert_result.inserted_id
                        created_recipe = await recipe_collection.find_one({"_id": created_recipe_id})
                        print(f"Stored AI-generated recipe with ID: {created_recipe_id}")

                        # Add to Chroma Vector Store
                        try:
                            page_content = f"Title: {created_recipe['title']}\n"
                            # ... (build page_content string including new fields like in recipe.py/store) ...
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
                                "source": created_recipe.get('source_url', 'ai_generated'),
                                "title": created_recipe['title'],
                                "mongo_id": str(created_recipe['_id'])
                            }
                            # ... (add other relevant metadata like in recipe.py/store) ...
                            if created_recipe.get('cuisine'): metadata['cuisine'] = created_recipe['cuisine']
                            if created_recipe.get('meal_type'): metadata['meal_type'] = created_recipe['meal_type']
                            if created_recipe.get('total_time_in_mins') is not None: metadata['total_time_in_mins'] = created_recipe['total_time_in_mins']
                            if created_recipe.get('protein_grams') is not None: metadata['protein_grams'] = created_recipe['protein_grams']
                            if created_recipe.get('fat_grams') is not None: metadata['fat_grams'] = created_recipe['fat_grams']


                            doc = Document(page_content=page_content, metadata=metadata)
                            vector_store.add(
                                documents=[doc.page_content],
                                metadatas=[doc.metadata],
                                ids=[str(created_recipe['_id'])]
                            )
                            print(f"Added AI-generated recipe {created_recipe['_id']} to Chroma.")
                        except Exception as chroma_e:
                            print(f"Error adding AI-generated recipe {created_recipe['_id']} to Chroma: {chroma_e}")
                            # Decide if you want to notify user about Chroma failure

                        # Set the user-facing answer
                        final_answer_for_user = (f"I couldn't find a matching recipe in the database, but I generated one using AI and have now added it for future reference.\n\n"
                                                 f"Title: {created_recipe['title']}\n"
                                                 f"(You can ask me about '{created_recipe['title']}' now.)")
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


# --- Existing Chat Endpoints (GET, DELETE) ---

@router.get("/", response_model=List[Dict[str, Any]])
async def get_all_chats(limit: int = 50, skip: int = 0, type: Optional[str] = None):
    """
    Retrieve chat history entries with pagination and optional type filtering.
    
    Args:
        limit: Maximum number of chat histories to return (default: 50)
        skip: Number of chat histories to skip (default: 0)
        type: Optional chat type to filter by ('gpt' or 'langchain')
        
    Returns:
        List of chat history objects
    """
    try:
        query = {}
        if type:
            query["type"] = type
            
        # Query chats with filter and pagination
        chats_cursor = chat_history_collection.find(query).sort("_id", -1).skip(skip).limit(limit) # Sort by newest
        chats = await chats_cursor.to_list(length=limit)

        # Format the results
        formatted_chats = []
        for chat in chats:
            chat_doc = {}
            chat_doc['id'] = str(chat['_id'])
            chat_doc['type'] = chat.get('type') # Include type
            # Include a preview or count if needed
            messages = chat.get('messages', [])
            if messages:
                 chat_doc['preview'] = messages[0].get('content', '')[:50] + '...' # First 50 chars
            else:
                 chat_doc['preview'] = 'Empty chat'
            chat_doc['message_count'] = len(messages)
            # chat_doc['messages'] = messages # Optionally return all messages here too

            formatted_chats.append(chat_doc)

        return formatted_chats

    except Exception as e:
        print(f"Error retrieving chats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat histories: {str(e)}")


@router.get("/{chat_id}", response_model=Dict[str, Any])
async def get_chat_by_id(chat_id: str):
    # This endpoint now correctly fetches the single document and its messages
    try:
        # Validate chat_id format
        if not ObjectId.is_valid(chat_id):
            raise HTTPException(status_code=400, detail=f"Invalid chat ID format: {chat_id}")

        # Query the specific chat document
        chat = await chat_history_collection.find_one({"_id": ObjectId(chat_id)})

        if not chat:
            raise HTTPException(status_code=404, detail=f"Chat with ID {chat_id} not found")

        # Format the result - include type
        return {
            "id": str(chat["_id"]),
            "messages": chat.get("messages", []),
            "type": chat.get("type") # Include the chat type
        }

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(f"Error retrieving chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")


@router.delete("/{chat_id}", response_model=Dict[str, str])
async def delete_chat(chat_id: str):
    # This endpoint remains the same, deleting the single document
    try:
        # Validate chat_id format
        if not ObjectId.is_valid(chat_id):
            raise HTTPException(status_code=400, detail=f"Invalid chat ID format: {chat_id}")

        # Delete the chat history document
        result = await chat_history_collection.delete_one({"_id": ObjectId(chat_id)})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Chat with ID {chat_id} not found")

        return {"message": f"Chat with ID {chat_id} successfully deleted"}

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(f"Error deleting chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete chat history: {str(e)}")