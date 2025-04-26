from fastapi import APIRouter, HTTPException, Body
from app.db.database import chat_history_collection, recipe_collection, db, client
from app.core.config import settings
from bson.objectid import ObjectId
from typing import List, Dict, Any, Optional

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
        answer = result.get("answer", "Sorry, I couldn't generate a response.")

        # Append current interaction to the list for DB update
        messages_for_db.append({"role": "user", "content": user_input})
        messages_for_db.append({"role": "assistant", "content": answer})

        # Update the single chat document in MongoDB
        await chat_history_collection.update_one(
            {"_id": ObjectId(chat_id)},
            {"$set": {"messages": messages_for_db, "type": "langchain"}}, # Add type field
            upsert=True # Create the document if it doesn't exist
        )

        # Context/source documents might be nested differently
        # source_documents = result.get("context", [])
        # sources = []
        # if source_documents:
        #     sources = [...] # Format sources if needed

        return {
            "answer": answer,
            "chat_id": chat_id, # Return the used/generated chat_id
            # "sources": sources
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