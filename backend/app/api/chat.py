from fastapi import APIRouter, HTTPException
from app.db.database import chat_history_collection
from bson.objectid import ObjectId
from typing import List, Dict, Any, Optional

# Langchain imports

router = APIRouter()


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