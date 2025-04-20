from fastapi import APIRouter, HTTPException
from app.db.database import chat_history_collection
from bson.objectid import ObjectId
from typing import List, Dict, Any, Optional
import json

router = APIRouter()

@router.get("/", response_model=List[Dict[str, Any]])
async def get_all_chats(limit: int = 50, skip: int = 0):
    """
    Retrieve all chat history entries with pagination support.
    
    Args:
        limit: Maximum number of chat histories to return (default: 50)
        skip: Number of chat histories to skip (default: 0)
        
    Returns:
        List of chat history objects
    """
    try:
        # Query all chats with pagination
        chats = await chat_history_collection.find().skip(skip).limit(limit).to_list(length=None)
        
        # Format the results
        formatted_chats = []
        for chat in chats:
            # Convert ObjectId to string
            chat['id'] = str(chat['_id'])
            del chat['_id']
            
            # # Add metadata
            # message_count = len(chat.get('messages', []))
            # last_message = chat.get('messages', [])[-1] if message_count > 0 else None
            # last_message_text = last_message.get('content', '') if last_message else ''
            
            # # Truncate long messages for preview
            # if last_message_text and len(last_message_text) > 100:
            #     last_message_text = last_message_text[:100] + "..."
                
            # chat['message_count'] = message_count
            # chat['last_message'] = last_message_text
            # chat['last_updated'] = chat.get('last_updated', None)
            
            formatted_chats.append(chat)
            
        return formatted_chats
        
    except Exception as e:
        print(f"Error retrieving chats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat histories: {str(e)}")

@router.get("/{chat_id}", response_model=Dict[str, Any])
async def get_chat_by_id(chat_id: str):
    """
    Retrieve a specific chat history by ID.
    
    Args:
        chat_id: The ID of the chat history to retrieve
        
    Returns:
        Chat history object with messages
    """
    try:
        # Validate chat_id format
        if not ObjectId.is_valid(chat_id):
            raise HTTPException(status_code=400, detail=f"Invalid chat ID format: {chat_id}")
        
        # Query the specific chat
        chat = await chat_history_collection.find_one({"_id": ObjectId(chat_id)})
        
        if not chat:
            raise HTTPException(status_code=404, detail=f"Chat with ID {chat_id} not found")
        
        # Format the result
        chat['id'] = str(chat['_id'])
        del chat['_id']
        
        # Ensure messages are properly formatted
        if 'messages' in chat:
            for message in chat['messages']:
                # Handle potential JSON strings in content
                if isinstance(message.get('content'), str) and message.get('content').startswith('[') and message.get('content').endswith(']'):
                    try:
                        message['content'] = json.loads(message['content'])
                    except json.JSONDecodeError:
                        # Keep as string if not valid JSON
                        pass
        
        return chat
        
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(f"Error retrieving chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")

@router.delete("/{chat_id}", response_model=Dict[str, str])
async def delete_chat(chat_id: str):
    """
    Delete a specific chat history by ID.
    
    Args:
        chat_id: The ID of the chat history to delete
        
    Returns:
        Confirmation message
    """
    try:
        # Validate chat_id format
        if not ObjectId.is_valid(chat_id):
            raise HTTPException(status_code=400, detail=f"Invalid chat ID format: {chat_id}")
        
        # Delete the chat history
        result = await chat_history_collection.delete_one({"_id": ObjectId(chat_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Chat with ID {chat_id} not found")
        
        return {"message": f"Chat with ID {chat_id} successfully deleted"}
        
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(f"Error deleting chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete chat history: {str(e)}")