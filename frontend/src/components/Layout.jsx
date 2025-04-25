import React, { useState, useEffect } from 'react';
import './Layout.css';
import Chat from './Chat';
import RecipeStore from './RecipeStore';
import { useNavigate, useParams } from 'react-router-dom';
import axios from 'axios';

const Layout = () => {
  const [chats, setChats] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const navigate = useNavigate();
  const { chatId } = useParams();

  // Fetch chat histories when component mounts
  useEffect(() => {
    fetchChats();
  }, []);

  const fetchChats = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get('http://localhost:8000/chat');
      setChats(response.data);
    } catch (error) {
      console.error('Error fetching chats:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const createNewChat = () => {
    navigate('/');
  };

  const selectChat = (chatId) => {
    navigate(`/chat/${chatId}`);
  };

  const deleteChat = async (chatId, event) => {
    event.stopPropagation();
    try {
      await axios.delete(`http://localhost:8000/chat/${chatId}`);
      setChats(chats.filter(chat => chat.id !== chatId));
      
      // If the deleted chat was active, navigate to home
      if (window.location.pathname.includes(chatId)) {
        navigate('/');
      }
    } catch (error) {
      console.error('Error deleting chat:', error);
    }
  };

  // Format a preview of the chat content
  const formatChatPreview = (messages) => {
    if (!messages || messages.length === 0) {
      return "Empty chat";
    }
    
    // Find the first user message to use as title
    const firstUserMessage = messages.find(m => m.role === 'user');
    if (firstUserMessage) {
      let content = firstUserMessage.content;
      // Truncate if too long
      if (content.length > 30) {
        content = content.substring(0, 30) + '...';
      }
      return content;
    }
    
    return "New conversation";
  };

  return (
    <div className="layout">
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>RecipeGPT</h2>
          <button className="new-chat-btn" onClick={createNewChat}>
            <span>+</span> New Chat
          </button>
        </div>

        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'chat' ? 'active' : ''}`} 
            onClick={() => setActiveTab('chat')}
          >
            Chat
          </button>
          <button 
            className={`tab ${activeTab === 'store' ? 'active' : ''}`} 
            onClick={() => setActiveTab('store')}
          >
            Store Recipe
          </button>
        </div>

        {activeTab === 'chat' && (
          <div className="chat-list">
            {isLoading ? (
              <div className="loading-chats">Loading...</div>
            ) : chats.length === 0 ? (
              <div className="empty-chats">No chat history</div>
            ) : (
              chats.map((chat) => (
                <div 
                  key={chat.id} 
                  className={`chat-item ${chat.id === chatId ? 'active' : ''}`}
                  onClick={() => selectChat(chat.id)}
                >
                  <span className="chat-title">{formatChatPreview(chat.messages)}</span>
                  <button 
                    className="delete-chat-btn"
                    onClick={(e) => deleteChat(chat.id, e)}
                    aria-label="Delete chat"
                  >
                    &times;
                  </button>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      <div className="main-content">
        {activeTab === 'chat' ? (
          <Chat chatId={chatId} onChatCreated={fetchChats} />
        ) : (
          <RecipeStore />
        )}
      </div>
    </div>
  );
};

export default Layout;