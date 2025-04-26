import React, { useState, useEffect } from 'react';
import './Layout.css';
import Chat from './Chat';
import RecipeStore from './RecipeStore';
// Import useLocation
import { useNavigate, useParams, useLocation } from 'react-router-dom';
import axios from 'axios';

const Layout = () => {
  const [chats, setChats] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('gpt'); // Default to GPT chat
  const navigate = useNavigate();
  const { chatId } = useParams();
  const location = useLocation(); // Get location object

  // Determine the chatId to pass to the Chat component based on the URL path
  const chatComponentChatId = location.pathname.startsWith('/chat/') ? chatId : undefined;

  // Fetch chat histories when component mounts or activeTab changes
  useEffect(() => {
    if (activeTab === 'gpt' || activeTab === 'langchain') {
      fetchChats(activeTab);
    }
    // Clear chats list when switching to store tab
    if (activeTab === 'store') {
        setChats([]);
    }
    // If a chatId is present in URL, try to determine the correct tab
    // This might require an extra fetch or logic if switching tabs directly via URL isn't the primary flow
  }, [activeTab]); // Re-fetch when activeTab changes

  // Fetch specific chat when chatId changes (if needed, or rely on Chat component)
  // useEffect(() => {
  //   if (chatId) { /* Potentially fetch chat details to set activeTab? */ }
  // }, [chatId]);

  const fetchChats = async (type) => {
    setIsLoading(true);
    try {
      // Pass the chat type as a query parameter
      const response = await axios.get(`http://localhost:8000/chat?type=${type}`);
      setChats(response.data);
    } catch (error) {
      console.error('Error fetching chats:', error);
      setChats([]); // Clear chats on error
    } finally {
      setIsLoading(false);
    }
  };

  const createNewChat = () => {
    // Navigate to base URL, the Chat component will handle creating a new chat for the active tab type
    navigate('/');
  };

  const selectChat = (chatId) => {
    // Navigate to the specific chat URL, activeTab remains the same
    navigate(`/chat/${chatId}`);
  };

  const deleteChat = async (chatId, event) => {
    event.stopPropagation();
    try {
      await axios.delete(`http://localhost:8000/chat/${chatId}`);
      // Refetch chats for the current active tab after deletion
      fetchChats(activeTab);
      
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
            className={`tab ${activeTab === 'gpt' ? 'active' : ''}`} 
            onClick={() => setActiveTab('gpt')}
          >
            GPT Chat
          </button>
          <button 
            className={`tab ${activeTab === 'langchain' ? 'active' : ''}`} 
            onClick={() => setActiveTab('langchain')}
          >
            Langchain Chat
          </button>
          <button 
            className={`tab ${activeTab === 'store' ? 'active' : ''}`} 
            onClick={() => setActiveTab('store')}
          >
            Store Recipe
          </button>
        </div>

        {(activeTab === 'gpt' || activeTab === 'langchain') && (
          <div className="chat-list">
            {isLoading ? (
              <div className="loading-chats">Loading...</div>
            ) : chats.length === 0 ? (
              <div className="empty-chats">No {activeTab === 'gpt' ? 'GPT' : 'Langchain'} chats</div>
            ) : (
              // Use chatComponentChatId for highlighting the active item
              chats.map((chat) => (
                <div 
                  key={chat.id} 
                  className={`chat-item ${chat.id === chatComponentChatId ? 'active' : ''}`}
                  onClick={() => selectChat(chat.id)}
                >
                  {/* Use preview from backend if available, otherwise format */}
                  <span className="chat-title">{chat.preview || formatChatPreview(chat.messages)}</span>
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
        {/* Pass the conditionally determined chatComponentChatId to Chat components */}
        {activeTab === 'gpt' ? (
          <Chat chatId={chatComponentChatId} chatType="gpt" onChatCreated={() => fetchChats('gpt')} />
        ) : activeTab === 'langchain' ? (
          <Chat chatId={chatComponentChatId} chatType="langchain" onChatCreated={() => fetchChats('langchain')} />
        ) : (
          <RecipeStore />
        )}
      </div>
    </div>
  );
};

export default Layout;