// src/components/Chat.jsx
import React, { useState, useRef, useEffect } from 'react';
import './Chat.css';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Chat = ({ chatId, chatType, onChatCreated }) => { // Accept chatType prop
  const [messages, setMessages] = useState([]); // Initialize empty
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentChatId, setCurrentChatId] = useState(chatId);
  const messagesEndRef = useRef(null);
  const navigate = useNavigate();

  // Initial welcome message based on chat type
  const getWelcomeMessage = (type) => {
    if (type === 'langchain') {
      return {
        text: "Welcome to Langchain Recipe Chat! Ask me anything about the stored recipes.",
        sender: 'bot',
        timestamp: new Date().toISOString()
      };
    } else { // Default to GPT
      return {
        text: "Welcome to GPT Recipe Chat! You can search for recipes or ask for customizations. Try something like 'Find me pasta recipes' or 'How can I make this pasta dish dairy-free?'",
        sender: 'bot',
        timestamp: new Date().toISOString()
      };
    }
  };

  // Auto-scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Update currentChatId when props chatId changes
  useEffect(() => {
    setCurrentChatId(chatId);
  }, [chatId]);

  // Load chat messages when chatId changes or chatType changes (to reset view)
  useEffect(() => {
    if (chatId) {
      loadChatHistory(chatId);
    } else {
      // Reset to initial state for new chat based on type
      setMessages([getWelcomeMessage(chatType)]);
      setCurrentChatId(null); // Ensure currentChatId is reset for new chat
    }
  }, [chatId, chatType]); // Add chatType dependency

  const loadChatHistory = async (id) => {
    setIsLoading(true);
    try {
      const response = await axios.get(`http://localhost:8000/chat/${id}`);
      if (response.data && response.data.messages) {
        // Convert the chat history format
        const formattedMessages = response.data.messages.map(message => {
          let content = message.content;
          // ... (existing content parsing logic) ...
          return {
            text: content,
            sender: message.role === 'assistant' ? 'bot' : 'user',
            timestamp: new Date().toISOString() // Use current time or ideally timestamp from DB if available
          };
        });
        // Set messages, add welcome message if history is empty
        setMessages(formattedMessages.length > 0 ? formattedMessages : [getWelcomeMessage(chatType)]);
      } else {
        // If no messages found, show welcome message
        setMessages([getWelcomeMessage(chatType)]);
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
      setMessages([{
        text: "Error loading chat history. Please try again.",
        sender: 'bot',
        timestamp: new Date().toISOString(),
        isError: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = {
      text: input,
      sender: 'user',
      timestamp: new Date().toISOString()
    };
    
    // Optimistically update UI
    setMessages(prev => [...prev, userMessage]);
    const currentInput = input; // Store input before clearing
    setInput('');
    setIsLoading(true);

    try {
      let response;
      const payload = {
        prompt: currentInput,
        chat_id: currentChatId // Send currentChatId (null if new chat)
      };

      let endpoint = '';
      if (chatType === 'langchain') {
        endpoint = 'http://localhost:8000/recipe/langchain';
      } else { // Default to GPT
        endpoint = 'http://localhost:8000/recipe/classify';
      }

      response = await axios.post(endpoint, payload);

      let botResponseText = "Sorry, I couldn't process that request.";
      let newChatId = currentChatId; // Assume existing chat ID unless updated

      if (chatType === 'langchain') {
        botResponseText = response.data.answer || botResponseText;
        newChatId = response.data.chat_id; // Langchain endpoint returns chat_id
      } else { // GPT
        botResponseText = response.data.result || response.data.results?.join('\n\n') || botResponseText;
        newChatId = response.data.chat_id; // Classify endpoint returns chat_id
      }

      const botMessage = {
        text: botResponseText,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        // type: response.data.type // Include type if relevant for GPT response styling
      };
      
      setMessages(prev => [...prev, botMessage]);

      // If it was a new chat, update state and URL
      if (!currentChatId && newChatId) {
        setCurrentChatId(newChatId);
        if (onChatCreated) onChatCreated(); // Notify layout to refresh list
        navigate(`/chat/${newChatId}`); // Update URL
      }
      
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        text: error.response?.data?.detail || "Sorry, there was an error processing your request.",
        sender: 'bot',
        timestamp: new Date().toISOString(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Render changes as tags for customization messages
  const renderChanges = (changes) => {
    if (!changes || changes.length === 0) return null;
    
    return (
      <div className="change-tags">
        {changes.map((change, idx) => (
          <span key={idx} className={`change-tag ${change.type}`}>
            {change.type.replace('_', ' ')}
          </span>
        ))}
      </div>
    );
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        {/* Update title based on chatType */}
        <h2>{chatType === 'langchain' ? 'Langchain Chat' : 'GPT Chat'}</h2>
        <p className="chat-subtitle">{chatType === 'langchain' ? 'Chat with recipes using vector search' : 'Search for recipes or request customizations'}</p>
      </div>
      
      <div className="messages-container">
        {messages.map((message, index) => (
          <div 
            key={index} 
            className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'} ${message.isError ? 'error-message' : ''} ${message.type ? `type-${message.type}` : ''}`}
          >
            <div className="message-content">
              {message.text}
              {message.changes && renderChanges(message.changes)}
            </div>
            <div className="message-timestamp">
              {new Date(message.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message bot-message loading-message">
            <div className="loading-indicator">
              <div className="dot"></div>
              <div className="dot"></div>
              <div className="dot"></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="input-container" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={chatType === 'langchain' ? 'Ask about stored recipes...' : 'Search or customize recipes...'}
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default Chat;