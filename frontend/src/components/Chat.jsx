// src/components/Chat.jsx
import React, { useState, useRef, useEffect } from 'react';
import './Chat.css';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Chat = ({ chatId, onChatCreated }) => {
  const [messages, setMessages] = useState([{
    text: "Welcome to RecipeGPT! You can search for recipes or ask for customizations. Try something like 'Find me pasta recipes' or 'How can I make this pasta dish dairy-free?'",
    sender: 'bot',
    timestamp: new Date().toISOString()
  }]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  // Store the current chat ID in state to use it for all requests
  const [currentChatId, setCurrentChatId] = useState(chatId);
  const messagesEndRef = useRef(null);
  const navigate = useNavigate();

  // Auto-scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Update currentChatId when props chatId changes (when switching chats)
  useEffect(() => {
    setCurrentChatId(chatId);
  }, [chatId]);

  // Load chat messages when chatId changes
  useEffect(() => {
    if (chatId) {
      loadChatHistory(chatId);
    } else {
      // Reset to initial state for new chat
      setMessages([{
        text: "Welcome to RecipeGPT! You can search for recipes or ask for customizations. Try something like 'Find me pasta recipes' or 'How can I make this pasta dish dairy-free?'",
        sender: 'bot',
        timestamp: new Date().toISOString()
      }]);
    }
  }, [chatId]);

  const loadChatHistory = async (id) => {
    setIsLoading(true);
    try {
      const response = await axios.get(`http://localhost:8000/chat/${id}`);
      if (response.data && response.data.messages) {
        // Convert the chat history format to our message format
        const formattedMessages = response.data.messages.map(message => {
          let content = message.content;
          
          // Try to parse content if it's a stringified JSON array
          if (typeof content === 'string' && content.startsWith('[') && content.endsWith(']')) {
            try {
              const parsed = JSON.parse(content);
              if (Array.isArray(parsed)) {
                content = parsed.join('\n\n');
              }
            } catch (e) {
              // Keep as string if not valid JSON
            }
          }
          
          return {
            text: content,
            sender: message.role === 'assistant' ? 'bot' : 'user',
            timestamp: new Date().toISOString()
          };
        });
        setMessages(formattedMessages);
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

    // Add user message
    const userMessage = {
      text: input,
      sender: 'user',
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      let response;
      
      // If we have a currentChatId, we're in an existing conversation
      if (currentChatId) {
        response = await axios.post('http://localhost:8000/recipe/classify', {
          prompt: input,
          chat_id: currentChatId
        });
      } else {
        // For new chat, first classify the query type
        const classifyResponse = await axios.post('http://localhost:8000/recipe/classify', {
          prompt: input
        });
        
        // Store the newly created chat ID
        if (classifyResponse.data.chat_id) {
          const newChatId = classifyResponse.data.chat_id;
          
          // Update the current chat ID in state
          setCurrentChatId(newChatId);
          
          // Notify parent about new chat creation
          if (onChatCreated) onChatCreated();
          
          // Update the URL properly using navigate
          navigate(`/chat/${newChatId}`);
        }
        
        response = classifyResponse;
      }

      // Add bot response
      const botMessage = {
        text: response.data.result || response.data.results?.join('\n\n') || "Sorry, I couldn't process that request.",
        sender: 'bot',
        timestamp: new Date().toISOString(),
        type: response.data.type
      };
      
      setMessages(prev => [...prev, botMessage]);
      
    } catch (error) {
      console.error('Error:', error);
      // Add error message
      const errorMessage = {
        text: "Sorry, there was an error processing your request.",
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
        <h2>RecipeGPT Chat</h2>
        <p className="chat-subtitle">Search for recipes or request customizations in one place</p>
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
          placeholder="Search for recipes or ask for customizations..."
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