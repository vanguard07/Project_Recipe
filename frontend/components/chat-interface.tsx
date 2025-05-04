"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Send, Loader2 } from "lucide-react"
import { ChatMessage } from "@/components/chat-message"

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

type ChatType = "langchain" | "gpt"

interface ChatInterfaceProps {
  chatType: ChatType
  chatId?: string
  isNew?: boolean
}

// Loading indicator component
const LoadingIndicator = () => (
  <div className="flex items-center space-x-2 p-4 bg-muted/50 rounded-lg mb-4 animate-pulse">
    <Loader2 className="h-4 w-4 animate-spin" />
    <span className="text-sm">Processing your request...</span>
  </div>
);

export default function ChatInterface({ chatType, chatId, isNew = false }: ChatInterfaceProps) {
  const router = useRouter()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // If we have a chatId, load messages for this chat from the backend
    if (chatId && !isNew) {
      fetchMessages(chatId);
    }
  }, [chatId, isNew])

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Function to fetch messages from the backend API
  const fetchMessages = async (id: string) => {
    try {
      const response = await fetch(`http://localhost:8000/chat/${id}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch chat messages: ${response.status}`);
      }
      
      const chatData = await response.json();
      
      // Transform the API response to match our Message type
      const transformedMessages: Message[] = chatData.messages.map((msg: any, index: number) => ({
        id: index.toString(), // Use index as ID since backend might not have message IDs
        role: msg.role as "user" | "assistant",
        content: msg.content,
        timestamp: new Date() // Backend might not provide timestamps
      }));
      
      setMessages(transformedMessages);
    } catch (error) {
      console.error("Error fetching chat messages:", error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    }

    const updatedMessages = [...messages, userMessage]
    setMessages(updatedMessages)
    setInput("")
    setIsLoading(true)

    try {
      // Call to backend API
      const response = await backendCall(chatType, input, chatId)

      // Get the chat_id from the response
      const responseId = response.chat_id || chatId

      // If this is a new chat, redirect to the new chat URL with the backend-provided ID
      if (isNew && response.chat_id) {
        router.push(`/chat/${chatType}/${response.chat_id}`)
      }

      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: response.result,
        timestamp: new Date(),
      }

      const finalMessages = [...updatedMessages, assistantMessage]
      setMessages(finalMessages)

    } catch (error) {
      console.error("Error getting response:", error)
      // Add error message
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: "Sorry, there was an error processing your request.",
        timestamp: new Date(),
      }

      const finalMessages = [...updatedMessages, errorMessage]
      setMessages(finalMessages)

    } finally {
      setIsLoading(false)
    }
  }

  // Make API call to backend
  const backendCall = async (
    type: ChatType,
    message: string,
    existingChatId?: string,
  ): Promise<{ result: string; chat_id?: string }> => {
    try {
      const endpoint = type === "langchain" ? "/recipe/langchain" : "/recipe/classify";
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: message,
          chat_id: existingChatId,
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();
      
      return {
        result: type === "langchain" ? data.answer : data.result,
        chat_id: data.chat_id,
      };
    } catch (error) {
      console.error("API call error:", error);
      throw error; // Rethrow the error to be handled in the calling function
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-2">Start a new conversation</h2>
              <p className="text-muted-foreground">
                {chatType === "langchain"
                  ? "Ask about recipes using LangChain AI"
                  : "Chat with GPT about your favorite recipes"}
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {isLoading && <LoadingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="border-t p-4">
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={isLoading}
            className="flex-1"
          />
          <Button type="submit" disabled={isLoading || !input.trim()}>
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </form>
      </div>
    </div>
  )
}
