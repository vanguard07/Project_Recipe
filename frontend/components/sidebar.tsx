"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { usePathname, useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { PlusCircle, MessageSquare, Trash2 } from "lucide-react"
import { cn } from "@/lib/utils"

type Chat = {
  id: string
  title: string
  type: "langchain" | "gpt"
  createdAt: Date
}

export default function Sidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const [activeTab, setActiveTab] = useState<"langchain" | "gpt" | "recipes">("langchain")
  const [chats, setChats] = useState<Chat[]>([])
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    // Set active tab based on URL
    if (pathname.includes("/chat/langchain")) {
      setActiveTab("langchain")
    } else if (pathname.includes("/chat/gpt")) {
      setActiveTab("gpt")
    } else if (pathname.includes("/recipes")) {
      setActiveTab("recipes")
    }

    // Fetch chats from backend when the active tab changes
    fetchChats()
  }, [pathname, activeTab])

  // Function to fetch chats from the backend API
  const fetchChats = async () => {
    if (activeTab === "recipes") return;
    
    setIsLoading(true);
    try {
      // Fetch chats from the backend with the current active tab as type filter
      const response = await fetch(`http://localhost:8000/chat/?type=${activeTab}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch chats: ${response.status}`);
      }
      
      const chatData = await response.json();
      
      // Transform the data to match our Chat type
      const transformedChats: Chat[] = chatData.map((chat: any) => ({
        id: chat.id,
        title: chat.preview || `Chat ${chat.id.substring(0, 6)}...`,
        type: chat.type as "langchain" | "gpt",
        createdAt: new Date()
      }));
      
      setChats(transformedChats);
    } catch (error) {
      console.error("Error fetching chats:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const deleteChat = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    
    try {
      // Call backend API to delete the chat
      const response = await fetch(`http://localhost:8000/chat/${id}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        console.error(`Failed to delete chat from backend: ${response.status}`);
        return;
      }
      
      console.log(`Successfully deleted chat ${id} from backend`);
      
      // Update local state
      const updatedChats = chats.filter((chat) => chat.id !== id)
      setChats(updatedChats)

      // If we're currently on the chat that's being deleted, navigate back to the main tab
      if (pathname.includes(id)) {
        router.push(`/chat/${activeTab}`)
      }
      
    } catch (error) {
      console.error("Error deleting chat:", error);
    }
  }

  const handleTabClick = (tab: "langchain" | "gpt" | "recipes") => {
    setActiveTab(tab)
    if (tab === "recipes") {
      router.push("/recipes")
    } else {
      router.push(`/chat/${tab}`)
    }
  }

  const filteredChats = chats.filter((chat) => chat.type === activeTab)

  return (
    <div className="w-64 bg-background border-r h-full flex flex-col">
      <div className="p-4 border-b">
        <h1 className="text-xl font-bold">Recipe Chat</h1>
      </div>

      <div className="flex border-b">
        <Button
          variant="ghost"
          className={cn("flex-1 rounded-none", activeTab === "langchain" && "bg-muted")}
          onClick={() => handleTabClick("langchain")}
        >
          LangChain
        </Button>

        <Button
          variant="ghost"
          className={cn("flex-1 rounded-none", activeTab === "gpt" && "bg-muted")}
          onClick={() => handleTabClick("gpt")}
        >
          GPT
        </Button>

        <Button
          variant="ghost"
          className={cn("flex-1 rounded-none", activeTab === "recipes" && "bg-muted")}
          onClick={() => handleTabClick("recipes")}
        >
          Recipes
        </Button>
      </div>

      {(activeTab === "langchain" || activeTab === "gpt") && (
        <>
          <div className="p-2">
            <Button
              variant="outline"
              className="w-full justify-start"
              onClick={() => router.push(`/chat/${activeTab}/new`)}
            >
              <PlusCircle className="mr-2 h-4 w-4" />
              New Chat
            </Button>
          </div>

          <div className="flex-1 overflow-auto p-2">
            <h2 className="text-sm font-semibold mb-2">Chat History</h2>
            {isLoading ? (
              <p className="text-sm text-muted-foreground">Loading chats...</p>
            ) : filteredChats.length > 0 ? (
              <ul className="space-y-1">
                {filteredChats.map((chat) => (
                  <li key={chat.id} className="flex items-center group">
                    <Button
                      variant="ghost"
                      className={cn(
                        "flex-1 h-auto justify-start text-sm py-1 px-2 rounded hover:bg-muted truncate",
                        pathname.includes(chat.id) && "bg-muted",
                      )}
                      onClick={() => router.push(`/chat/${chat.type}/${chat.id}`)}
                    >
                      <MessageSquare className="inline-block mr-2 h-4 w-4" />
                      {chat.title}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 opacity-0 group-hover:opacity-100"
                      onClick={(e) => deleteChat(chat.id, e)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted-foreground">No chats yet</p>
            )}
          </div>
        </>
      )}
    </div>
  )
}
