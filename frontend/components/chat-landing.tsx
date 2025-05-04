"use client"

import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"
import { MessageSquare, PlusCircle } from "lucide-react"

export default function ChatLanding({ chatType }: { chatType: "langchain" | "gpt" }) {
  const router = useRouter()

  return (
    <div className="flex flex-col items-center justify-center h-full p-4">
      <div className="text-center max-w-md">
        <MessageSquare className="mx-auto h-12 w-12 text-primary mb-4" />
        <h1 className="text-2xl font-bold mb-2">
          {chatType === "langchain" ? "LangChain Recipe Chat" : "GPT Recipe Chat"}
        </h1>
        <p className="text-muted-foreground mb-6">
          {chatType === "langchain"
            ? "Chat with your recipes using LangChain's powerful AI capabilities."
            : "Use OpenAI's GPT models to get recipe suggestions and cooking advice."}
        </p>
        <Button size="lg" onClick={() => router.push(`/chat/${chatType}/new`)} className="mx-auto">
          <PlusCircle className="mr-2 h-5 w-5" />
          Start a New Chat
        </Button>
      </div>
    </div>
  )
}
