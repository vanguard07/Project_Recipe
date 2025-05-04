"use client"

import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { cn } from "@/lib/utils"
import ReactMarkdown from "react-markdown"

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

export function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user"

  return (
    <div className={cn("flex items-start gap-3", isUser && "flex-row-reverse")}>
      <Avatar className={cn("h-8 w-8", isUser ? "bg-primary" : "bg-muted")}>
        <AvatarFallback>{isUser ? "U" : "AI"}</AvatarFallback>
      </Avatar>

      <div
        className={cn("rounded-lg px-4 py-2 max-w-[80%]", isUser ? "bg-primary text-primary-foreground" : "bg-muted")}
      >
        <div className="prose prose-sm dark:prose-invert">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
        <div className={cn("text-xs mt-1", isUser ? "text-primary-foreground/70" : "text-muted-foreground")}>
          {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </div>
      </div>
    </div>
  )
}
