import ChatInterface from "@/components/chat-interface"

export default function GptChatPage({ params }: { params: { chatId: string } }) {
  return <ChatInterface chatType="gpt" chatId={params.chatId} />
}
