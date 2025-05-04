import ChatInterface from "@/components/chat-interface"

export default function LangchainChatPage({ params }: { params: { chatId: string } }) {
  return <ChatInterface chatType="langchain" chatId={params.chatId} />
}
