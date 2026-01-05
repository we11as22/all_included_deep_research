'use client';

import { useEffect, useRef } from 'react';
import { LocalChatMessage } from '@/stores/chatStore';
import { MessageItem } from './MessageItem';

interface MessageListProps {
  messages: LocalChatMessage[];
}

export function MessageList({ messages }: MessageListProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const lastMessageCountRef = useRef(messages.length);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messages.length > lastMessageCountRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
    lastMessageCountRef.current = messages.length;
  }, [messages.length]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center space-y-4 max-w-md">
          <h3 className="text-2xl font-semibold">Start a conversation</h3>
          <p className="text-muted-foreground">
            Ask me anything! I can help you research topics with web search, deep search, or comprehensive research modes.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4">
      {messages.map((message) => (
        <MessageItem key={message.id} message={message} />
      ))}
    </div>
  );
}
