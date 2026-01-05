'use client';

import { FormEvent } from 'react';
import { useChatStore } from '@/stores/chatStore';
import { useChat } from '@/hooks/useChat';
import { Textarea } from './ui/textarea';
import { Button } from './ui/button';
import { ModeSelectorDropdown } from './ModeSelectorDropdown';
import { ArrowUpRight, X } from 'lucide-react';

export function ChatInput() {
  const input = useChatStore((state) => state.input);
  const mode = useChatStore((state) => state.mode);
  const isStreaming = useChatStore((state) => state.isStreaming);
  const setInput = useChatStore((state) => state.setInput);
  const setMode = useChatStore((state) => state.setMode);

  const { sendMessage, cancelStream } = useChat();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    const messageContent = input;
    setInput(''); // Clear input immediately for better UX

    try {
      await sendMessage(messageContent);
    } catch (error) {
      console.error('Failed to send message:', error);
      // Restore input on error
      setInput(messageContent);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  const handleCancel = () => {
    cancelStream();
  };

  return (
    <form onSubmit={handleSubmit} className="shrink-0 border-t bg-background p-4">
      <div className="flex gap-2 items-end">
        <ModeSelectorDropdown value={mode} onChange={setMode} disabled={isStreaming} />
        <div className="flex-1">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isStreaming ? 'Streaming response...' : 'Ask anything...'}
            disabled={isStreaming}
            className="min-h-[60px] max-h-[200px] resize-none"
            rows={2}
          />
        </div>
        {isStreaming ? (
          <Button
            type="button"
            variant="destructive"
            size="icon"
            onClick={handleCancel}
            title="Cancel streaming"
          >
            <X className="h-4 w-4" />
          </Button>
        ) : (
          <Button
            type="submit"
            size="icon"
            disabled={!input.trim() || isStreaming}
            title="Send message (Enter)"
          >
            <ArrowUpRight className="h-4 w-4" />
          </Button>
        )}
      </div>
      <div className="mt-2 text-xs text-muted-foreground">
        Press Enter to send, Shift + Enter for new line
      </div>
    </form>
  );
}
