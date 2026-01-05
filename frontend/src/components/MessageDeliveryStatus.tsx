'use client';

import { LocalChatMessage } from '@/stores/chatStore';
import { Loader2, Check, AlertCircle, Clock } from 'lucide-react';
import { Button } from './ui/button';
import { useChat } from '@/hooks/useChat';

interface MessageDeliveryStatusProps {
  message: LocalChatMessage;
}

export function MessageDeliveryStatus({ message }: MessageDeliveryStatusProps) {
  const { retryMessage } = useChat();

  // Only show delivery status for user messages
  if (message.role !== 'user') {
    return null;
  }

  const handleRetry = () => {
    retryMessage(message.id);
  };

  return (
    <div className="flex items-center gap-1 mt-1 text-xs">
      {message.status === 'sending' && (
        <>
          <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
          <span className="text-muted-foreground">Sending...</span>
        </>
      )}
      {message.status === 'sent' && (
        <>
          <Check className="h-3 w-3 text-green-500" />
          <span className="text-green-500">Sent</span>
        </>
      )}
      {message.status === 'failed' && (
        <>
          <AlertCircle className="h-3 w-3 text-red-500" />
          <span className="text-red-500">Failed</span>
          <Button
            variant="link"
            size="sm"
            className="h-auto p-0 text-xs text-red-500 underline"
            onClick={handleRetry}
          >
            Retry
          </Button>
        </>
      )}
      {message.optimistic && (
        <>
          <Clock className="h-3 w-3 text-yellow-500" />
          <span className="text-yellow-500">Pending</span>
        </>
      )}
      {message.error && (
        <span className="text-red-500 text-xs">({message.error})</span>
      )}
    </div>
  );
}
