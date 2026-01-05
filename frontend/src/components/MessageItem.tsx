'use client';

import React from 'react';
import { LocalChatMessage } from '@/stores/chatStore';
import { cn } from '@/lib/utils';
import Markdown from 'markdown-to-jsx';
import { MessageDeliveryStatus } from './MessageDeliveryStatus';

interface MessageItemProps {
  message: LocalChatMessage;
}

export const MessageItem = React.memo(
  ({ message }: MessageItemProps) => {
    return (
      <div
        className={cn(
          'p-4 rounded-lg mb-4 animate-fade-rise',
          message.role === 'user' && 'bg-blue-50 dark:bg-blue-950 ml-auto max-w-[80%]',
          message.role === 'assistant' && 'bg-white dark:bg-gray-800 mr-auto',
          message.role === 'system' && 'bg-muted/60 text-muted-foreground mx-auto max-w-[85%]'
        )}
      >
        {message.content && (
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <Markdown
              options={{
                overrides: {
                  a: {
                    props: {
                      target: '_blank',
                      rel: 'noopener noreferrer',
                      className: 'text-blue-600 hover:underline',
                    },
                  },
                },
              }}
            >
              {message.content}
            </Markdown>
          </div>
        )}
        {!message.content && message.role === 'assistant' && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <div className="animate-pulse">Thinking...</div>
          </div>
        )}
        <MessageDeliveryStatus message={message} />
      </div>
    );
  },
  (prev, next) => {
    // Only re-render if content or status changed
    return (
      prev.message.id === next.message.id &&
      prev.message.content === next.message.content &&
      prev.message.status === next.message.status &&
      prev.message.error === next.message.error
    );
  }
);

MessageItem.displayName = 'MessageItem';
