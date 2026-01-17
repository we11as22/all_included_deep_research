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
          <div className="prose prose-sm dark:prose-invert max-w-none
                          prose-headings:font-semibold prose-headings:text-foreground prose-headings:mt-4 prose-headings:mb-2
                          prose-h1:text-2xl prose-h1:font-bold prose-h1:mt-6 prose-h1:mb-3
                          prose-h2:text-xl prose-h2:font-semibold prose-h2:mt-5 prose-h2:mb-2
                          prose-h3:text-lg prose-h3:font-semibold prose-h3:mt-4 prose-h3:mb-2
                          prose-p:text-foreground prose-p:leading-relaxed prose-p:my-3 prose-p:whitespace-pre-wrap
                          prose-strong:text-foreground prose-strong:font-semibold
                          prose-em:text-foreground prose-em:italic
                          prose-ul:text-foreground prose-ul:my-3 prose-ul:pl-6
                          prose-ol:text-foreground prose-ol:my-3 prose-ol:pl-6
                          prose-li:text-foreground prose-li:my-1
                          prose-code:text-foreground prose-code:bg-muted prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:font-mono
                          prose-pre:bg-muted prose-pre:text-foreground prose-pre:p-4 prose-pre:rounded-lg prose-pre:overflow-x-auto prose-pre:whitespace-pre-wrap
                          prose-blockquote:border-l-4 prose-blockquote:border-primary prose-blockquote:pl-4 prose-blockquote:italic prose-blockquote:my-4 prose-blockquote:text-foreground/80
                          prose-a:text-blue-600 dark:prose-a:text-blue-400 prose-a:no-underline hover:prose-a:underline prose-a:font-medium
                          prose-hr:border-gray-300 dark:prose-hr:border-gray-600 prose-hr:my-6
                          prose-table:w-full prose-table:my-4
                          prose-th:border prose-th:border-border prose-th:px-4 prose-th:py-2 prose-th:bg-muted prose-th:font-semibold prose-th:text-foreground
                          prose-td:border prose-td:border-border prose-td:px-4 prose-td:py-2 prose-td:text-foreground
                          break-words">
            <Markdown
              options={{
                forceBlock: true,
                forceInline: false,
                wrapper: React.Fragment,
                overrides: {
                  a: {
                    component: ({ href, children, ...props }) => (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 dark:text-blue-400 hover:underline font-medium"
                        {...props}
                      >
                        {children}
                      </a>
                    ),
                  },
                  h1: {
                    component: ({ children, ...props }) => (
                      <h1 className="text-2xl font-bold mt-6 mb-3 text-foreground" {...props}>
                        {children}
                      </h1>
                    ),
                  },
                  h2: {
                    component: ({ children, ...props }) => (
                      <h2 className="text-xl font-semibold mt-5 mb-2 text-foreground" {...props}>
                        {children}
                      </h2>
                    ),
                  },
                  h3: {
                    component: ({ children, ...props }) => (
                      <h3 className="text-lg font-semibold mt-4 mb-2 text-foreground" {...props}>
                        {children}
                      </h3>
                    ),
                  },
                  h4: {
                    component: ({ children, ...props }) => (
                      <h4 className="text-base font-semibold mt-3 mb-1 text-foreground" {...props}>
                        {children}
                      </h4>
                    ),
                  },
                  p: {
                    component: ({ children, ...props }) => (
                      <p className="mb-3 leading-relaxed text-foreground whitespace-pre-wrap" {...props}>
                        {children}
                      </p>
                    ),
                  },
                  strong: {
                    component: ({ children, ...props }) => (
                      <strong className="font-semibold text-foreground" {...props}>
                        {children}
                      </strong>
                    ),
                  },
                  em: {
                    component: ({ children, ...props }) => (
                      <em className="italic text-foreground" {...props}>
                        {children}
                      </em>
                    ),
                  },
                  ul: {
                    component: ({ children, ...props }) => (
                      <ul className="list-disc list-inside my-3 pl-6 text-foreground space-y-1" {...props}>
                        {children}
                      </ul>
                    ),
                  },
                  ol: {
                    component: ({ children, ...props }) => (
                      <ol className="list-decimal list-inside my-3 pl-6 text-foreground space-y-1" {...props}>
                        {children}
                      </ol>
                    ),
                  },
                  li: {
                    component: ({ children, ...props }) => (
                      <li className="text-foreground my-1" {...props}>
                        {children}
                      </li>
                    ),
                  },
                  code: {
                    component: ({ children, ...props }) => (
                      <code className="bg-muted px-1.5 py-0.5 rounded text-sm font-mono text-foreground" {...props}>
                        {children}
                      </code>
                    ),
                  },
                  pre: {
                    component: ({ children, ...props }) => (
                      <pre className="bg-muted p-4 rounded-lg overflow-x-auto my-4 text-foreground" {...props}>
                        {children}
                      </pre>
                    ),
                  },
                  blockquote: {
                    component: ({ children, ...props }) => (
                      <blockquote className="border-l-4 border-primary pl-4 italic my-4 text-foreground/80" {...props}>
                        {children}
                      </blockquote>
                    ),
                  },
                  hr: {
                    component: ({ ...props }) => (
                      <hr className="border-gray-300 dark:border-gray-600 my-6" {...props} />
                    ),
                  },
                },
              }}
            >
              {message.content || ''}
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
