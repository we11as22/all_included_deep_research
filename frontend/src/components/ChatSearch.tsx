'use client';

import { Search, X, MessageSquare, Clock } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { searchChatMessages } from '@/lib/api';
import type { ChatMessageSearchResult } from '@/lib/api';

interface ChatSearchProps {
  onChatSelect: (chatId: string, messageId?: string) => void;
  onClose: () => void;
}

export function ChatSearch({ onChatSelect, onClose }: ChatSearchProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<ChatMessageSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    const timeoutId = setTimeout(async () => {
      setIsSearching(true);
      try {
        const searchResults = await searchChatMessages(query, 5);
        setResults(searchResults);
      } catch (error) {
        console.error('Chat message search failed:', error);
        setResults([]);
      } finally {
        setIsSearching(false);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [query]);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }
  };

  const truncateText = (text: string, maxLength: number = 200) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <Card className="relative z-10 w-full max-w-3xl border-border/60 bg-background shadow-xl">
        <div className="flex items-center gap-2 border-b border-border/60 p-4">
          <Search className="h-5 w-5 text-muted-foreground" />
          <Input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search messages using AI hybrid search (semantic + keywords)..."
            className="flex-1 border-0 bg-transparent focus-visible:ring-0"
          />
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="max-h-[32rem] overflow-y-auto p-2">
          {isSearching && (
            <div className="p-4 text-center text-sm text-muted-foreground">
              Searching with AI hybrid search...
            </div>
          )}
          {!isSearching && query && results.length === 0 && (
            <div className="p-4 text-center text-sm text-muted-foreground">
              No messages found
            </div>
          )}
          {!isSearching && results.length > 0 && (
            <div className="space-y-2">
              {results.map((message, index) => (
                <button
                  key={`${message.chat_id}-${message.message_id}`}
                  onClick={() => {
                    // Use message_message_id if available, otherwise message_id
                    const targetMessageId = message.message_message_id || message.message_id?.toString();
                    try {
                      onChatSelect(message.chat_id, targetMessageId);
                      onClose();
                    } catch (error) {
                      console.error('Failed to select chat from search result:', error);
                      // Fallback: try without messageId
                      try {
                        onChatSelect(message.chat_id);
                        onClose();
                      } catch (fallbackError) {
                        console.error('Failed to select chat (fallback):', fallbackError);
                      }
                    }
                  }}
                  className={cn(
                    "w-full rounded-lg p-4 text-left transition-colors",
                    "hover:bg-muted/50",
                    "border border-transparent hover:border-border/60",
                    "group relative"
                  )}
                >
                  {/* Header with chat title and score */}
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <MessageSquare className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                      <div className="font-medium text-sm truncate">{message.chat_title}</div>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <div className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary font-medium">
                        #{index + 1}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {(message.score * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>

                  {/* Message content */}
                  <div className="text-sm text-foreground/90 mb-2 line-clamp-3">
                    {truncateText(message.content)}
                  </div>

                  {/* Footer with role and date */}
                  <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <div className={cn(
                        "px-2 py-0.5 rounded",
                        message.role === 'user' ? "bg-blue-500/10 text-blue-600 dark:text-blue-400" : "bg-green-500/10 text-green-600 dark:text-green-400"
                      )}>
                        {message.role}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {formatDate(message.created_at)}
                    </div>
                    <div className="text-muted-foreground/60">
                      {message.search_mode}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
        {!isSearching && results.length > 0 && (
          <div className="border-t border-border/60 px-4 py-2 text-xs text-muted-foreground">
            Found {results.length} relevant message{results.length !== 1 ? 's' : ''} • Click to open chat
          </div>
        )}
      </Card>
    </div>
  );
}

