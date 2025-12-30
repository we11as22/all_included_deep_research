'use client';

import { Search, X } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { searchChats } from '@/lib/api';
import type { Chat } from '@/lib/api';

interface ChatSearchProps {
  onChatSelect: (chatId: string) => void;
  onClose: () => void;
}

export function ChatSearch({ onChatSelect, onClose }: ChatSearchProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Chat[]>([]);
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
        const searchResults = await searchChats(query);
        setResults(searchResults);
      } catch (error) {
        console.error('Chat search failed:', error);
        setResults([]);
      } finally {
        setIsSearching(false);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [query]);

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <Card className="relative z-10 w-full max-w-2xl border-border/60 bg-background shadow-xl">
        <div className="flex items-center gap-2 border-b border-border/60 p-4">
          <Search className="h-5 w-5 text-muted-foreground" />
          <Input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search chats by message content..."
            className="flex-1 border-0 bg-transparent focus-visible:ring-0"
          />
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="max-h-96 overflow-y-auto p-2">
          {isSearching && (
            <div className="p-4 text-center text-sm text-muted-foreground">Searching...</div>
          )}
          {!isSearching && query && results.length === 0 && (
            <div className="p-4 text-center text-sm text-muted-foreground">No chats found</div>
          )}
          {!isSearching && results.length > 0 && (
            <div className="space-y-1">
              {results.map((chat) => (
                <button
                  key={chat.id}
                  onClick={() => {
                    onChatSelect(chat.id);
                    onClose();
                  }}
                  className={cn(
                    "w-full rounded-lg p-3 text-left transition-colors",
                    "hover:bg-muted/50",
                    "border border-transparent hover:border-border/60"
                  )}
                >
                  <div className="font-medium text-sm">{chat.title}</div>
                  {chat.metadata?.preview && (
                    <div className="mt-1 text-xs text-muted-foreground line-clamp-2">
                      {chat.metadata.preview}
                    </div>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}

