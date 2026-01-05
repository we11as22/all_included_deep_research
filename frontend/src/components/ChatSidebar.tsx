'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Plus, Trash2, MessageSquare } from 'lucide-react';
import { listChats, createChat, deleteChat, deleteAllChats, type Chat, type Pagination } from '@/lib/api';
import { cn } from '@/lib/utils';

interface ChatSidebarProps {
  currentChatId: string | null;
  onChatSelect: (chatId: string, messageId?: string) => void;
  onNewChat: () => void;
  refreshTrigger?: number; // Trigger to refresh chat list
}

export function ChatSidebar({ currentChatId, onChatSelect, onNewChat, refreshTrigger }: ChatSidebarProps) {
  const pageSize = 50;
  const [chats, setChats] = useState<Chat[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [pagination, setPagination] = useState<Pagination>({
    limit: pageSize,
    offset: 0,
    total: 0,
    has_more: false,
  });
  const cacheKey = 'chatListCache';

  const loadChats = async (options?: { offset?: number; append?: boolean }): Promise<Chat[]> => {
    const offset = options?.offset ?? 0;
    const append = options?.append ?? false;

    try {
      if (append) {
        setLoadingMore(true);
      } else {
        setLoading(true);
      }
      const response = await listChats({ limit: pageSize, offset });
      setChats((prevChats) => (append ? [...prevChats, ...response.chats] : response.chats));
      setPagination(response.pagination);
      if (!append) {
        try {
          localStorage.setItem(cacheKey, JSON.stringify({
            chats: response.chats,
            pagination: response.pagination,
          }));
        } catch (error) {
          console.warn('Failed to cache chat list:', error);
        }
      }
      return response.chats;
    } catch (error) {
      console.error('Failed to load chats:', error);
      return [];
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  };

  useEffect(() => {
    try {
      const cached = localStorage.getItem(cacheKey);
      if (cached) {
        const parsed = JSON.parse(cached);
        if (Array.isArray(parsed?.chats)) {
          setChats(parsed.chats);
          if (parsed.pagination) {
            setPagination(parsed.pagination);
          }
          setLoading(false);
        }
      }
    } catch (error) {
      console.warn('Failed to read chat list cache:', error);
    }
    loadChats();
  }, []);

  // Refresh chat list when refreshTrigger changes (e.g., after title generation)
  useEffect(() => {
    if (refreshTrigger !== undefined && refreshTrigger > 0) {
      loadChats();
    }
  }, [refreshTrigger]);

  // Listen for chat-not-found events to reload chat list
  useEffect(() => {
    const handleChatNotFound = (event?: CustomEvent) => {
      console.log('Chat not found event received', event?.detail);
      // Reload chat list to remove deleted chat
      loadChats();
    };
    const handleChatRefresh = () => {
      loadChats();
    };
    window.addEventListener('chat-not-found', handleChatNotFound as EventListener);
    window.addEventListener('chat-list-refresh', handleChatRefresh as EventListener);
    return () => {
      window.removeEventListener('chat-not-found', handleChatNotFound as EventListener);
      window.removeEventListener('chat-list-refresh', handleChatRefresh as EventListener);
    };
  }, []);

  const handleDelete = async (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    
    // Confirm deletion
    const chatTitle = chats.find(c => c.id === chatId)?.title || 'this chat';
    if (!confirm(`Are you sure you want to delete "${chatTitle}"? This action cannot be undone.`)) {
      return;
    }
    
    const wasCurrentChat = currentChatId === chatId;
    
    // Optimistically remove chat from UI immediately
    setChats(prevChats => prevChats.filter(c => c.id !== chatId));
    
    // If deleting current chat, switch to new chat immediately
    if (wasCurrentChat) {
      onNewChat();
    }
    
    setLoading(true);
    
    try {
      // Delete chat on server - wait for completion
      await deleteChat(chatId);
      
      // Wait for database transaction to fully commit
      // PostgreSQL may need a moment for the transaction to be visible to other connections
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // Reload chat list to ensure sync with server
      await loadChats();
      
    } catch (error) {
      console.error('Failed to delete chat:', error);
      
      // Reload chat list to restore correct state (chat might still exist)
      await loadChats();
      
      // Show error message
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      alert(`Failed to delete chat: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  const handleNewChat = async () => {
    try {
      const newChat = await createChat('New Chat');
      onChatSelect(newChat.id);
      await loadChats();
    } catch (error) {
      console.error('Failed to create chat:', error);
      onNewChat();
    }
  };

  const handleLoadMore = async () => {
    if (!pagination.has_more || loadingMore) {
      return;
    }
    const nextOffset = pagination.offset + pagination.limit;
    await loadChats({ offset: nextOffset, append: true });
  };

  const handleClearAll = async () => {
    if (!confirm('Are you sure you want to delete ALL chats? This action cannot be undone.')) {
      return;
    }

    try {
      setLoading(true);
      // Immediately clear the chat list in UI to prevent showing stale data
      setChats([]);

      // Clear current chat selection immediately
      if (currentChatId) {
        onNewChat();
      }

      await deleteAllChats();

      setPagination({
        limit: pageSize,
        offset: 0,
        total: 0,
        has_more: false,
      });
    } catch (error) {
      console.error('Failed to clear all chats:', error);
      // Reload chat list anyway to sync with server
      await loadChats();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-64 border-r border-border bg-background/50 p-4 flex flex-col h-full">
      <Button onClick={handleNewChat} className="w-full mb-2" size="sm">
        <Plus className="h-4 w-4 mr-2" />
        New Chat
      </Button>

      {chats.length > 0 && (
        <Button
          onClick={handleClearAll}
          variant="destructive"
          className="w-full mb-4"
          size="sm"
          disabled={loading}
        >
          <Trash2 className="h-4 w-4 mr-2" />
          Clear All
        </Button>
      )}

      <div className="flex-1 overflow-y-auto space-y-1">
        {loading ? (
          <div className="text-sm text-muted-foreground text-center py-4">Loading...</div>
        ) : chats.length === 0 ? (
          <div className="text-sm text-muted-foreground text-center py-4">No chats yet</div>
        ) : (
          <>
            {chats.map((chat) => (
              <Card
                key={chat.id}
                className={cn(
                  'p-3 cursor-pointer hover:bg-accent transition-colors',
                  currentChatId === chat.id && 'bg-accent border-primary'
                )}
                onClick={() => onChatSelect(chat.id)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <MessageSquare className="h-4 w-4 text-muted-foreground shrink-0" />
                      <p className="text-sm font-medium truncate">{chat.title}</p>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {new Date(chat.updated_at).toLocaleDateString()}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 shrink-0"
                    onClick={(e) => handleDelete(chat.id, e)}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
              </Card>
            ))}
            {pagination.has_more && (
              <div className="flex justify-center py-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleLoadMore}
                  disabled={loadingMore}
                >
                  {loadingMore ? 'Loading...' : 'Load more'}
                </Button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
