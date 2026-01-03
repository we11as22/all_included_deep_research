'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Plus, Trash2, MessageSquare } from 'lucide-react';
import { listChats, createChat, deleteChat, type Chat } from '@/lib/api';
import { cn } from '@/lib/utils';

interface ChatSidebarProps {
  currentChatId: string | null;
  onChatSelect: (chatId: string, messageId?: string) => void;
  onNewChat: () => void;
  refreshTrigger?: number; // Trigger to refresh chat list
}

export function ChatSidebar({ currentChatId, onChatSelect, onNewChat, refreshTrigger }: ChatSidebarProps) {
  const [chats, setChats] = useState<Chat[]>([]);
  const [loading, setLoading] = useState(true);

  const loadChats = async () => {
    try {
      setLoading(true);
      const chatList = await listChats();
      setChats(chatList);
    } catch (error) {
      console.error('Failed to load chats:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
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
    const handleChatNotFound = () => {
      loadChats();
    };
    window.addEventListener('chat-not-found', handleChatNotFound);
    return () => {
      window.removeEventListener('chat-not-found', handleChatNotFound);
    };
  }, []);

  const handleDelete = async (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    
    // Confirm deletion
    const chatTitle = chats.find(c => c.id === chatId)?.title || 'this chat';
    if (!confirm(`Are you sure you want to delete "${chatTitle}"? This action cannot be undone.`)) {
      return;
    }
    
    try {
      await deleteChat(chatId);
      // Reload chat list to ensure sync with server
      await loadChats();
      if (currentChatId === chatId) {
        onNewChat();
      }
    } catch (error) {
      console.error('Failed to delete chat:', error);
      // Reload chat list anyway to sync with server state
      await loadChats();
      alert(`Failed to delete chat: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const handleNewChat = async () => {
    try {
      const newChat = await createChat('New Chat');
      setChats([newChat, ...chats]);
      onChatSelect(newChat.id);
    } catch (error) {
      console.error('Failed to create chat:', error);
      onNewChat();
    }
  };

  const handleClearAll = async () => {
    if (!confirm('Are you sure you want to delete ALL chats? This action cannot be undone.')) {
      return;
    }

    try {
      setLoading(true);
      // Delete all chats - use allSettled to continue even if some fail
      const results = await Promise.allSettled(
        chats.map(chat => deleteChat(chat.id))
      );
      
      // Count successful deletions
      const successful = results.filter(r => r.status === 'fulfilled').length;
      const failed = results.filter(r => r.status === 'rejected').length;
      
      if (failed > 0) {
        console.warn(`Failed to delete ${failed} chat(s)`, results.filter(r => r.status === 'rejected'));
      }
      
      // Reload chat list to get current state from server
      await loadChats();
      
      // If all chats were deleted, switch to new chat
      if (successful === chats.length) {
        onNewChat();
      }
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
          chats.map((chat) => (
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
          ))
        )}
      </div>
    </div>
  );
}

