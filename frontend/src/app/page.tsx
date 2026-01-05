'use client';

import { useEffect } from 'react';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { ChatSidebar } from '@/components/ChatSidebar';
import { ChatContainer } from '@/components/ChatContainer';
import { ConnectionStatus } from '@/components/ConnectionStatus';
import { ThemeToggle } from '@/components/ThemeToggle';
import { ChatSearch } from '@/components/ChatSearch';
import { Button } from '@/components/ui/button';
import { useChatStore } from '@/stores/chatStore';
import { useUIStore } from '@/stores/uiStore';
import { socketService } from '@/lib/socket';
import { offlineQueue } from '@/lib/offlineQueue';
import { useSocketEvents } from '@/hooks/useSocketEvents';
import { Search } from 'lucide-react';

export default function HomePage() {
  const currentChatId = useChatStore((state) => state.currentChatId);
  const messages = useChatStore((state) => state.messages);
  const loadChat = useChatStore((state) => state.loadChat);
  const clearCurrentChat = useChatStore((state) => state.clearCurrentChat);
  const showChatSearch = useUIStore((state) => state.showChatSearch);
  const setShowChatSearch = useUIStore((state) => state.setShowChatSearch);

  // Initialize Socket.IO and offline queue on mount
  useEffect(() => {
    let cancelled = false;

    const init = async () => {
      try {
        await offlineQueue.init();
      } catch (error) {
        console.error('Failed to initialize offline queue:', error);
        return;
      }

      if (cancelled) return;

      offlineQueue.getAll().then((messages) => {
        messages.forEach((msg) => {
          useChatStore.getState().queueMessage(msg);
        });
      });

      socketService.connect();
    };

    const idleCallback = typeof window !== 'undefined' ? (window as any).requestIdleCallback : null;
    const cancelIdle = typeof window !== 'undefined' ? (window as any).cancelIdleCallback : null;

    const schedule = idleCallback
      ? idleCallback(() => {
          init();
        }, { timeout: 1000 })
      : window.setTimeout(() => init(), 0);

    // Cleanup on unmount
    return () => {
      cancelled = true;
      if (cancelIdle) {
        cancelIdle(schedule as number);
      } else {
        window.clearTimeout(schedule as number);
      }
      socketService.disconnect();
    };
  }, []);

  // Subscribe to socket events
  useSocketEvents();

  // Restore chat from localStorage on mount
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const savedChatId = localStorage.getItem('currentChatId');
    if (!savedChatId) return;

    if (savedChatId !== currentChatId || messages.length === 0) {
      loadChat(savedChatId).catch((error) => {
        console.error('Failed to load saved chat:', error);
        // Clear invalid chat ID
        localStorage.removeItem('currentChatId');
        useChatStore.getState().setCurrentChatId(null);
      });
    }
  }, [currentChatId, messages.length, loadChat]);

  const handleChatSelect = async (chatId: string) => {
    try {
      await loadChat(chatId);
    } catch (error) {
      console.error('Failed to load chat:', error);
    }
  };

  const handleNewChat = () => {
    clearCurrentChat();
  };

  return (
    <ErrorBoundary>
      <div className="flex h-screen bg-background overflow-hidden">
        {/* Chat Sidebar */}
        <ChatSidebar
          currentChatId={currentChatId}
          onChatSelect={handleChatSelect}
          onNewChat={handleNewChat}
        />

        {/* Main Chat Area */}
        <main className="flex-1 relative flex flex-col min-h-0 overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between border-b px-4 py-2">
            <h1 className="text-lg font-semibold">Deep Research Chat</h1>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setShowChatSearch(true)}
                aria-label="Search messages"
              >
                <Search className="h-4 w-4" />
              </Button>
              <ThemeToggle />
            </div>
          </div>

          {/* Connection Status Indicator */}
          <ConnectionStatus />

          {/* Chat Container */}
          <ChatContainer />

          {/* Chat Search Modal */}
          {showChatSearch && (
            <div className="absolute inset-0 z-40 bg-background/80 backdrop-blur-sm">
              <ChatSearch
                onClose={() => setShowChatSearch(false)}
                onChatSelect={handleChatSelect}
              />
            </div>
          )}
        </main>
      </div>
    </ErrorBoundary>
  );
}
