'use client';

import { useEffect, useState } from 'react';
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
  const connectionStatus = useChatStore((state) => state.connectionStatus);
  const [isInitializing, setIsInitializing] = useState(true);

  // Initialize Socket.IO and offline queue on mount
  useEffect(() => {
    let cancelled = false;
    let initTimeout: NodeJS.Timeout | null = null;

    const finishInitialization = () => {
      if (cancelled) return;
      setIsInitializing(false);
      if (initTimeout) {
        clearTimeout(initTimeout);
        initTimeout = null;
      }
    };

    const init = async () => {
      try {
        // Set connecting status
        useChatStore.getState().setConnectionStatus('connecting');
        
        // Health check with timeout
        const healthCheckPromise = (async () => {
          try {
            const { checkBackendHealth } = await import('@/lib/api');
            return await checkBackendHealth(3, 200).catch(() => false);
          } catch (error) {
            return false;
          }
        })();
        
        const timeoutPromise = new Promise<boolean>((resolve) => {
          setTimeout(() => resolve(false), 3000);
        });
        
        await Promise.race([healthCheckPromise, timeoutPromise]);
        
        if (cancelled) return;

        // Initialize offline queue (non-blocking)
        offlineQueue.init().catch(() => {
          // Ignore errors
        });

        // Load queued messages (non-blocking)
        offlineQueue.getAll().then((messages) => {
          if (cancelled) return;
          messages.forEach((msg) => {
            useChatStore.getState().queueMessage(msg);
          });
        }).catch(() => {
          // Ignore errors
        });

        // Connect socket
        if (!socketService.isConnected()) {
          socketService.connect();
        }
        
        // Listen to connection status to finish initialization
        let unsubscribe: (() => void) | null = null;
        let hasFinished = false;
        
        unsubscribe = useChatStore.subscribe((state) => {
          if (state.connectionStatus === 'online' && !hasFinished) {
            hasFinished = true;
            finishInitialization();
            if (unsubscribe) {
              unsubscribe();
              unsubscribe = null;
            }
          }
        });
        
        // Timeout fallback - always finish initialization after 3 seconds
        initTimeout = setTimeout(() => {
          if (!cancelled && !hasFinished) {
            hasFinished = true;
            finishInitialization();
            if (unsubscribe) {
              unsubscribe();
              unsubscribe = null;
            }
          }
        }, 3000);
        
      } catch (error) {
        console.error('Failed to initialize:', error);
        // Still try to connect
        if (!socketService.isConnected()) {
          socketService.connect();
        }
        finishInitialization();
      }
    };

    init();

    // Cleanup on unmount
    return () => {
      cancelled = true;
      if (initTimeout) {
        clearTimeout(initTimeout);
      }
      // Don't disconnect socket on unmount - let it stay connected
    };
  }, []);

  // Subscribe to socket events
  useSocketEvents();

  // Restore chat from localStorage on mount
  // CRITICAL: Only load once, with timeout protection
  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (isInitializing) return; // Don't load chat while initializing

    const savedChatId = localStorage.getItem('currentChatId');
    if (!savedChatId) return;

    // CRITICAL: Only load if chat ID changed
    if (savedChatId !== currentChatId) {
      // Chat ID changed - load new chat with timeout
      const timeoutPromise = new Promise<void>((resolve) => {
        setTimeout(() => {
          console.warn('Chat load timeout - skipping');
          resolve();
        }, 5000); // 5 second timeout
      });
      
      Promise.race([
        loadChat(savedChatId),
        timeoutPromise
      ]).catch((error) => {
        console.error('Failed to load saved chat:', error);
        // Clear invalid chat ID
        localStorage.removeItem('currentChatId');
        useChatStore.getState().setCurrentChatId(null);
      });
    }
    // CRITICAL: Don't reload if we already have messages - prevents infinite loops
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentChatId, isInitializing]);

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

  // Show loading screen while initializing
  if (isInitializing) {
    return (
      <ErrorBoundary>
        <div className="flex h-screen bg-background items-center justify-center">
          <div className="flex flex-col items-center gap-4">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
            <p className="text-muted-foreground">Connecting to server...</p>
          </div>
        </div>
      </ErrorBoundary>
    );
  }

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
