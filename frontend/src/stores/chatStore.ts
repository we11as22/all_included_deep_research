import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { ProgressState } from '@/components/ChatProgressPanel';
import { ChatMode } from '@/components/ModeSelectorDropdown';
import { getChat, addMessage as apiAddMessage, createChatWithMessage } from '@/lib/api';

export type ConnectionStatus = 'online' | 'offline' | 'connecting' | 'reconnecting';
export type MessageStatus = 'sending' | 'sent' | 'failed';

export interface LocalChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  optimistic?: boolean;
  status?: MessageStatus;
  error?: string;
}

export interface QueuedMessage {
  id: string;
  chatId: string;
  content: string;
  mode: ChatMode;
  timestamp: number;
}

interface ChatStore {
  // Connection state
  connectionStatus: ConnectionStatus;
  socketConnected: boolean;

  // Current session
  currentChatId: string | null;
  currentSessionId: string | null;

  // Messages
  messages: LocalChatMessage[];
  messageQueue: QueuedMessage[];

  // Progress tracking
  progressByMessage: Record<string, ProgressState>;

  // UI state
  isStreaming: boolean;
  activeMessageId: string | null;
  progressPanelMessageId: string | null;
  error: string | null;
  mode: ChatMode;
  input: string;

  // Actions
  setConnectionStatus: (status: ConnectionStatus) => void;
  setSocketConnected: (connected: boolean) => void;
  setCurrentChatId: (chatId: string | null) => void;
  setCurrentSessionId: (sessionId: string | null) => void;
  setMessages: (messages: LocalChatMessage[]) => void;
  addMessage: (message: LocalChatMessage) => void;
  updateMessage: (id: string, updates: Partial<LocalChatMessage>) => void;
  queueMessage: (message: QueuedMessage) => void;
  dequeueMessage: (id: string) => void;
  updateProgress: (messageId: string, progress: Partial<ProgressState>) => void;
  setIsStreaming: (streaming: boolean) => void;
  setActiveMessageId: (messageId: string | null) => void;
  setProgressPanelMessageId: (messageId: string | null) => void;
  setError: (error: string | null) => void;
  setMode: (mode: ChatMode) => void;
  setInput: (input: string) => void;

  // Optimistic updates
  addOptimisticMessage: (message: Omit<LocalChatMessage, 'id' | 'optimistic' | 'status'>) => string;
  confirmMessage: (tempId: string, serverId: string) => void;
  rejectMessage: (tempId: string, error: string) => void;

  // Chat operations
  loadChat: (chatId: string) => Promise<void>;
  clearCurrentChat: () => void;
  clearMessages: () => void;
}

const makeId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      // Initial state
      connectionStatus: 'offline',
      socketConnected: false,
      currentChatId: null,
      currentSessionId: null,
      messages: [],
      messageQueue: [],
      progressByMessage: {},
      isStreaming: false,
      activeMessageId: null,
      progressPanelMessageId: null,
      error: null,
      mode: (process.env.NEXT_PUBLIC_DEFAULT_MODE as ChatMode) || 'search',
      input: '',

      // Actions
      setConnectionStatus: (status) => set({ connectionStatus: status }),
      setSocketConnected: (connected) => set({ socketConnected: connected }),
      setCurrentChatId: (chatId) => {
        set({ currentChatId: chatId });
        if (typeof window !== 'undefined') {
          if (chatId) {
            localStorage.setItem('currentChatId', chatId);
          } else {
            localStorage.removeItem('currentChatId');
          }
        }
      },
      setCurrentSessionId: (sessionId) => {
        set({ currentSessionId: sessionId });
        if (typeof window !== 'undefined') {
          if (sessionId) {
            localStorage.setItem('currentSessionId', sessionId);
          } else {
            localStorage.removeItem('currentSessionId');
          }
        }
      },
      setMessages: (messages) => set({ messages }),
      addMessage: (message) =>
        set((state) => ({
          messages: [...state.messages, message],
        })),
      updateMessage: (id, updates) =>
        set((state) => ({
          messages: state.messages.map((msg) =>
            msg.id === id ? { ...msg, ...updates } : msg
          ),
        })),
      queueMessage: (message) =>
        set((state) => ({
          messageQueue: [...state.messageQueue, message],
        })),
      dequeueMessage: (id) =>
        set((state) => ({
          messageQueue: state.messageQueue.filter((msg) => msg.id !== id),
        })),
      updateProgress: (messageId, progress) =>
        set((state) => ({
          progressByMessage: {
            ...state.progressByMessage,
            [messageId]: {
              ...(state.progressByMessage[messageId] || {}),
              ...progress,
            } as ProgressState,
          },
        })),
      setIsStreaming: (streaming) => set({ isStreaming: streaming }),
      setActiveMessageId: (messageId) => set({ activeMessageId: messageId }),
      setProgressPanelMessageId: (messageId) => set({ progressPanelMessageId: messageId }),
      setError: (error) => set({ error }),
      setMode: (mode) => set({ mode }),
      setInput: (input) => set({ input }),

      // Optimistic updates
      addOptimisticMessage: (message) => {
        const tempId = `temp-${makeId()}`;
        set((state) => ({
          messages: [
            ...state.messages,
            {
              ...message,
              id: tempId,
              optimistic: true,
              status: 'sending' as MessageStatus,
            },
          ],
        }));
        return tempId;
      },

      confirmMessage: (tempId, serverId) =>
        set((state) => ({
          messages: state.messages.map((msg) =>
            msg.id === tempId
              ? { ...msg, id: serverId, optimistic: false, status: 'sent' as MessageStatus }
              : msg
          ),
        })),

      rejectMessage: (tempId, error) =>
        set((state) => ({
          messages: state.messages.map((msg) =>
            msg.id === tempId
              ? { ...msg, status: 'failed' as MessageStatus, error }
              : msg
          ),
        })),

      // Chat operations
      loadChat: async (chatId) => {
        try {
          const chatData = await getChat(chatId);
          if (chatData && chatData.chat && chatData.messages) {
            const sortedMessages = [...chatData.messages].sort((a, b) => {
              const aTime = a.created_at ? new Date(a.created_at).getTime() : 0;
              const bTime = b.created_at ? new Date(b.created_at).getTime() : 0;
              return aTime - bTime;
            });

            const loadedMessages: LocalChatMessage[] = sortedMessages.map((msg) => ({
              id: msg.message_id || makeId(),
              role: msg.role as 'user' | 'assistant' | 'system',
              content: msg.content || '',
              status: 'sent' as MessageStatus,
            }));

            set({
              currentChatId: chatId,
              messages: loadedMessages,
              activeMessageId: null,
              progressPanelMessageId: null,
              isStreaming: false,
              error: null,
            });
          }
        } catch (error: any) {
          console.error('Failed to load chat:', error);
          set({ error: error.message || 'Failed to load chat' });
        }
      },

      clearCurrentChat: () =>
        set({
          currentChatId: null,
          messages: [],
          progressByMessage: {},
          isStreaming: false,
          activeMessageId: null,
          progressPanelMessageId: null,
          error: null,
          input: '',
        }),

      clearMessages: () =>
        set({
          messages: [],
          progressByMessage: {},
          activeMessageId: null,
          progressPanelMessageId: null,
        }),
    }),
    {
      name: 'chat-store',
      partialize: (state) => ({
        currentChatId: state.currentChatId,
        currentSessionId: state.currentSessionId,
        mode: state.mode,
      }),
    }
  )
);
