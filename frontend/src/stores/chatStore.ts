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
  progressByChat: Record<string, Record<string, ProgressState>>;
  progressPanelByChat: Record<string, string | null>;
  activeMessageByChat: Record<string, string | null>;

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
  updateProgress: (messageId: string, progress: Partial<ProgressState>, chatId?: string | null) => void;
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
      progressByChat: {},
      progressPanelByChat: {},
      activeMessageByChat: {},
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
      updateProgress: (messageId, progress, chatId) =>
        set((state) => {
          const targetChatId = chatId || state.currentChatId;
          const existingMessageProgress = state.progressByMessage[messageId] || {};
          // Ensure all arrays are initialized to prevent undefined.length errors
          // Merge existing progress with new progress, ensuring arrays are never undefined
          const nextMessageProgress: ProgressState = {
            status: progress.status !== undefined ? progress.status : (existingMessageProgress.status || ''),
            step: progress.step !== undefined ? progress.step : (existingMessageProgress.step || ''),
            memoryContext: progress.memoryContext !== undefined ? progress.memoryContext : (existingMessageProgress.memoryContext || []),
            researchPlan: progress.researchPlan !== undefined ? progress.researchPlan : (existingMessageProgress.researchPlan ?? null),
            topics: progress.topics !== undefined ? progress.topics : (existingMessageProgress.topics || []),
            findings: progress.findings !== undefined ? progress.findings : (existingMessageProgress.findings || []),
            sources: progress.sources !== undefined ? progress.sources : (existingMessageProgress.sources || []),
            agentTodos: progress.agentTodos !== undefined ? progress.agentTodos : (existingMessageProgress.agentTodos || {}),
            agentNotes: progress.agentNotes !== undefined ? progress.agentNotes : (existingMessageProgress.agentNotes || {}),
            isComplete: progress.isComplete !== undefined ? progress.isComplete : (existingMessageProgress.isComplete || false),
            mode: progress.mode !== undefined ? progress.mode : existingMessageProgress.mode,
            sessionId: progress.sessionId !== undefined ? progress.sessionId : existingMessageProgress.sessionId,
            error: progress.error !== undefined ? progress.error : existingMessageProgress.error,
            queries: progress.queries !== undefined ? progress.queries : (existingMessageProgress.queries || []),
          };

          if (!targetChatId) {
            return {
              progressByMessage: {
                ...state.progressByMessage,
                [messageId]: nextMessageProgress,
              },
            };
          }

          const existingChatProgress = state.progressByChat[targetChatId] || {};
          return {
            progressByMessage: {
              ...state.progressByMessage,
              [messageId]: nextMessageProgress,
            },
            progressByChat: {
              ...state.progressByChat,
              [targetChatId]: {
                ...existingChatProgress,
                [messageId]: nextMessageProgress,
              },
            },
          };
        }),
      setIsStreaming: (streaming) => set({ isStreaming: streaming }),
      setActiveMessageId: (messageId) =>
        set((state) => {
          const chatId = state.currentChatId;
          if (!chatId) {
            return { activeMessageId: messageId };
          }
          return {
            activeMessageId: messageId,
            activeMessageByChat: {
              ...state.activeMessageByChat,
              [chatId]: messageId,
            },
          };
        }),
      setProgressPanelMessageId: (messageId) =>
        set((state) => {
          const chatId = state.currentChatId;
          if (!chatId) {
            return { progressPanelMessageId: messageId };
          }
          return {
            progressPanelMessageId: messageId,
            progressPanelByChat: {
              ...state.progressPanelByChat,
              [chatId]: messageId,
            },
          };
        }),
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

            get().setCurrentChatId(chatId);
            set((state) => ({
              messages: loadedMessages,
              progressByMessage: state.progressByChat?.[chatId] || {},
              activeMessageId: state.activeMessageByChat?.[chatId] || null,
              progressPanelMessageId: state.progressPanelByChat?.[chatId] || null,
              isStreaming: false,
              error: null,
            }));
          }
        } catch (error: any) {
          console.error('Failed to load chat:', error);
          set({ error: error.message || 'Failed to load chat' });
        }
      },

      clearCurrentChat: () => {
        get().setCurrentChatId(null);
        set({
          messages: [],
          progressByMessage: {},
          isStreaming: false,
          activeMessageId: null,
          progressPanelMessageId: null,
          error: null,
          input: '',
        });
      },

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
        progressByChat: state.progressByChat,
        progressPanelByChat: state.progressPanelByChat,
        activeMessageByChat: state.activeMessageByChat,
      }),
    }
  )
);
