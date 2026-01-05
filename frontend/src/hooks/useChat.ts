import { useChatStore } from '@/stores/chatStore';
import { socketService } from '@/lib/socket';
import { createChatWithMessage, addMessage as apiAddMessage } from '@/lib/api';

const makeId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

export function useChat() {
  const store = useChatStore();

  const sendMessage = async (content: string) => {
    if (!content.trim()) {
      throw new Error('Message content cannot be empty');
    }

    // Add optimistic user message
    const userMessageId = makeId();
    const tempId = store.addOptimisticMessage({
      role: 'user',
      content,
    });

    try {
      let chatId = store.currentChatId;

      // If no chat exists, create atomically with message
      if (!chatId) {
        const { chat, message } = await createChatWithMessage(
          content.slice(0, 80), // title
          content,
          tempId
        );
        chatId = chat.id;
        store.setCurrentChatId(chatId);
        store.confirmMessage(tempId, message.message_id);
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('chat-list-refresh'));
        }
      } else {
        // Save user message to existing chat
        const saved = await apiAddMessage(chatId, 'user', content, tempId);
        store.confirmMessage(tempId, saved.message_id);
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('chat-list-refresh'));
        }
      }

      // Add placeholder for assistant message
      const assistantMessageId = makeId();
      store.addMessage({
        id: assistantMessageId,
        role: 'assistant',
        content: '',
      });
      store.setActiveMessageId(assistantMessageId);
      store.setProgressPanelMessageId(assistantMessageId);

      // Initialize progress state
      store.updateProgress(assistantMessageId, {
        status: 'Initializing...',
        step: 'init',
        queries: [],
        sources: [],
        findings: [],
        memoryContext: [],
        topics: [],
        researchPlan: null,
        agentTodos: {},
        agentNotes: {},
        isComplete: false,
        mode: store.mode,
      });

      // Send via Socket.IO
      await socketService.sendMessage({
        id: assistantMessageId,
        chatId,
        content,
        mode: store.mode,
      });
    } catch (error: any) {
      console.error('Failed to send message:', error);
      store.rejectMessage(tempId, error.message || 'Failed to send message');
      throw error;
    }
  };

  const retryMessage = async (messageId: string) => {
    const message = store.messages.find((m) => m.id === messageId);
    if (!message || message.status !== 'failed') {
      return;
    }

    // Reset status
    store.updateMessage(messageId, {
      status: 'sending',
      error: undefined,
    });

    try {
      const chatId = store.currentChatId;
      if (!chatId) {
        throw new Error('No chat ID available');
      }

      await socketService.sendMessage({
        id: messageId,
        chatId,
        content: message.content,
        mode: store.mode,
      });

      store.updateMessage(messageId, {
        status: 'sent',
      });
    } catch (error: any) {
      console.error('Failed to retry message:', error);
      store.updateMessage(messageId, {
        status: 'failed',
        error: error.message || 'Failed to retry message',
      });
    }
  };

  const cancelStream = () => {
    const sessionId = store.currentSessionId;
    if (sessionId) {
      socketService.cancelStream(sessionId);
      store.setIsStreaming(false);
      store.setActiveMessageId(null);
    }
  };

  return {
    sendMessage,
    retryMessage,
    cancelStream,
  };
}
