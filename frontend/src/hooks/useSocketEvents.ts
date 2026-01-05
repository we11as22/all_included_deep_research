import { useEffect } from 'react';
import { useChatStore } from '@/stores/chatStore';
import { socketService } from '@/lib/socket';
import { addMessage as apiAddMessage } from '@/lib/api';

export function useSocketEvents() {
  useEffect(() => {
    const chunkBuffers = new Map<string, string>();
    const flushTimers = new Map<string, number>();

    const resolveMessageId = (data: any) => data?.messageId || data?.message_id;
    const resolveChatId = (data: any) => data?.chatId || data?.chat_id;

    const flushChunkBuffer = (messageId: string) => {
      const buffered = chunkBuffers.get(messageId);
      if (!buffered) return;
      chunkBuffers.delete(messageId);

      const currentMessages = useChatStore.getState().messages;
      const targetMessage = currentMessages.find((msg) => msg.id === messageId);
      if (!targetMessage) return;

      useChatStore.getState().updateMessage(messageId, {
        content: (targetMessage.content || '') + buffered,
      });
    };

    const scheduleFlush = (messageId: string) => {
      if (flushTimers.has(messageId)) return;
      const timer = window.setTimeout(() => {
        flushTimers.delete(messageId);
        flushChunkBuffer(messageId);
      }, 50);
      flushTimers.set(messageId, timer);
    };

    // Handler for stream events
    const handleStreamEvent = (event: CustomEvent) => {
      const { type, data } = event.detail;

      const currentMessages = useChatStore.getState().messages;
      const assistantMessage =
        currentMessages.find((m) => m.role === 'assistant' && m.content === '') ||
        currentMessages[currentMessages.length - 1];

      const messageId = resolveMessageId(data) || assistantMessage?.id;
      const chatId = resolveChatId(data) || useChatStore.getState().currentChatId;
      const activeMessageId = useChatStore.getState().activeMessageId;
      const isActiveMessage = !messageId || messageId === activeMessageId;

      if (!messageId) return;

      if (!assistantMessage && !currentMessages.some((msg) => msg.id === messageId)) {
        if (type !== 'final_report' && type !== 'error' && type !== 'done') {
          return;
        }
      }

      switch (type) {
        case 'init':
          if (isActiveMessage) {
            useChatStore.getState().setIsStreaming(true);
          }
          if (data.sessionId) {
            useChatStore.getState().setCurrentSessionId(data.sessionId);
          }
          break;

        case 'status':
          useChatStore.getState().updateProgress(messageId, {
            status: data.message,
            step: data.step,
          });
          break;

        case 'search_queries':
          useChatStore.getState().updateProgress(messageId, {
            queries: data.queries || [],
          });
          break;

        case 'planning':
          useChatStore.getState().updateProgress(messageId, {
            researchPlan: data.plan,
            topics: data.topics || [],
          });
          break;

        case 'memory_search':
          useChatStore.getState().updateProgress(messageId, {
            memoryContext: data.preview || [],
          });
          break;

        case 'source_found':
          useChatStore.getState().updateProgress(messageId, {
            sources: [
              ...(useChatStore.getState().progressByMessage[messageId]?.sources || []),
              { url: data.url, title: data.title },
            ].slice(-60), // Keep last 60
          });
          break;

        case 'finding':
          useChatStore.getState().updateProgress(messageId, {
            findings: [
              ...(useChatStore.getState().progressByMessage[messageId]?.findings || []),
              { topic: data.topic, summary: data.summary },
            ].slice(-30), // Keep last 30
          });
          break;

        case 'agent_todo':
          if (data.researcher_id) {
            useChatStore.getState().updateProgress(messageId, {
              agentTodos: {
                ...(useChatStore.getState().progressByMessage[messageId]?.agentTodos || {}),
                [data.researcher_id]: data.todos || [],
              },
            });
          }
          break;

        case 'agent_note':
          if (data.researcher_id && data.note) {
            const existingNotes =
              useChatStore.getState().progressByMessage[messageId]?.agentNotes?.[data.researcher_id] || [];
            useChatStore.getState().updateProgress(messageId, {
              agentNotes: {
                ...(useChatStore.getState().progressByMessage[messageId]?.agentNotes || {}),
                [data.researcher_id]: [...existingNotes, data.note].slice(-20),
              },
            });
          }
          break;

        case 'report_chunk':
          if (data.content) {
            const existing = chunkBuffers.get(messageId) || '';
            chunkBuffers.set(messageId, existing + data.content);
            scheduleFlush(messageId);
          }
          break;

        case 'final_report':
          flushChunkBuffer(messageId);
          if (data.report) {
            useChatStore.getState().updateMessage(messageId, {
              content: data.report,
            });

            // Save to database
            if (chatId) {
              apiAddMessage(chatId, 'assistant', data.report, messageId).catch((err) => {
                console.error('Failed to save assistant message:', err);
              });
            }
            if (typeof window !== 'undefined') {
              window.dispatchEvent(new CustomEvent('chat-list-refresh'));
            }

            useChatStore.getState().updateProgress(messageId, {
              isComplete: true,
            });
          }
          break;

        case 'error':
          flushChunkBuffer(messageId);
          useChatStore.getState().setError(data.error || 'An error occurred');
          useChatStore.getState().updateMessage(messageId, {
            content: `Error: ${data.error}`,
            status: 'failed',
          });
          if (isActiveMessage) {
            useChatStore.getState().setIsStreaming(false);
            useChatStore.getState().setActiveMessageId(null);
          }
          break;

        case 'done':
          flushChunkBuffer(messageId);
          if (isActiveMessage) {
            useChatStore.getState().setIsStreaming(false);
            useChatStore.getState().setActiveMessageId(null);
          }
          useChatStore.getState().updateProgress(messageId, {
            isComplete: true,
          });
          break;

        default:
          console.log('Unhandled stream event:', type, data);
      }
    };

    // Listen to socket stream events
    if (typeof window !== 'undefined') {
      window.addEventListener('socket-stream-event', handleStreamEvent as EventListener);
    }

    return () => {
      if (typeof window !== 'undefined') {
        window.removeEventListener('socket-stream-event', handleStreamEvent as EventListener);
      }
      flushTimers.forEach((timer) => window.clearTimeout(timer));
      flushTimers.clear();
      chunkBuffers.clear();
    };
  }, []);
}
