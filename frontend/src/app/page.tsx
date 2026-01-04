'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { ModeSelectorDropdown, ChatMode } from '@/components/ModeSelectorDropdown';
import { streamChatProgress } from '@/lib/api';
import { ChatProgressPanel, ProgressState } from '@/components/ChatProgressPanel';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { ArrowUpRight, Brain, Download, Loader2, Search, X } from 'lucide-react';
import { cancelChat, getChat, addMessage, createChat, downloadPDF, type ChatMessage as DBChatMessage } from '@/lib/api';
import { cn } from '@/lib/utils';
import Markdown from 'markdown-to-jsx';
import { ChatSidebar } from '@/components/ChatSidebar';
import { ThemeToggle } from '@/components/ThemeToggle';
import { ChatSearch } from '@/components/ChatSearch';

type LocalChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
};

const makeId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
const formatModeLabel = (mode: ChatMode) => {
  if (mode === 'chat') return 'chat';
  if (mode === 'search') return 'web search';
  if (mode === 'deep_search') return 'deep search';
  if (mode === 'deep_research') return 'deep research';
  return 'search';
};
const suggestions = [
  'Give me a fast landscape of EU AI regulation updates in 2024.',
  'Summarize recent supply chain shocks for lithium batteries.',
  'What are the top 3 competitors to Perplexity in 2024 and why?',
];
const MAX_PROGRESS_SOURCES = 60;
const MAX_PROGRESS_FINDINGS = 30;
const MAX_PROGRESS_AGENT_NOTES = 20;
const MAX_PROGRESS_QUERIES = 12;
const MAX_SAVED_PROGRESS = 20;
const DEBUG_MODE = process.env.NEXT_PUBLIC_DEBUG_MODE === 'true';

const summarizeProgressState = (progress: ProgressState) => ({
  status: progress.status,
  step: progress.step,
  queries: progress.queries?.length || 0,
  memory: progress.memoryContext.length,
  topics: progress.topics.length,
  findings: progress.findings.length,
  sources: progress.sources.length,
  agents: Object.keys(progress.agentTodos).length,
  todos: Object.values(progress.agentTodos).reduce((sum, items) => sum + items.length, 0),
  notes: Object.values(progress.agentNotes).reduce((sum, items) => sum + items.length, 0),
  isComplete: progress.isComplete,
});

export default function HomePage() {
  const initialMode = (process.env.NEXT_PUBLIC_DEFAULT_MODE as ChatMode) || 'search';
  const [mode, setMode] = useState<ChatMode>(initialMode);
  const [messages, setMessages] = useState<LocalChatMessage[]>([]);
  const [progressByMessage, setProgressByMessage] = useState<Record<string, ProgressState>>({});
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [showChatSearch, setShowChatSearch] = useState(false);
  const [chatListRefreshTrigger, setChatListRefreshTrigger] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Save progress to localStorage
  useEffect(() => {
    if (Object.keys(progressByMessage).length === 0) {
      return;
    }
    try {
      const assistantIds = messages.filter((msg) => msg.role === 'assistant').map((msg) => msg.id);
      const keepIds = assistantIds.slice(-MAX_SAVED_PROGRESS);
      const trimmed: Record<string, ProgressState> = {};
      keepIds.forEach((id) => {
        if (progressByMessage[id]) {
          trimmed[id] = progressByMessage[id];
        }
      });
      localStorage.setItem('progressByMessage', JSON.stringify(trimmed));
    } catch (e) {
      console.error('Failed to save progress to localStorage:', e);
    }
  }, [progressByMessage, messages]);

  // Load progress from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem('progressByMessage');
      if (saved) {
        const parsed = JSON.parse(saved);
        setProgressByMessage(parsed);
        if (DEBUG_MODE) {
          console.debug('[debug] restored progressByMessage', parsed);
        }
      }
    } catch (e) {
      console.error('Failed to load progress from localStorage:', e);
    }
  }, []);

  const messagePayload = useMemo(
    () => messages.map((message) => ({ role: message.role, content: message.content })),
    [messages]
  );

  // Get current progress for the latest assistant message
  const currentProgress = useMemo(() => {
    const assistantMessages = messages.filter(m => m.role === 'assistant');
    if (assistantMessages.length === 0) return null;
    const latestMessage = assistantMessages[assistantMessages.length - 1];
    return progressByMessage[latestMessage.id] || null;
  }, [messages, progressByMessage]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isStreaming]);


  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!input.trim() || isStreaming) return;

    setError(null);
    const userMessage: LocalChatMessage = { id: makeId(), role: 'user', content: input.trim() };
    const assistantMessage: LocalChatMessage = { id: makeId(), role: 'assistant', content: '' };
    const nextMessages = [...messages, userMessage];

    // Create chat if it doesn't exist
    let chatId = currentChatId;
    if (!chatId) {
      try {
        const newChat = await createChat(input.trim().slice(0, 80) || 'New Chat');
        chatId = newChat.id;
        setCurrentChatId(chatId);
      } catch (err) {
        console.error('Failed to create chat:', err);
      }
    }

    // Save user message to DB if chat exists
    if (chatId) {
      try {
        await addMessage(chatId, 'user', userMessage.content, userMessage.id);
      } catch (err) {
        console.error('Failed to save user message:', err);
      }
    }

    setMessages([...nextMessages, assistantMessage]);
    setProgressByMessage((prev) => ({
      ...prev,
      [assistantMessage.id]: {
        status: 'Starting...',
        step: 'init',
        memoryContext: [],
        researchPlan: null,
        topics: [],
        findings: [],
        sources: [],
        agentTodos: {},
        agentNotes: {},
        isComplete: false,
      },
    }));
    setInput('');
    setIsStreaming(true);

    let sessionId: string | null = null;
    let messageSaved = false; // Track if message was already saved
    try {
      // Map frontend mode to backend mode
      const backendMode = mode === 'chat' ? 'chat' : mode === 'search' ? 'search' : mode === 'deep_search' ? 'deep_search' : 'deep_research';
      
      for await (const event of streamChatProgress(
        [...messagePayload, { role: 'user' as const, content: userMessage.content }],
        backendMode as any
      )) {
        if (DEBUG_MODE) {
          console.debug('[debug] stream event', { type: event.type, data: event.data, sessionId: event.sessionId });
        }
        // Capture session ID from first event
        if (event.sessionId && !sessionId) {
          sessionId = event.sessionId;
          setCurrentSessionId(sessionId);
        }
        setProgressByMessage((prev) => {
          const existing = prev[assistantMessage.id];
          if (!existing) return prev;
          const next = { ...existing };

          switch (event.type) {
            case 'init':
              next.status = 'Session initialized';
              next.step = 'init';
              break;
            case 'status':
              next.status = event.data.message;
              next.step = event.data.step || 'progress';
              break;
            case 'memory_search':
              next.memoryContext = event.data.preview || [];
              break;
            case 'search_queries':
              next.queries = (event.data.queries || []).slice(0, MAX_PROGRESS_QUERIES);
              break;
            case 'planning':
              next.researchPlan = event.data.plan;
              next.topics = event.data.topics || [];
              break;
            case 'source_found':
              if (event.data.url || event.data.title) {
                const candidate = {
                  url: event.data.url,
                  title: event.data.title,
                  researcher_id: event.data.researcher_id,
                };
                const seen = new Set<string>();
                const deduped = [...next.sources, candidate].filter((item) => {
                  const key = `${item.url || ''}|${item.title || ''}|${item.researcher_id || ''}`;
                  if (seen.has(key)) return false;
                  seen.add(key);
                  return true;
                });
                next.sources = deduped.slice(-MAX_PROGRESS_SOURCES);
              }
              break;
            case 'finding':
              if (event.data.topic || event.data.summary) {
                next.findings = [...next.findings, {
                  topic: event.data.topic,
                  summary: event.data.summary,
                  findings_count: event.data.findings_count,
                  researcher_id: event.data.researcher_id,
                }].slice(-MAX_PROGRESS_FINDINGS);
              }
              break;
            case 'agent_todo': {
              const researcherId = event.data.researcher_id || 'agent';
              const todos = (event.data.todos || []).map((todo: { id?: string; task?: string; title?: string; status: string; note?: string | null; url?: string | null }) => ({
                title: todo.task || todo.title || 'Task',
                status: todo.status,
                note: todo.note || null,
                url: todo.url || null,
              }));
              next.agentTodos = {
                ...next.agentTodos,
                [researcherId]: todos,
              };
              break;
            }
            case 'agent_note': {
              const researcherId = event.data.researcher_id || 'agent';
              if (event.data.note) {
                const existingNotes = next.agentNotes[researcherId] || [];
                next.agentNotes = {
                  ...next.agentNotes,
                  [researcherId]: [...existingNotes, event.data.note].slice(-MAX_PROGRESS_AGENT_NOTES),
                };
              }
              break;
            }
            case 'final_report':
              next.isComplete = true;
              break;
            case 'error':
              next.error = event.data.error;
              next.isComplete = true;
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMessage.id
                    ? { ...msg, content: msg.content + `\n\n❌ **Ошибка:** ${event.data.error}` }
                    : msg
                )
              );
              break;
            case 'done':
              next.isComplete = true;
              // Save assistant message to DB if not already saved and chat exists
              // CRITICAL: Use chatId from closure (created in handleSubmit) or currentChatId
              const targetChatIdForDone = chatId || currentChatId;
              if (targetChatIdForDone && !messageSaved) {
                // Use setTimeout to ensure messages state is updated from report_chunk events
                setTimeout(async () => {
                  setMessages((prev) => {
                    const currentMessage = prev.find(m => m.id === assistantMessage.id);
                    if (currentMessage && currentMessage.content.trim()) {
                      // Save asynchronously without blocking
                      addMessage(targetChatIdForDone, 'assistant', currentMessage.content, assistantMessage.id)
                        .then(() => {
                          messageSaved = true;
                          if (DEBUG_MODE) {
                            console.debug('[debug] saved assistant message on done', { 
                              messageId: assistantMessage.id,
                              chatId: targetChatIdForDone,
                              contentLength: currentMessage.content.length 
                            });
                          }
                        })
                        .catch((err) => {
                          console.error('Failed to save assistant message on done:', err);
                          // Retry once after a short delay
                          setTimeout(async () => {
                            try {
                              await addMessage(targetChatIdForDone, 'assistant', currentMessage.content, assistantMessage.id);
                              messageSaved = true;
                              console.log('Assistant message saved on retry (done event)');
                            } catch (retryErr) {
                              console.error('Failed to save assistant message on retry (done event):', retryErr);
                            }
                          }, 1000);
                        });
                    }
                    return prev;
                  });
                }, 500);
              } else if (!targetChatIdForDone) {
                console.warn('Cannot save assistant message on done - no chat ID available', {
                  chatId,
                  currentChatId,
                  messageId: assistantMessage.id
                });
              }
              break;
          }

          if (DEBUG_MODE) {
            console.debug('[debug] progress update', { event: event.type, summary: summarizeProgressState(next) });
          }

          return { ...prev, [assistantMessage.id]: next };
        });

        if (event.type === 'report_chunk') {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantMessage.id
                ? { ...message, content: message.content + event.data.content }
                : message
            )
          );
          if (DEBUG_MODE) {
            console.debug('[debug] report chunk', { length: event.data.content?.length || 0 });
          }
        }

        if (event.type === 'final_report') {
          const finalContent = event.data.report || '';
          if (DEBUG_MODE) {
            console.debug('[debug] final report received', { 
              length: finalContent.length,
              assistantMessageId: assistantMessage.id,
              currentChatId: chatId || currentChatId,
              preview: finalContent.substring(0, 200)
            });
          }
          
          // CRITICAL: Update message content immediately
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantMessage.id
                ? { ...message, content: finalContent }
                : message
            )
          );
          
          // CRITICAL: Save final report to DB immediately
          const targetChatIdForFinalReport = chatId || currentChatId;
          if (targetChatIdForFinalReport && finalContent.trim()) {
            // Save immediately without setTimeout - this is the final content
            addMessage(targetChatIdForFinalReport, 'assistant', finalContent, assistantMessage.id)
              .then(() => {
                messageSaved = true;
                if (DEBUG_MODE) {
                  console.debug('[debug] saved final report to DB', { 
                    messageId: assistantMessage.id,
                    chatId: targetChatIdForFinalReport,
                    contentLength: finalContent.length 
                  });
                }
              })
              .catch((err) => {
                console.error('Failed to save final report to DB:', err);
                // Retry once after a short delay
                setTimeout(async () => {
                  try {
                    await addMessage(targetChatIdForFinalReport, 'assistant', finalContent, assistantMessage.id);
                    messageSaved = true;
                    console.log('Final report saved to DB on retry');
                  } catch (retryErr) {
                    console.error('Failed to save final report to DB on retry:', retryErr);
                  }
                }, 1000);
              });
          } else if (!targetChatIdForFinalReport) {
            console.warn('Cannot save final report - no chat ID available', {
              chatId,
              currentChatId,
              messageId: assistantMessage.id
            });
          }
          
          // CRITICAL: Save assistant message to DB - use chatId from closure or currentChatId
          const targetChatId = chatId || currentChatId;
          if (targetChatId && finalContent.trim() && !messageSaved) {
            try {
              await addMessage(targetChatId, 'assistant', finalContent, assistantMessage.id);
              messageSaved = true;
              if (DEBUG_MODE) {
                console.debug('[debug] saved assistant message on final_report', { 
                  messageId: assistantMessage.id, 
                  chatId: targetChatId,
                  contentLength: finalContent.length 
                });
              }
            } catch (err) {
              console.error('Failed to save assistant message:', err);
              // Retry once after a short delay
              setTimeout(async () => {
                try {
                  await addMessage(targetChatId, 'assistant', finalContent, assistantMessage.id);
                  messageSaved = true;
                  console.log('Assistant message saved on retry');
                } catch (retryErr) {
                  console.error('Failed to save assistant message on retry:', retryErr);
                }
              }, 1000);
            }
          } else if (!targetChatId) {
            console.warn('Cannot save assistant message - no chat ID available', {
              chatId,
              currentChatId,
              messageId: assistantMessage.id
            });
          }
        }
      }
    } catch (err) {
      let errorMessage = 'Chat request failed';
      if (err instanceof Error) {
        errorMessage = err.message;
      } else if (typeof err === 'string') {
        errorMessage = err;
      } else if (err && typeof err === 'object') {
        // Try to extract error message from object
        errorMessage = (err as any).message || (err as any).error || JSON.stringify(err);
      }
      
      console.error('Chat stream error:', err);
      setError(errorMessage);
      setMessages((prev) =>
        prev.map((message) =>
          message.id === assistantMessage.id
            ? { ...message, content: `${message.content}\n\n❌ **Ошибка:** ${errorMessage}` }
            : message
        )
      );
      setProgressByMessage((prev) => ({
        ...prev,
        [assistantMessage.id]: {
          ...prev[assistantMessage.id],
          status: 'Error',
          step: 'error',
          error: errorMessage,
          isComplete: true,
        },
      }));
    } finally {
      setIsStreaming(false);
      setCurrentSessionId(null);

      // Auto-generate title if this is the first message
      if (chatId && messages.length <= 2) {
        try {
          const { generateChatTitle } = await import('@/lib/api');
          await generateChatTitle(chatId);
          // Trigger chat list refresh to show updated title
          setChatListRefreshTrigger(prev => prev + 1);
          if (DEBUG_MODE) {
            console.debug('[debug] Title auto-generated for new chat');
          }
        } catch (error) {
          console.error('Failed to auto-generate title:', error);
        }
      }
    }
  };

  const handleChatSelect = async (chatId: string, messageId?: string) => {
    // Reset streaming state when selecting a chat
    setIsStreaming(false);
    setCurrentSessionId(null);
    setError(null);
    
    try {
      const chatData = await getChat(chatId);
      setCurrentChatId(chatId);
      const dbMessages: DBChatMessage[] = chatData.messages as DBChatMessage[];
      
      // CRITICAL: Load ALL messages from DB, including assistant messages
      // Sort by created_at to maintain order
      const sortedMessages = [...dbMessages].sort((a, b) => {
        const aTime = a.created_at ? new Date(a.created_at).getTime() : 0;
        const bTime = b.created_at ? new Date(b.created_at).getTime() : 0;
        return aTime - bTime;
      });
      
      const loadedMessages = sortedMessages.map((msg) => ({
        id: msg.message_id || makeId(), // Use message_id from DB
        role: msg.role as 'user' | 'assistant',
        content: msg.content || '', // Ensure content is never undefined
      }));
      
      // CRITICAL: Set messages state - this ensures all messages from DB are displayed
      setMessages(loadedMessages);
      
      // Mark all loaded messages as complete (they're in DB, so they're finished)
      // This prevents "Searching and drafting..." from showing for completed chats
      setProgressByMessage((prev) => {
        const updated = { ...prev };
        loadedMessages.forEach((msg) => {
          if (msg.role === 'assistant' && msg.content.trim()) {
            // If message exists in DB with content, it's complete
            updated[msg.id] = {
              ...updated[msg.id],
              isComplete: true,
              status: 'Complete',
              step: 'complete',
            };
          }
        });
        return updated;
      });
      
      // DON'T clear progress - it's keyed by message_id and preserved in localStorage
      // Progress for this chat's messages will be shown automatically when switching back
      if (DEBUG_MODE) {
        console.debug('[debug] chat selected', { chatId, messages: dbMessages.length, targetMessageId: messageId });
      }

      // If a specific message was requested, scroll to it after messages are rendered
      if (messageId) {
        setTimeout(() => {
          const messageElement = document.querySelector(`[data-message-id="${messageId}"]`);
          if (messageElement) {
            messageElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            // Highlight the message briefly
            messageElement.classList.add('ring-2', 'ring-primary', 'ring-offset-2');
            setTimeout(() => {
              messageElement.classList.remove('ring-2', 'ring-primary', 'ring-offset-2');
            }, 2000);
          }
        }, 100);
      }
    } catch (error: any) {
      console.error('Failed to load chat:', error);
      // If chat not found (404), reload chat list to sync with server
      if (error?.response?.status === 404 || error?.message?.includes('404') || error?.message?.includes('not found')) {
        console.warn('Chat not found, reloading chat list');
        // Trigger chat list reload in parent component if available
        window.dispatchEvent(new CustomEvent('chat-not-found', { detail: { chatId } }));
      }
      // Don't set currentChatId if chat doesn't exist
      setCurrentChatId(null);
    }
  };

  const handleNewChat = () => {
    setCurrentChatId(null);
    setMessages([]);
    setProgressByMessage({});
    setInput('');
    setIsStreaming(false);  // Ensure streaming is reset
    setCurrentSessionId(null);
    setError(null);
    localStorage.removeItem('progressByMessage');
    if (DEBUG_MODE) {
      console.debug('[debug] new chat started');
    }
  };

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      <ChatSidebar
        currentChatId={currentChatId}
        onChatSelect={handleChatSelect}
        onNewChat={handleNewChat}
        refreshTrigger={chatListRefreshTrigger}
      />
      <main className="flex-1 relative flex flex-col min-h-0 overflow-hidden bg-[radial-gradient(circle_at_top,_#f8fbff,_#f4f5f7_50%,_#eef2f7_100%)] dark:bg-[radial-gradient(circle_at_top,_#0f172a,_#1e293b_50%,_#334155_100%)]">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(15,23,42,0.04)_1px,transparent_1px),linear-gradient(180deg,rgba(15,23,42,0.04)_1px,transparent_1px)] bg-[size:56px_56px] opacity-40 dark:opacity-20" />
        <div className="absolute -top-20 left-10 h-64 w-64 rounded-full bg-amber-100/40 dark:bg-amber-900/20 blur-3xl" />
        <div className="absolute right-0 top-40 h-72 w-72 rounded-full bg-sky-100/40 dark:bg-sky-900/20 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-64 w-64 rounded-full bg-rose-100/40 dark:bg-rose-900/20 blur-3xl" />
      </div>

      <header className="relative z-10 border-b border-border/60 bg-background/70 dark:bg-background/90 backdrop-blur">
        <div className="container mx-auto px-4 py-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/15 text-primary">
                <Brain className="h-5 w-5" />
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground">All-Included</p>
                <h1 className="text-2xl font-semibold">Deep Search Console</h1>
              </div>
            </div>
            <div className="flex items-center gap-3 text-sm text-muted-foreground">
              <Badge variant="outline" className="bg-background/80">
                Memory + Web
              </Badge>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setShowChatSearch(true)}
                className="h-9 w-9"
                title="Search chats"
              >
                <Search className="h-4 w-4" />
              </Button>
              <ThemeToggle />
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-muted-foreground hover:text-foreground"
              >
                Repo <ArrowUpRight className="h-4 w-4" />
              </a>
            </div>
          </div>
        </div>
      </header>

      <section className="relative z-10 container mx-auto px-4 py-4 flex-1 flex flex-col min-h-0 overflow-hidden">
        <div className="grid gap-8 lg:grid-cols-[1fr_400px] flex-1 min-h-0 overflow-hidden">
          {/* Main Chat Area */}
          <Card className="flex h-full flex-col border-border/60 bg-background/85 dark:bg-background/95 shadow-lg backdrop-blur overflow-hidden">
            <div className="flex items-center justify-between border-b border-border/60 px-6 py-4 flex-shrink-0">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground">Conversation</p>
                <h2 className="text-lg font-semibold">Search &amp; research chat</h2>
              </div>
              <Badge variant="secondary" className="rounded-full px-3 py-1 text-xs">
                {formatModeLabel(mode)}
              </Badge>
            </div>

            <div className="flex-1 overflow-y-auto overflow-x-hidden px-6 py-5 min-h-0">
              {messages.length === 0 ? (
                <div className="flex h-full flex-col items-start justify-center gap-4">
                  <h3 className="text-2xl font-semibold">Ask anything that needs proof.</h3>
                  <p className="max-w-md text-sm text-muted-foreground">
                    Try: "Give me a quick landscape of European AI regulation changes in 2024," or
                    "Summarize recent supply chain shocks for lithium batteries."
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {suggestions.map((suggestion) => (
                      <button
                        key={suggestion}
                        type="button"
                        onClick={() => setInput(suggestion)}
                        className="rounded-full border border-border/60 bg-white/90 dark:bg-background/80 px-3 py-1 text-xs text-foreground/80 transition hover:border-primary/50 hover:text-foreground"
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {messages.map((message, index) => (
                    <div
                      key={message.id}
                      data-message-id={message.id}
                      className={cn(
                        'max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-sm animate-fade-rise transition-all',
                        message.role === 'user'
                          ? 'ml-auto bg-primary text-primary-foreground dark:bg-primary dark:text-primary-foreground'
                          : 'border border-border/60 bg-white/90 dark:bg-background/80 text-foreground'
                      )}
                      style={{ animationDelay: `${index * 40}ms` }}
                    >
                      <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:mt-4 prose-headings:mb-2 prose-p:my-2 prose-a:text-primary prose-a:underline prose-ul:my-2 prose-ol:my-2 prose-li:my-1">
                        <Markdown
                          options={{
                            overrides: {
                              a: {
                                component: ({ href, children, ...props }) => (
                                  <a
                                    href={href}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-primary hover:underline font-medium"
                                    {...props}
                                  >
                                    {children}
                                  </a>
                                ),
                              },
                              p: {
                                component: ({ children, ...props }) => (
                                  <p className="leading-relaxed" {...props}>
                                    {children}
                                  </p>
                                ),
                              },
                            },
                          }}
                        >
                          {(() => {
                            // Make citations [1], [2] clickable by converting them to markdown links
                            let processedContent = message.content;
                            
                            // Extract sources from the report (look for Sources section)
                            const sourcesSection = processedContent.match(/##\s+Sources\s+([\s\S]*?)(?=##|$)/i);
                            const sources: Record<number, string> = {};
                            
                            if (sourcesSection) {
                              const sourcesText = sourcesSection[1];
                              // Match patterns like [1] Title: URL or 1. Title: URL
                              const sourceMatches = sourcesText.matchAll(/(?:\[(\d+)\]|(\d+)\.)\s+([^:]+):\s+(https?:\/\/[^\s\)]+)/gi);
                              for (const match of sourceMatches) {
                                const num = parseInt(match[1] || match[2]);
                                const url = match[4];
                                sources[num] = url;
                              }
                            }
                            
                            // Also try to find sources in the format [1] Title: URL anywhere in the text
                            const inlineSourceMatches = processedContent.matchAll(/\[(\d+)\]\s+([^:]+):\s+(https?:\/\/[^\s\)]+)/gi);
                            for (const match of inlineSourceMatches) {
                              const num = parseInt(match[1]);
                              const url = match[3];
                              if (!sources[num]) {
                                sources[num] = url;
                              }
                            }
                            
                            // Replace citations [1], [2] with clickable links if source found
                            if (Object.keys(sources).length > 0) {
                              processedContent = processedContent.replace(
                                /\[(\d+)\](?!\s*[:\[])/g,
                                (match, num) => {
                                  const citationNum = parseInt(num);
                                  if (sources[citationNum]) {
                                    return `[${num}](${sources[citationNum]})`;
                                  }
                                  return match;
                                }
                              );
                            }
                            
                            return processedContent;
                          })()}
                        </Markdown>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {isStreaming && !currentProgress?.isComplete && (
                <div className="mt-4 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Searching and drafting...
                  </div>
                  {currentSessionId && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={async () => {
                        if (currentSessionId) {
                          try {
                            await cancelChat(currentSessionId);
                            setIsStreaming(false);
                            setCurrentSessionId(null);
                          } catch (err) {
                            console.error('Failed to cancel:', err);
                          }
                        }
                      }}
                    >
                      <X className="h-4 w-4 mr-1" />
                      Cancel
                    </Button>
                  )}
                </div>
              )}
              {error && (
                <p className="mt-4 text-sm text-destructive">Error: {error}</p>
              )}
              <div ref={scrollRef} />
            </div>

            <form onSubmit={handleSubmit} className="border-t border-border/60 px-6 py-4">
              <div className="relative">
                <Textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder="Type your question, include any constraints or sources to prioritize."
                  className="min-h-[90px] resize-none border-border/60 bg-background/80 dark:bg-background/90 pr-24"
                />
                <div className="absolute bottom-3 right-3">
                  <Button
                    type="submit"
                    size="sm"
                    className="h-8 gap-2 rounded-full px-4"
                    disabled={isStreaming || !input.trim()}
                  >
                    {isStreaming ? 'Sending' : 'Send'}
                    <ArrowUpRight className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
              {/* Mode selector at bottom left */}
              <div className="mt-2">
                <ModeSelectorDropdown 
                  selected={mode} 
                  onChange={setMode}
                />
              </div>
            </form>
          </Card>

          {/* Progress Panel - Right Side - Always show for deep_research when streaming or has progress */}
          {(mode === 'deep_research' || mode === 'deep_search' || mode === 'search') && (isStreaming || currentProgress) && (
            <div className="lg:sticky lg:top-4 lg:h-[85vh] lg:overflow-y-auto">
              <Card className="border-border/60 bg-background/85 dark:bg-background/95 shadow-lg backdrop-blur h-full flex flex-col">
                <div className="border-b border-border/60 px-4 py-3 flex items-center justify-between flex-shrink-0">
                  <h3 className="text-sm font-semibold">Research Progress</h3>
                  {currentProgress?.isComplete && mode === 'deep_research' && currentSessionId && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={async () => {
                        if (currentSessionId) {
                          try {
                            await downloadPDF(currentSessionId);
                          } catch (err) {
                            console.error('Failed to download PDF:', err);
                            setError(err instanceof Error ? err.message : 'Failed to download PDF');
                          }
                        }
                      }}
                      className="h-7 gap-1.5"
                    >
                      <Download className="h-3.5 w-3.5" />
                      PDF
                    </Button>
                  )}
                </div>
                <div className="p-4 overflow-y-auto flex-1 min-h-0">
                  {currentProgress ? (
                    <ChatProgressPanel progress={currentProgress} />
                  ) : (
                    <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
                      <Loader2 className="h-4 w-4 animate-spin mr-2" />
                      Initializing...
                    </div>
                  )}
                </div>
              </Card>
            </div>
          )}
        </div>
      </section>
      </main>
      {showChatSearch && (
        <ChatSearch
          onChatSelect={handleChatSelect}
          onClose={() => setShowChatSearch(false)}
        />
      )}
    </div>
  );
}
