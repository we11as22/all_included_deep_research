'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { ModeSelector, ChatMode } from '@/components/ModeSelector';
import { streamChatProgress } from '@/lib/api';
import { ChatProgressPanel, ProgressState } from '@/components/ChatProgressPanel';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { ArrowUpRight, Brain, Loader2, X } from 'lucide-react';
import { cancelChat, getChat, addMessage, createChat, type ChatMessage as DBChatMessage, type StreamChatMessage } from '@/lib/api';
import { cn } from '@/lib/utils';
import Markdown from 'markdown-to-jsx';
import { ChatSidebar } from '@/components/ChatSidebar';

type LocalChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
};

const makeId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
const formatModeLabel = (mode: ChatMode) => {
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
  const scrollRef = useRef<HTMLDivElement>(null);

  const messagePayload = useMemo(
    () => messages.map((message) => ({ role: message.role, content: message.content })),
    [messages]
  );

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
    try {
      for await (const event of streamChatProgress(
        [...messagePayload, { role: 'user' as const, content: userMessage.content }],
        mode
      )) {
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
              next.queries = event.data.queries || [];
              break;
            case 'planning':
              next.researchPlan = event.data.plan;
              next.topics = event.data.topics || [];
              break;
            case 'source_found':
              if (event.data.url || event.data.title) {
                next.sources = [...next.sources, {
                  url: event.data.url,
                  title: event.data.title,
                  researcher_id: event.data.researcher_id,
                }];
              }
              break;
            case 'finding':
              if (event.data.topic || event.data.summary) {
                next.findings = [...next.findings, {
                  topic: event.data.topic,
                  summary: event.data.summary,
                  findings_count: event.data.findings_count,
                  researcher_id: event.data.researcher_id,
                }];
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
                  [researcherId]: [...existingNotes, event.data.note],
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
              // Also add error to message content for visibility
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
              break;
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
        }

        if (event.type === 'final_report') {
          const finalContent = event.data.report || '';
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantMessage.id
                ? { ...message, content: finalContent }
                : message
            )
          );
          
          // Save assistant message to DB if chat exists
          if (currentChatId && finalContent) {
            try {
              await addMessage(currentChatId, 'assistant', finalContent, assistantMessage.id);
            } catch (err) {
              console.error('Failed to save assistant message:', err);
            }
          }
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Chat request failed';
      setError(message);
      setMessages((prev) =>
        prev.map((message) =>
          message.id === assistantMessage.id
            ? { ...message, content: `${message.content}\n\nError: ${message}` }
            : message
        )
      );
      setProgressByMessage((prev) => ({
        ...prev,
        [assistantMessage.id]: {
          ...prev[assistantMessage.id],
          status: 'Error',
          step: 'error',
          error: message,
          isComplete: true,
        },
      }));
    } finally {
      setIsStreaming(false);
    }
  };

  const handleChatSelect = async (chatId: string) => {
    try {
      const chatData = await getChat(chatId);
      setCurrentChatId(chatId);
      const dbMessages: DBChatMessage[] = chatData.messages as DBChatMessage[];
      setMessages(
        dbMessages.map((msg) => ({
          id: msg.message_id,
          role: msg.role as 'user' | 'assistant',
          content: msg.content,
        }))
      );
      setProgressByMessage({});
    } catch (error) {
      console.error('Failed to load chat:', error);
    }
  };

  const handleNewChat = () => {
    setCurrentChatId(null);
    setMessages([]);
    setProgressByMessage({});
    setInput('');
  };

  return (
    <div className="flex min-h-screen bg-background">
      <ChatSidebar
        currentChatId={currentChatId}
        onChatSelect={handleChatSelect}
        onNewChat={handleNewChat}
      />
      <main className="flex-1 relative min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top,_#f8fbff,_#f4f5f7_50%,_#eef2f7_100%)]">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(15,23,42,0.04)_1px,transparent_1px),linear-gradient(180deg,rgba(15,23,42,0.04)_1px,transparent_1px)] bg-[size:56px_56px] opacity-40" />
        <div className="absolute -top-20 left-10 h-64 w-64 rounded-full bg-amber-100/40 blur-3xl" />
        <div className="absolute right-0 top-40 h-72 w-72 rounded-full bg-sky-100/40 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-64 w-64 rounded-full bg-rose-100/40 blur-3xl" />
      </div>

      <header className="relative z-10 border-b border-border/60 bg-background/70 backdrop-blur">
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

      <section className="relative z-10 container mx-auto px-4 py-10">
        <div className="grid gap-8 lg:grid-cols-[320px_1fr]">
          <aside className="space-y-6">
            <Card className="border-border/60 bg-background/80 p-5 shadow-sm">
              <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground">Mode</p>
              <h2 className="mt-2 text-lg font-semibold">Choose a search lane</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                Each lane changes how aggressively the agent searches, reranks, and synthesizes.
              </p>
              <div className="mt-4">
                <ModeSelector selected={mode} onChange={setMode} />
              </div>
            </Card>

            <Card className="border-border/60 bg-background/80 p-5 shadow-sm">
              <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground">Capabilities</p>
              <div className="mt-3 space-y-3 text-sm text-muted-foreground">
                <p>Hybrid memory + web retrieval with inline citations.</p>
                <p>Web search expands queries and summarizes long sources.</p>
                <p>Deep search runs more iterations with broader sources.</p>
                <p>Deep research orchestrates multi-agent synthesis.</p>
              </div>
            </Card>
          </aside>

          <Card className="flex min-h-[600px] max-h-[85vh] flex-col border-border/60 bg-background/85 shadow-lg backdrop-blur">
            <div className="flex items-center justify-between border-b border-border/60 px-6 py-4">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground">Conversation</p>
                <h2 className="text-lg font-semibold">Search &amp; research chat</h2>
              </div>
              <Badge variant="secondary" className="rounded-full px-3 py-1 text-xs">
                {formatModeLabel(mode)}
              </Badge>
            </div>

            <div className="flex-1 overflow-y-auto px-6 py-5">
              {messages.length === 0 ? (
                <div className="flex h-full flex-col items-start justify-center gap-4">
                  <h3 className="text-2xl font-semibold">Ask anything that needs proof.</h3>
                  <p className="max-w-md text-sm text-muted-foreground">
                    Try: “Give me a quick landscape of European AI regulation changes in 2024,” or
                    “Summarize recent supply chain shocks for lithium batteries.”
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {suggestions.map((suggestion) => (
                      <button
                        key={suggestion}
                        type="button"
                        onClick={() => setInput(suggestion)}
                        className="rounded-full border border-border/60 bg-white/90 px-3 py-1 text-xs text-foreground/80 transition hover:border-primary/50 hover:text-foreground"
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
                      className={cn(
                        'max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-sm animate-fade-rise',
                        message.role === 'user'
                          ? 'ml-auto bg-foreground text-background'
                          : 'border border-border/60 bg-white/90 text-foreground'
                      )}
                      style={{ animationDelay: `${index * 40}ms` }}
                    >
                      {message.role === 'assistant' && progressByMessage[message.id] && (
                        <ChatProgressPanel progress={progressByMessage[message.id]} />
                      )}
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
                          {message.content}
                        </Markdown>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {isStreaming && (
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
              <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
                <div className="flex-1">
                  <Textarea
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    placeholder="Type your question, include any constraints or sources to prioritize."
                    className="min-h-[90px] resize-none border-border/60 bg-background/80"
                  />
                </div>
                <Button
                  type="submit"
                  className="h-12 gap-2 rounded-full px-6"
                  disabled={isStreaming || !input.trim()}
                >
                  {isStreaming ? 'Searching' : 'Send'}
                  <ArrowUpRight className="h-4 w-4" />
                </Button>
              </div>
            </form>
          </Card>
        </div>
      </section>
      </main>
    </div>
  );
}
