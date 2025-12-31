'use client';

import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Brain, CheckCircle2, Loader2, Search } from 'lucide-react';
import Markdown from 'markdown-to-jsx';

const agentAccents = [
  { border: 'border-l-4 border-amber-400 dark:border-amber-500', badge: 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300' },
  { border: 'border-l-4 border-teal-400 dark:border-teal-500', badge: 'bg-teal-100 dark:bg-teal-900/30 text-teal-700 dark:text-teal-300' },
  { border: 'border-l-4 border-rose-400 dark:border-rose-500', badge: 'bg-rose-100 dark:bg-rose-900/30 text-rose-700 dark:text-rose-300' },
  { border: 'border-l-4 border-indigo-400 dark:border-indigo-500', badge: 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300' },
  { border: 'border-l-4 border-emerald-400 dark:border-emerald-500', badge: 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300' },
];

const getAgentAccent = (agentId: string) => {
  const hash = Array.from(agentId).reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return agentAccents[hash % agentAccents.length];
};

export type ProgressFinding = {
  topic?: string;
  summary?: string;
  findings_count?: number;
  researcher_id?: string;
};

export type ProgressSource = {
  url?: string;
  title?: string;
  researcher_id?: string;
};

export type AgentTodoItem = {
  title: string;
  status: string;
  note?: string | null;
  url?: string | null;
};

export type AgentNote = {
  title: string;
  summary: string;
  urls?: string[];
  shared?: boolean;
};

export type ProgressState = {
  status: string;
  step: string;
  memoryContext: Array<{ title: string; score: number; file_path?: string }>;
  researchPlan: string | null;
  topics: string[];
  findings: ProgressFinding[];
  sources: ProgressSource[];
  agentTodos: Record<string, AgentTodoItem[]>;
  agentNotes: Record<string, AgentNote[]>;
  isComplete: boolean;
  error?: string | null;
  queries?: string[];
};

export function ChatProgressPanel({ progress }: { progress: ProgressState }) {
  if (!progress) return null;

  return (
    <Card className="mb-4 border-border/60 bg-background/95 dark:bg-background/90 p-4 text-xs text-foreground shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {progress.isComplete ? (
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
          ) : (
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          )}
          <span className="text-foreground dark:text-foreground">{progress.status}</span>
        </div>
        <Badge variant="outline" className="text-[10px] uppercase tracking-[0.2em]">
          {progress.step}
        </Badge>
      </div>

      {progress.queries && progress.queries.length > 0 && (
        <div className="mt-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-foreground">
            <Search className="h-4 w-4" />
            Search Queries
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {progress.queries.map((query, idx) => (
              <Badge key={idx} variant="secondary" className="text-[10px]">
                {query}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {progress.memoryContext.length > 0 && (
        <div className="mt-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-foreground">
            <Brain className="h-4 w-4" />
            Memory Context
          </div>
          <div className="mt-2 space-y-1">
            {progress.memoryContext.map((item, idx) => (
              <div 
                key={idx} 
                className="group relative flex items-center justify-between gap-2 rounded p-1 hover:bg-background/50 dark:hover:bg-background/70"
                title={item.file_path ? `${item.title}\nFile: ${item.file_path}\nScore: ${item.score.toFixed(3)}` : `${item.title}\nScore: ${item.score.toFixed(3)}`}
              >
                <span className="truncate text-foreground dark:text-foreground">{item.title}</span>
                <span className="text-[10px] text-foreground/70 dark:text-foreground/70">score {item.score.toFixed(2)}</span>
                {/* Tooltip on hover */}
                <div className="absolute bottom-full left-0 mb-2 hidden w-64 rounded-lg border border-border/60 bg-background dark:bg-background p-2 text-[10px] shadow-lg group-hover:block z-10">
                  <div className="font-semibold text-foreground dark:text-foreground">{item.title}</div>
                  {item.file_path && <div className="mt-1 text-foreground/70 dark:text-foreground/70">File: {item.file_path}</div>}
                  <div className="mt-1 text-foreground/70 dark:text-foreground/70">Score: {item.score.toFixed(3)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {progress.researchPlan && (
        <div className="mt-3">
          <div className="text-xs font-semibold text-foreground">Research Plan</div>
          <div className="mt-2 prose prose-sm dark:prose-invert max-w-none prose-headings:mt-2 prose-headings:mb-1 prose-p:my-1 prose-a:text-primary prose-a:underline prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5">
            <Markdown
              options={{
                overrides: {
                  a: {
                    component: ({ href, children, ...props }) => (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary hover:underline text-[11px]"
                        {...props}
                      >
                        {children}
                      </a>
                    ),
                  },
                  p: {
                    component: ({ children, ...props }) => (
                      <p className="text-[11px] leading-relaxed text-foreground dark:text-foreground" {...props}>
                        {children}
                      </p>
                    ),
                  },
                  h1: {
                    component: ({ children, ...props }) => (
                      <h1 className="text-sm font-semibold mt-2 mb-1 text-foreground dark:text-foreground" {...props}>
                        {children}
                      </h1>
                    ),
                  },
                  h2: {
                    component: ({ children, ...props }) => (
                      <h2 className="text-xs font-semibold mt-2 mb-1 text-foreground dark:text-foreground" {...props}>
                        {children}
                      </h2>
                    ),
                  },
                  h3: {
                    component: ({ children, ...props }) => (
                      <h3 className="text-xs font-medium mt-1 mb-0.5 text-foreground dark:text-foreground" {...props}>
                        {children}
                      </h3>
                    ),
                  },
                  ul: {
                    component: ({ children, ...props }) => (
                      <ul className="text-[11px] list-disc list-inside my-1 text-foreground dark:text-foreground" {...props}>
                        {children}
                      </ul>
                    ),
                  },
                  ol: {
                    component: ({ children, ...props }) => (
                      <ol className="text-[11px] list-decimal list-inside my-1 text-foreground dark:text-foreground" {...props}>
                        {children}
                      </ol>
                    ),
                  },
                },
              }}
            >
              {progress.researchPlan}
            </Markdown>
          </div>
          {progress.topics.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {progress.topics.map((topic, idx) => (
                <Badge key={idx} variant="outline" className="text-[10px]">
                  {topic}
                </Badge>
              ))}
            </div>
          )}
        </div>
      )}

      {progress.findings.length > 0 && (
        <div className="mt-3">
          <div className="text-xs font-semibold text-foreground">Findings</div>
          <div className="mt-2 space-y-2">
            {progress.findings.map((finding, idx) => (
              <div 
                key={idx} 
                className="group relative rounded-lg border border-border/60 bg-background/80 dark:bg-background/70 p-2 hover:bg-background/90"
                title={finding.summary || finding.topic || 'Finding'}
              >
                <div className="font-semibold text-foreground dark:text-foreground">{finding.topic || 'Finding'}</div>
                <div className="mt-1 line-clamp-2 text-[11px] text-foreground/80 dark:text-foreground/90">{finding.summary || ''}</div>
                {/* Full description tooltip */}
                {finding.summary && (
                  <div className="absolute bottom-full left-0 mb-2 hidden w-80 rounded-lg border border-border/60 bg-background dark:bg-background p-3 text-[11px] shadow-lg group-hover:block z-10">
                    <div className="font-semibold mb-2 text-foreground dark:text-foreground">{finding.topic || 'Finding'}</div>
                    <div className="text-foreground/80 dark:text-foreground/80 leading-relaxed">{finding.summary}</div>
                    {finding.findings_count && (
                      <div className="mt-2 text-[10px] text-foreground/70 dark:text-foreground/70">
                        Related findings: {finding.findings_count}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {progress.sources.length > 0 && (
        <div className="mt-3">
          <div className="text-xs font-semibold text-foreground">Sources</div>
          <div className="mt-2 space-y-1">
            {progress.sources.slice(0, 6).map((source, idx) => (
              <a
                key={idx}
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group relative block truncate text-xs text-primary hover:underline"
                title={source.url}
              >
                {source.title || source.url || 'Source'}
                {/* Full URL tooltip */}
                {source.url && (
                  <div className="absolute bottom-full left-0 mb-2 hidden w-80 rounded-lg border border-border/60 bg-background dark:bg-background p-2 text-[10px] shadow-lg group-hover:block z-10 break-all">
                    <div className="font-semibold text-foreground dark:text-foreground">{source.title || 'Source'}</div>
                    <div className="mt-1 text-foreground/70 dark:text-foreground/70">{source.url}</div>
                  </div>
                )}
              </a>
            ))}
            {progress.sources.length > 6 && (
              <div className="text-[10px] text-foreground/70 dark:text-foreground/70">+{progress.sources.length - 6} more</div>
            )}
          </div>
        </div>
      )}

      {Object.keys(progress.agentTodos).length > 0 && (
        <div className="mt-3">
          <div className="text-xs font-semibold text-foreground">Agent Tasks</div>
          <div className="mt-2 space-y-2">
            {Object.entries(progress.agentTodos).map(([agentId, todos]) => {
              const notes = progress.agentNotes[agentId] || [];
              const pending = todos.filter((item) => item.status !== 'done');
              const inProgress = todos.filter((item) => item.status === 'in_progress');
              const completed = todos.filter((item) => item.status === 'done');
              const accent = getAgentAccent(agentId);
              return (
                <div
                  key={agentId}
                  className={`rounded-lg border border-border/60 bg-background/80 dark:bg-background/70 p-2 transition-all duration-300 ${accent.border}`}
                >
                  <div className="flex items-center justify-between text-[11px] font-semibold text-foreground dark:text-foreground">
                    <span>{agentId.replace('agent_', 'agent ')}</span>
                    <div className="flex items-center gap-2">
                      {inProgress.length > 0 && (
                        <span className="rounded-full px-2 py-0.5 text-[9px] bg-amber-500/20 text-amber-600 dark:text-amber-400 animate-pulse">
                          {inProgress.length} in progress
                        </span>
                      )}
                      <span className={`rounded-full px-2 py-0.5 text-[9px] ${accent.badge} dark:bg-background/80 dark:text-foreground`}>
                        {pending.length} pending • {completed.length} done
                      </span>
                    </div>
                  </div>
                  <div className="mt-2 space-y-1">
                    {todos.map((todo, idx) => (
                      <div 
                        key={`${todo.title}-${idx}`} 
                        className={`flex items-start gap-2 text-[10px] transition-all duration-300 ${
                          todo.status === 'done' 
                            ? 'opacity-60' 
                            : todo.status === 'in_progress' 
                            ? 'opacity-100 font-medium' 
                            : 'opacity-80'
                        }`}
                      >
                        <span className={`transition-colors duration-300 ${
                          todo.status === 'done' 
                            ? 'text-emerald-600 dark:text-emerald-400' 
                            : todo.status === 'in_progress' 
                            ? 'text-amber-600 dark:text-amber-400 animate-pulse' 
                            : 'text-muted-foreground dark:text-muted-foreground'
                        }`}>
                          {todo.status === 'done' ? '✓' : todo.status === 'in_progress' ? '⟳' : '○'}
                        </span>
                        <span className={`flex-1 transition-all duration-300 ${
                          todo.status === 'done' 
                            ? 'line-through text-foreground/60 dark:text-foreground/60' 
                            : todo.status === 'in_progress'
                            ? 'text-foreground dark:text-foreground font-medium'
                            : 'text-foreground dark:text-foreground'
                        }`}>
                          {todo.title}
                        </span>
                        {todo.note && (
                          <span className="text-[9px] text-muted-foreground/70 dark:text-muted-foreground/70" title={todo.note}>
                            ℹ
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                  {notes.length > 0 && (
                    <div className="mt-2 space-y-2 border-t border-border/60 pt-2">
                      {notes.slice(-2).map((note) => (
                        <div 
                          key={note.title} 
                          className="group relative rounded-md bg-background/80 dark:bg-background/60 p-2 hover:bg-background/90 dark:hover:bg-background/80"
                          title={note.summary}
                        >
                          <div className="text-[10px] font-semibold text-foreground dark:text-foreground">{note.title}</div>
                          <div className="mt-1 line-clamp-2 text-[10px] text-foreground/80 dark:text-foreground/90">
                            {note.summary}
                          </div>
                          {note.urls && note.urls.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-2 text-[10px]">
                              {note.urls.slice(0, 2).map((url) => (
                                <a
                                  key={url}
                                  href={url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="max-w-[120px] truncate rounded-full border border-border/60 bg-background px-2 py-0.5 text-[9px] text-primary hover:underline"
                                  title={url}
                                >
                                  {url.replace(/^https?:\/\//, '').replace(/^www\./, '').split('/')[0].slice(0, 15)}
                                </a>
                              ))}
                            </div>
                          )}
                          {/* Full note tooltip */}
                          <div className="absolute bottom-full left-0 mb-2 hidden w-80 rounded-lg border border-border/60 bg-background dark:bg-background p-3 text-[10px] shadow-lg group-hover:block z-10">
                            <div className="font-semibold mb-2 text-foreground dark:text-foreground">{note.title}</div>
                            <div className="text-foreground/80 dark:text-foreground/80 leading-relaxed">{note.summary}</div>
                            {note.urls && note.urls.length > 0 && (
                              <div className="mt-2 space-y-1">
                                <div className="font-semibold text-foreground dark:text-foreground">URLs:</div>
                                {note.urls.map((url) => (
                                  <a
                                    key={url}
                                    href={url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="block truncate text-primary hover:underline"
                                  >
                                    {url}
                                  </a>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {progress.error && <div className="mt-3 text-[11px] text-destructive">{progress.error}</div>}
    </Card>
  );
}
