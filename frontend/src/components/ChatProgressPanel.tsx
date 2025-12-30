'use client';

import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Brain, CheckCircle2, Loader2, Search } from 'lucide-react';

const agentAccents = [
  { border: 'border-l-4 border-amber-400', badge: 'bg-amber-100 text-amber-700' },
  { border: 'border-l-4 border-teal-400', badge: 'bg-teal-100 text-teal-700' },
  { border: 'border-l-4 border-rose-400', badge: 'bg-rose-100 text-rose-700' },
  { border: 'border-l-4 border-indigo-400', badge: 'bg-indigo-100 text-indigo-700' },
  { border: 'border-l-4 border-emerald-400', badge: 'bg-emerald-100 text-emerald-700' },
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
    <Card className="mb-4 border-border/60 bg-white/80 p-4 text-xs text-muted-foreground shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {progress.isComplete ? (
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
          ) : (
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          )}
          <span>{progress.status}</span>
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
              <div key={idx} className="flex items-center justify-between gap-2">
                <span className="truncate">{item.title}</span>
                <span className="text-[10px]">score {item.score.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {progress.researchPlan && (
        <div className="mt-3">
          <div className="text-xs font-semibold text-foreground">Research Plan</div>
          <p className="mt-2 whitespace-pre-wrap text-[11px] text-muted-foreground">
            {progress.researchPlan}
          </p>
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
              <div key={idx} className="rounded-lg border border-border/60 bg-white/70 p-2">
                <div className="font-semibold text-foreground">{finding.topic || 'Finding'}</div>
                <div className="mt-1 line-clamp-2 text-[11px]">{finding.summary || ''}</div>
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
                className="block truncate text-xs text-primary hover:underline"
              >
                {source.title || source.url || 'Source'}
              </a>
            ))}
            {progress.sources.length > 6 && (
              <div className="text-[10px]">+{progress.sources.length - 6} more</div>
            )}
          </div>
        </div>
      )}

      {Object.keys(progress.agentTodos).length > 0 && (
        <div className="mt-3">
          <div className="text-xs font-semibold text-foreground">Agent Threads</div>
          <div className="mt-2 space-y-2">
            {Object.entries(progress.agentTodos).map(([agentId, todos]) => {
              const notes = progress.agentNotes[agentId] || [];
              const pending = todos.filter((item) => item.status !== 'done');
              const accent = getAgentAccent(agentId);
              return (
                <div
                  key={agentId}
                  className={`rounded-lg border border-border/60 bg-white/70 p-2 ${accent.border}`}
                >
                  <div className="flex items-center justify-between text-[11px] font-semibold text-foreground">
                    <span>{agentId.replace('agent_', 'agent ')}</span>
                    <span className={`rounded-full px-2 py-0.5 text-[9px] ${accent.badge}`}>
                      {pending.length} pending
                    </span>
                  </div>
                  <div className="mt-2 space-y-1">
                    {todos.map((todo, idx) => (
                      <div key={`${todo.title}-${idx}`} className="flex items-start gap-2 text-[10px]">
                        <span className={todo.status === 'done' ? 'text-emerald-600' : todo.status === 'in_progress' ? 'text-amber-600' : 'text-muted-foreground'}>
                          {todo.status === 'done' ? '●' : todo.status === 'in_progress' ? '◐' : '○'}
                        </span>
                        <span className="flex-1">{todo.title}</span>
                      </div>
                    ))}
                  </div>
                  {notes.length > 0 && (
                    <div className="mt-2 space-y-2 border-t border-border/60 pt-2">
                      {notes.slice(-2).map((note) => (
                        <div key={note.title} className="rounded-md bg-white/80 p-2">
                          <div className="text-[10px] font-semibold text-foreground">{note.title}</div>
                          <div className="mt-1 line-clamp-2 text-[10px] text-muted-foreground">
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
