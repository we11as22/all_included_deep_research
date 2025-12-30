'use client';

import { useEffect, useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ResearchEvent, streamResearch } from '@/lib/api';
import { Loader2, CheckCircle, XCircle, FileText, Search, Lightbulb, Brain } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ResearchStreamProps {
  query: string;
  mode: 'speed' | 'balanced' | 'quality';
  onComplete?: (report: string) => void;
  onError?: (error: string) => void;
}

interface StreamState {
  status: string;
  step: string;
  memoryContext: any[];
  researchPlan: string | null;
  topics: string[];
  findings: any[];
  sources: any[];
  reportChunks: string[];
  finalReport: string | null;
  error: string | null;
  isComplete: boolean;
}

export function ResearchStream({ query, mode, onComplete, onError }: ResearchStreamProps) {
  const [state, setState] = useState<StreamState>({
    status: 'Initializing...',
    step: 'init',
    memoryContext: [],
    researchPlan: null,
    topics: [],
    findings: [],
    sources: [],
    reportChunks: [],
    finalReport: null,
    error: null,
    isComplete: false,
  });

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;

    async function runResearch() {
      try {
        for await (const event of streamResearch({ query, mode, saveToMemory: true })) {
          if (cancelled) break;

          setState((prev) => {
            const newState = { ...prev };

            switch (event.type) {
              case 'init':
                newState.status = 'Research started';
                break;

              case 'status':
                newState.status = event.data.message || newState.status;
                newState.step = event.data.step || newState.step;
                break;

              case 'memory_search':
                newState.memoryContext = event.data.preview || [];
                break;

              case 'planning':
                newState.researchPlan = event.data.plan;
                newState.topics = event.data.topics || [];
                break;

              case 'source_found':
                newState.sources.push(event.data);
                break;

              case 'finding':
                newState.findings.push(event.data);
                break;

              case 'report_chunk':
                newState.reportChunks.push(event.data.content);
                break;

              case 'final_report':
                newState.finalReport = event.data.report;
                newState.isComplete = true;
                onComplete?.(event.data.report);
                break;

              case 'error':
                newState.error = event.data.error;
                newState.isComplete = true;
                onError?.(event.data.error);
                break;

              case 'done':
                newState.isComplete = true;
                break;
            }

            return newState;
          });
        }
      } catch (error) {
        if (!cancelled) {
          const errorMsg = error instanceof Error ? error.message : 'Research failed';
          setState((prev) => ({
            ...prev,
            error: errorMsg,
            isComplete: true,
          }));
          onError?.(errorMsg);
        }
      }
    }

    runResearch();

    return () => {
      cancelled = true;
    };
  }, [query, mode]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [state]);

  return (
    <div className="space-y-4">
      {/* Status Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Research Progress</CardTitle>
            <div className="flex items-center gap-2">
              {!state.isComplete ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="text-sm text-muted-foreground">{state.status}</span>
                </>
              ) : state.error ? (
                <>
                  <XCircle className="h-4 w-4 text-destructive" />
                  <span className="text-sm text-destructive">Failed</span>
                </>
              ) : (
                <>
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm text-green-500">Complete</span>
                </>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Memory Context */}
      {state.memoryContext.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Memory Context ({state.memoryContext.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {state.memoryContext.map((ctx, idx) => (
                <div key={idx} className="text-sm">
                  <span className="font-medium">{ctx.title}</span>
                  <span className="text-muted-foreground ml-2">
                    (score: {ctx.score.toFixed(2)})
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Research Plan */}
      {state.researchPlan && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Research Plan
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="text-sm whitespace-pre-wrap">{state.researchPlan}</div>
              {state.topics.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {state.topics.map((topic, idx) => (
                    <Badge key={idx} variant="outline">
                      {topic}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Findings */}
      {state.findings.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Lightbulb className="h-4 w-4" />
              Research Findings ({state.findings.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {state.findings.map((finding, idx) => (
                <div key={idx} className="border-l-2 border-primary pl-4">
                  <h4 className="font-medium text-sm mb-1">{finding.topic || 'Finding'}</h4>
                  <p className="text-sm text-muted-foreground">{finding.summary || ''}</p>
                  <div className="mt-2">
                    <Badge variant="secondary" className="text-xs">
                      {finding.findings_count} key findings
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Sources */}
      {state.sources.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Search className="h-4 w-4" />
              Sources ({state.sources.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {state.sources.slice(0, 10).map((source, idx) => (
                <div key={idx} className="text-sm">
                  {source.url ? (
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline"
                    >
                      {source.title || source.url || 'Source'}
                    </a>
                  ) : (
                    <span>{source.title || 'Source'}</span>
                  )}
                </div>
              ))}
              {state.sources.length > 10 && (
                <p className="text-xs text-muted-foreground">
                  And {state.sources.length - 10} more sources...
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Final Report */}
      {(state.reportChunks.length > 0 || state.finalReport) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Final Research Report</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="prose prose-slate dark:prose-invert max-w-none">
              <div className="whitespace-pre-wrap">
                {state.finalReport || state.reportChunks.join('')}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Error */}
      {state.error && (
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="text-base text-destructive flex items-center gap-2">
              <XCircle className="h-4 w-4" />
              Error
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">{state.error}</p>
          </CardContent>
        </Card>
      )}

      <div ref={bottomRef} />
    </div>
  );
}

