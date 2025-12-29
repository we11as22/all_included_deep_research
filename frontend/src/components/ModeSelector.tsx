'use client';

import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { Brain, Radar, Search } from 'lucide-react';

export type ChatMode = 'search' | 'deep_search' | 'deep_research';

interface ModeSelectorProps {
  selected: ChatMode;
  onChange: (mode: ChatMode) => void;
}

const modes = [
  {
    id: 'search' as const,
    name: 'Web Search',
    icon: Search,
    description: 'Query rewriting + multi-query web search',
    details: 'Expand • rerank • summarize',
    color: 'text-amber-600',
  },
  {
    id: 'deep_search' as const,
    name: 'Deep Search',
    icon: Radar,
    description: 'Quality web search with deeper iterations',
    details: 'More rounds • broader sources',
    color: 'text-teal-600',
  },
  {
    id: 'deep_research' as const,
    name: 'Deep Research',
    icon: Brain,
    description: 'Full multi-agent report synthesis',
    details: 'Plan • parallel • report',
    color: 'text-slate-700',
  },
];

export function ModeSelector({ selected, onChange }: ModeSelectorProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {modes.map((mode) => {
        const Icon = mode.icon;
        const isSelected = selected === mode.id;

        return (
          <Card
            key={mode.id}
            className={cn(
              'p-4 cursor-pointer transition-all hover:shadow-md',
              isSelected && 'ring-2 ring-primary'
            )}
            onClick={() => onChange(mode.id)}
          >
            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Icon className={cn('h-5 w-5', mode.color)} />
                  <h3 className="font-semibold">{mode.name}</h3>
                </div>
                {isSelected && <Badge>Selected</Badge>}
              </div>
              <p className="text-sm text-muted-foreground">{mode.description}</p>
              <p className="text-xs text-muted-foreground">{mode.details}</p>
            </div>
          </Card>
        );
      })}
    </div>
  );
}
