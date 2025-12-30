'use client';

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { Brain, MessageSquare, Radar, Search } from 'lucide-react';
import { useState } from 'react';

export type ChatMode = 'chat' | 'search' | 'deep_search' | 'deep_research';

interface ModeSelectorInlineProps {
  selected: ChatMode;
  onChange: (mode: ChatMode) => void;
  onHover?: (mode: ChatMode | null) => void;
}

const modes = [
  {
    id: 'chat' as const,
    name: 'Chat',
    icon: MessageSquare,
    description: 'Simple conversation with LLM without web search',
    fullDescription: 'Direct conversation with the language model. No web search, no external sources. Perfect for general questions, brainstorming, or when you want a quick response without research.',
    color: 'text-blue-600',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
  },
  {
    id: 'search' as const,
    name: 'Web Search',
    icon: Search,
    description: 'Query rewriting + multi-query web search',
    fullDescription: 'Fast web search with query expansion and reranking. Expands your query into multiple search queries, finds relevant sources, and summarizes results. Best for quick fact-checking and current information.',
    color: 'text-amber-600',
    bgColor: 'bg-amber-50',
    borderColor: 'border-amber-200',
  },
  {
    id: 'deep_search' as const,
    name: 'Deep Search',
    icon: Radar,
    description: 'Quality web search with deeper iterations',
    fullDescription: 'Enhanced web search with multiple rounds of query refinement. Performs deeper iterations, explores broader sources, and provides more comprehensive results. Ideal for complex topics requiring thorough investigation.',
    color: 'text-teal-600',
    bgColor: 'bg-teal-50',
    borderColor: 'border-teal-200',
  },
  {
    id: 'deep_research' as const,
    name: 'Deep Research',
    icon: Brain,
    description: 'Full multi-agent report synthesis',
    fullDescription: 'Comprehensive multi-agent research system. Creates a research plan, deploys multiple specialized agents in parallel, synthesizes findings, and generates a detailed report. Perfect for in-depth research requiring multiple perspectives and thorough analysis.',
    color: 'text-slate-700',
    bgColor: 'bg-slate-50',
    borderColor: 'border-slate-200',
  },
];

export function ModeSelectorInline({ selected, onChange, onHover }: ModeSelectorInlineProps) {
  const [hoveredMode, setHoveredMode] = useState<ChatMode | null>(null);

  const handleMouseEnter = (mode: ChatMode) => {
    setHoveredMode(mode);
    onHover?.(mode);
  };

  const handleMouseLeave = () => {
    setHoveredMode(null);
    onHover?.(null);
  };

  return (
    <div className="relative">
      <div className="flex items-center gap-2">
        {modes.map((mode) => {
          const Icon = mode.icon;
          const isSelected = selected === mode.id;
          const isHovered = hoveredMode === mode.id;

          return (
            <div
              key={mode.id}
              className="relative"
              onMouseEnter={() => handleMouseEnter(mode.id)}
              onMouseLeave={handleMouseLeave}
            >
              <button
                type="button"
                onClick={() => onChange(mode.id)}
                className={cn(
                  'flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all',
                  isSelected
                    ? `${mode.bgColor} ${mode.borderColor} border-2 text-foreground`
                    : 'border border-border/60 bg-background/80 text-muted-foreground hover:bg-background hover:text-foreground',
                  isHovered && !isSelected && 'border-primary/50'
                )}
              >
                <Icon className={cn('h-3.5 w-3.5', isSelected ? mode.color : 'text-muted-foreground')} />
                <span>{mode.name}</span>
                {isSelected && (
                  <Badge variant="secondary" className="ml-1 h-4 px-1.5 text-[9px]">
                    Active
                  </Badge>
                )}
              </button>

              {/* Tooltip on hover */}
              {isHovered && (
                <div
                  className={cn(
                    'absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-lg border p-3 text-xs shadow-lg',
                    mode.bgColor,
                    mode.borderColor,
                    'border-2'
                  )}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <Icon className={cn('h-4 w-4', mode.color)} />
                    <span className="font-semibold text-foreground">{mode.name}</span>
                  </div>
                  <p className="text-muted-foreground leading-relaxed">{mode.fullDescription}</p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

