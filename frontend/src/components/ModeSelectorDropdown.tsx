'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { Brain, ChevronDown, MessageSquare, Radar, Search } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';

export type ChatMode = 'chat' | 'search' | 'deep_search' | 'deep_research';

interface ModeSelectorDropdownProps {
  selected: ChatMode;
  onChange: (mode: ChatMode) => void;
}

const modes = [
  {
    id: 'chat' as const,
    name: 'Chat',
    icon: MessageSquare,
    description: 'Simple conversation with LLM without web search',
    fullDescription: 'Direct conversation with the language model. No web search, no external sources. Perfect for general questions, brainstorming, or when you want a quick response without research.',
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-50 dark:bg-blue-950',
    borderColor: 'border-blue-200 dark:border-blue-800',
  },
  {
    id: 'search' as const,
    name: 'Web Search',
    icon: Search,
    description: 'Query rewriting + multi-query web search',
    fullDescription: 'Fast web search with query expansion and reranking. Expands your query into multiple search queries, finds relevant sources, and summarizes results. Best for quick fact-checking and current information.',
    color: 'text-amber-600 dark:text-amber-400',
    bgColor: 'bg-amber-50 dark:bg-amber-950',
    borderColor: 'border-amber-200 dark:border-amber-800',
  },
  {
    id: 'deep_search' as const,
    name: 'Deep Search',
    icon: Radar,
    description: 'Quality web search with deeper iterations',
    fullDescription: 'Enhanced web search with multiple rounds of query refinement. Performs deeper iterations, explores broader sources, and provides more comprehensive results. Ideal for complex topics requiring thorough investigation.',
    color: 'text-teal-600 dark:text-teal-400',
    bgColor: 'bg-teal-50 dark:bg-teal-950',
    borderColor: 'border-teal-200 dark:border-teal-800',
  },
  {
    id: 'deep_research' as const,
    name: 'Deep Research',
    icon: Brain,
    description: 'Full multi-agent report synthesis',
    fullDescription: 'Comprehensive multi-agent research system. Creates a research plan, deploys multiple specialized agents in parallel, synthesizes findings, and generates a detailed report. Perfect for in-depth research requiring multiple perspectives and thorough analysis.',
    color: 'text-slate-700 dark:text-slate-300',
    bgColor: 'bg-slate-50 dark:bg-slate-950',
    borderColor: 'border-slate-200 dark:border-slate-800',
  },
];

export function ModeSelectorDropdown({ selected, onChange }: ModeSelectorDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const selectedMode = modes.find(m => m.id === selected) || modes[0];
  const Icon = selectedMode.icon;

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  return (
    <div className="relative" ref={dropdownRef}>
      <Button
        type="button"
        variant="outline"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium transition-all",
          "border-border/60 bg-background/80 hover:bg-background",
          "dark:bg-background/90 dark:hover:bg-background"
        )}
      >
        <Icon className={cn("h-3.5 w-3.5", selectedMode.color)} />
        <span className="text-foreground">{selectedMode.name}</span>
        <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", isOpen && "rotate-180")} />
      </Button>

      {isOpen && (
        <div className="absolute bottom-full left-0 mb-2 w-80 rounded-lg border border-border/60 bg-background shadow-lg z-50">
          <div className="p-2 space-y-1">
            {modes.map((mode) => {
              const ModeIcon = mode.icon;
              const isSelected = selected === mode.id;

              return (
                <button
                  key={mode.id}
                  type="button"
                  onClick={() => {
                    onChange(mode.id);
                    setIsOpen(false);
                  }}
                  className={cn(
                    "w-full flex items-start gap-3 rounded-lg p-3 text-left transition-all",
                    "hover:bg-muted/50",
                    isSelected && `${mode.bgColor} ${mode.borderColor} border-2`,
                    !isSelected && "border border-transparent"
                  )}
                >
                  <ModeIcon className={cn("h-4 w-4 mt-0.5 flex-shrink-0", mode.color)} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className={cn("font-semibold text-sm", isSelected ? "text-foreground" : "text-foreground")}>
                        {mode.name}
                      </span>
                      {isSelected && (
                        <Badge variant="secondary" className="h-5 px-1.5 text-[9px]">
                          Active
                        </Badge>
                      )}
                    </div>
                    <p className={cn("mt-1 text-xs leading-relaxed", isSelected ? "text-muted-foreground" : "text-muted-foreground")}>
                      {mode.fullDescription}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

