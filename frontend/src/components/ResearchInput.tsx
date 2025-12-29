'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Send } from 'lucide-react';

interface ResearchInputProps {
  onSubmit: (query: string) => void;
  disabled?: boolean;
}

export function ResearchInput({ onSubmit, disabled }: ResearchInputProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !disabled) {
      onSubmit(query.trim());
      setQuery('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 items-end">
      <div className="flex-1">
        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="What would you like to research? (e.g., 'Latest developments in quantum computing')"
          className="min-h-[100px] resize-none"
          disabled={disabled}
        />
      </div>
      <Button type="submit" disabled={disabled || !query.trim()} size="lg" className="h-[100px]">
        <Send className="h-5 w-5" />
      </Button>
    </form>
  );
}

