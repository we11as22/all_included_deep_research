'use client';

import React, { useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CodeBlockProps {
  language?: string;
  children: React.ReactNode;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({ language, children }) => {
  const [copied, setCopied] = useState(false);
  const codeText = typeof children === 'string' ? children : String(children);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(codeText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  return (
    <div className="relative my-4 rounded-lg overflow-hidden border border-border bg-muted/50">
      {language && (
        <div className="flex items-center justify-between px-4 py-2 bg-muted border-b border-border">
          <span className="text-xs font-mono text-foreground/70">{language}</span>
          <button
            onClick={handleCopy}
            className="p-1.5 rounded hover:bg-muted/80 transition-colors"
            title="Copy code"
          >
            {copied ? (
              <Check size={14} className="text-green-600 dark:text-green-400" />
            ) : (
              <Copy size={14} className="text-foreground/70 hover:text-foreground" />
            )}
          </button>
        </div>
      )}
      <pre
        className={cn(
          'overflow-x-auto p-4 text-sm font-mono text-foreground',
          !language && 'p-4'
        )}
      >
        <code>{codeText}</code>
      </pre>
    </div>
  );
};

export default CodeBlock;
