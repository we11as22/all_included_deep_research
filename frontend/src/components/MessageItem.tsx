'use client';

import React, { useMemo } from 'react';
import { LocalChatMessage } from '@/stores/chatStore';
import { useChatStore } from '@/stores/chatStore';
import { cn } from '@/lib/utils';
import Markdown, { RuleType } from 'markdown-to-jsx';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { MessageDeliveryStatus } from './MessageDeliveryStatus';
import Citation from './MessageRenderer/Citation';
import CodeBlock from './MessageRenderer/CodeBlock';

interface MessageItemProps {
  message: LocalChatMessage;
}

// Helper function to process content: LaTeX + Citations (like Perplexica)
const processContent = (content: string, sources: Array<{ url?: string; title?: string }> = []): string => {
  if (!content) return '';

  // Step 1: Handle block formulas first: $$ formula $$
  content = content.replace(
    /\$\$([^$]+)\$\$/g,
    (match, formula) => {
      try {
        const rendered = katex.renderToString(formula.trim(), { throwOnError: false, displayMode: true });
        return `<div class="my-4 overflow-x-auto katex-block">${rendered}</div>`;
      } catch (e) {
        return match; // Return original if rendering fails
      }
    }
  );
  
  // Step 2: Process citations BEFORE LaTeX (like Perplexica)
  // Replace [1], [2], [1,2] with <citation> tags if sources available
  // CRITICAL: Must process BEFORE markdown parsing to avoid conflicts with markdown links
  if (sources.length > 0) {
    // Match citations [1], [2], [1,2], [1, 2, 3] but NOT markdown links [text](url)
    // Strategy: Use negative lookahead to skip markdown links [text](url)
    // Also skip if content contains non-numeric characters (likely LaTeX or markdown link text)
    const citationRegex = /\[([^\]]+)\](?!\()/g;
    content = content.replace(citationRegex, (match, capturedContent: string, offset: number, string: string) => {
      // Skip if followed by ( - it's a markdown link
      const nextChar = string[offset + match.length];
      if (nextChar === '(') {
        return match;
      }
      
      // Parse citation numbers (can be single or comma-separated: [1], [1,2], [1, 2, 3])
      // Only process if content is purely numeric (with optional commas and spaces)
      const trimmed = capturedContent.trim();
      const isCitationPattern = /^(\d+)(\s*,\s*\d+)*$/.test(trimmed);
      
      if (!isCitationPattern) {
        // Not a citation number pattern, might be LaTeX or markdown link text - leave it
        return match;
      }

      const numbers = trimmed
        .split(',')
        .map((numStr: string) => numStr.trim())
        .filter((numStr: string) => /^\d+$/.test(numStr));

      if (numbers.length === 0) {
        return match;
      }

      // Build citation HTML tags
      const linksHtml = numbers
        .map((numStr: string) => {
          const number = parseInt(numStr);
          if (isNaN(number) || number <= 0 || number > sources.length) {
            return `[${numStr}]`; // Invalid citation number
          }

          const source = sources[number - 1];
          const url = source?.url;

          if (url) {
            return `<citation href="${url}">${numStr}</citation>`;
          } else {
            return `[${numStr}]`; // No URL available
          }
        })
        .join('');

      return linksHtml || match;
    });
  }
  
  // Step 3: Handle inline LaTeX formulas: [ formula ] but not markdown links [text](url) or citations [1]
  // Match [ formula ] where formula contains LaTeX operators
  // More specific pattern to avoid conflicts with markdown and citations
  content = content.replace(
    /\[([^\]]*(?:\\text|\\cdot|\\norm|\\sum|\\int|\\frac|\\sqrt|\\alpha|\\beta|\\gamma|\\delta|\\epsilon|\\theta|\\lambda|\\mu|\\pi|\\sigma|\\phi|\\omega|\\Delta|\\Gamma|\\Lambda|\\Omega|\\Phi|\\Pi|\\Sigma|\\Theta|\\Xi|\\^|_|\\{|\\}|=|\+|\-|\*|\/)[^\]]*)\]/g,
    (match, formula, offset, string) => {
      // Skip if it looks like a markdown link [text](url) - check if followed by (
      const nextChar = string[offset + match.length];
      if (nextChar === '(') {
        return match;
      }
      // Skip if it's just a citation number [1], [2], etc. (already processed)
      if (/^\[\d+\]$/.test(match)) {
        return match;
      }
      // Skip if it's a citation tag (already processed)
      if (match.includes('<citation')) {
        return match;
      }
      try {
        const rendered = katex.renderToString(formula, { throwOnError: false, displayMode: false });
        return `<span class="katex-inline">${rendered}</span>`;
      } catch (e) {
        return match; // Return original if rendering fails
      }
    }
  );
  
  return content;
};

export const MessageItem = React.memo(
  ({ message }: MessageItemProps) => {
    // Get sources from progress for citation processing (like Perplexica)
    const progress = useChatStore((state) => state.progressByMessage[message.id]);
    const sources = progress?.sources || [];

    // Pre-process content: LaTeX + Citations (like Perplexica)
    const processedContent = useMemo(() => {
      if (!message.content) return '';
      return processContent(message.content, sources);
    }, [message.content, sources]);

    return (
      <div
        className={cn(
          'p-4 rounded-lg mb-4 animate-fade-rise',
          message.role === 'user' && 'bg-blue-50 dark:bg-blue-950 ml-auto max-w-[80%]',
          message.role === 'assistant' && 'bg-white dark:bg-gray-800 mr-auto',
          message.role === 'system' && 'bg-muted/60 text-muted-foreground mx-auto max-w-[85%]'
        )}
      >
        {message.content && (
          <div className={cn(
            'prose prose-sm dark:prose-invert max-w-none break-words',
            'prose-h1:mb-3 prose-h1:mt-6 prose-h1:font-bold prose-h1:text-2xl',
            'prose-h2:mb-2 prose-h2:mt-6 prose-h2:font-[800]',
            'prose-h3:mt-4 prose-h3:mb-1.5 prose-h3:font-[600]',
            'prose-p:leading-relaxed prose-p:whitespace-pre-wrap',
            'prose-pre:p-0',
            'text-black dark:text-white'
          )}>
            <Markdown
              options={{
                forceBlock: true,
                forceInline: false,
                wrapper: React.Fragment,
                disableParsingRawHTML: false,
                // CRITICAL: Use minimal overrides like Perplexica - let prose classes handle most styling
                // Only override for special cases (code blocks, citations, LaTeX)
                renderRule(next, node, renderChildren, state) {
                  // Handle code blocks with syntax highlighting
                  if (node.type === RuleType.codeBlock) {
                    return (
                      <CodeBlock key={state.key} language={node.lang || ''}>
                        {node.text}
                      </CodeBlock>
                    );
                  }
                  // Handle inline code (let markdown handle it)
                  if (node.type === RuleType.codeInline) {
                    return next();
                  }
                  return next();
                },
                overrides: {
                  // Citation component (from Perplexica pattern)
                  citation: {
                    component: ({ href, children, ...props }) => (
                      <Citation href={href} {...props}>
                        {children}
                      </Citation>
                    ),
                  },
                  // Keep LaTeX rendering support
                  div: {
                    component: ({ className, children, ...props }) => {
                      // Preserve LaTeX blocks
                      if (className?.includes('katex-block')) {
                        return <div className={className} dangerouslySetInnerHTML={{ __html: children as string }} {...props} />;
                      }
                      return <div className={className} {...props}>{children}</div>;
                    },
                  },
                  span: {
                    component: ({ className, children, ...props }) => {
                      // Preserve LaTeX inline
                      if (className?.includes('katex-inline')) {
                        return <span className={className} dangerouslySetInnerHTML={{ __html: children as string }} {...props} />;
                      }
                      return <span className={className} {...props}>{children}</span>;
                    },
                  },
                },
              }}
            >
              {processedContent || ''}
            </Markdown>
          </div>
        )}
        {!message.content && message.role === 'assistant' && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <div className="animate-pulse">Thinking...</div>
          </div>
        )}
        <MessageDeliveryStatus message={message} />
      </div>
    );
  },
  (prev, next) => {
    // Only re-render if content or status changed
    return (
      prev.message.id === next.message.id &&
      prev.message.content === next.message.content &&
      prev.message.status === next.message.status &&
      prev.message.error === next.message.error
    );
  }
);

MessageItem.displayName = 'MessageItem';
