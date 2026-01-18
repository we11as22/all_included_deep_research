'use client';

import React from 'react';

interface CitationProps {
  href: string;
  children: React.ReactNode;
}

export const Citation: React.FC<CitationProps> = ({ href, children }) => {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="bg-muted dark:bg-muted/50 px-1.5 py-0.5 rounded ml-1 no-underline text-xs text-foreground/70 hover:text-foreground hover:bg-muted/80 dark:hover:bg-muted/70 transition-colors"
      onClick={(e) => {
        e.stopPropagation();
      }}
    >
      {children}
    </a>
  );
};

export default Citation;
