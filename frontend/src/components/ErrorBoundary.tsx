'use client';

import React from 'react';
import { ErrorBoundary as ReactErrorBoundary } from 'react-error-boundary';
import { AlertTriangle } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';

interface ErrorFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
}

function ErrorFallback({ error, resetErrorBoundary }: ErrorFallbackProps) {
  return (
    <div className="flex h-screen items-center justify-center bg-background p-4">
      <Card className="max-w-md p-6 space-y-4">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-6 w-6 text-red-500" />
          <h2 className="text-xl font-bold text-foreground">Что-то пошло не так</h2>
        </div>
        <p className="text-sm text-muted-foreground">
          Произошла непредвиденная ошибка. Попробуйте обновить страницу или нажмите кнопку ниже.
        </p>
        <pre className="mt-2 p-4 bg-muted rounded text-xs overflow-auto max-h-40">
          {error.message}
        </pre>
        <div className="flex gap-2">
          <Button onClick={resetErrorBoundary} className="flex-1">
            Попробовать снова
          </Button>
          <Button
            variant="outline"
            onClick={() => window.location.reload()}
            className="flex-1"
          >
            Обновить страницу
          </Button>
        </div>
      </Card>
    </div>
  );
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

export function ErrorBoundary({ children }: ErrorBoundaryProps) {
  return (
    <ReactErrorBoundary
      FallbackComponent={ErrorFallback}
      onReset={() => {
        // Reset application state if needed
        window.location.href = '/';
      }}
    >
      {children}
    </ReactErrorBoundary>
  );
}
