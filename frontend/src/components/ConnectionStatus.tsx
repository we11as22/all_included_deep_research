'use client';

import { useChatStore } from '@/stores/chatStore';
import { CheckCircle2, WifiOff, Loader2, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

export function ConnectionStatus() {
  const connectionStatus = useChatStore((state) => state.connectionStatus);
  const queueLength = useChatStore((state) => state.messageQueue.length);

  // Don't show if online with no queued messages
  if (connectionStatus === 'online' && queueLength === 0) {
    return null;
  }

  return (
    <div
      className={cn(
        'fixed top-4 right-4 z-50 px-4 py-2 rounded-lg shadow-lg flex items-center gap-2 text-sm font-medium',
        'transition-all duration-300',
        connectionStatus === 'online' && 'bg-green-500 text-white',
        connectionStatus === 'offline' && 'bg-red-500 text-white',
        connectionStatus === 'reconnecting' && 'bg-yellow-500 text-white',
        connectionStatus === 'connecting' && 'bg-blue-500 text-white'
      )}
    >
      {connectionStatus === 'online' && queueLength > 0 && (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Sending {queueLength} queued {queueLength === 1 ? 'message' : 'messages'}...</span>
        </>
      )}
      {connectionStatus === 'offline' && (
        <>
          <WifiOff className="h-4 w-4" />
          <span>Offline - messages will be queued</span>
          {queueLength > 0 && <span className="opacity-75">({queueLength} queued)</span>}
        </>
      )}
      {connectionStatus === 'reconnecting' && (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Reconnecting...</span>
          {queueLength > 0 && <span className="opacity-75">({queueLength} queued)</span>}
        </>
      )}
      {connectionStatus === 'connecting' && (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Connecting...</span>
        </>
      )}
    </div>
  );
}
