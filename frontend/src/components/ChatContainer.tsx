'use client';

import { useChatStore } from '@/stores/chatStore';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { ChatProgressPanel } from './ChatProgressPanel';

export function ChatContainer() {
  const messages = useChatStore((state) => state.messages);
  const progressByMessage = useChatStore((state) => state.progressByMessage);
  const progressPanelMessageId = useChatStore((state) => state.progressPanelMessageId);

  const progress = progressPanelMessageId
    ? progressByMessage[progressPanelMessageId]
    : undefined;
  const showProgressPanel = Boolean(progress);

  return (
    <div className="flex flex-1 min-h-0">
      <div className="flex-1 flex flex-col min-h-0">
        <MessageList messages={messages} />
        <ChatInput />
      </div>
      {showProgressPanel && progress && (
        <div className="w-[400px] border-l overflow-y-auto min-h-0">
          <ChatProgressPanel progress={progress} />
        </div>
      )}
    </div>
  );
}
