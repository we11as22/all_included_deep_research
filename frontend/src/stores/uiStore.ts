import { create } from 'zustand';

interface UIStore {
  showChatSearch: boolean;
  chatListRefreshTrigger: number;
  sidebarCollapsed: boolean;

  setShowChatSearch: (show: boolean) => void;
  triggerChatListRefresh: () => void;
  toggleSidebar: () => void;
}

export const useUIStore = create<UIStore>((set) => ({
  showChatSearch: false,
  chatListRefreshTrigger: 0,
  sidebarCollapsed: false,

  setShowChatSearch: (show) => set({ showChatSearch: show }),
  triggerChatListRefresh: () =>
    set((state) => ({ chatListRefreshTrigger: state.chatListRefreshTrigger + 1 })),
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
}));
