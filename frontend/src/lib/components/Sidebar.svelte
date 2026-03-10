<script lang="ts">
  interface Session { id: string; title: string; messages: unknown[] }

  interface Props {
    sessions: Session[];
    currentId: string | null;
    open: boolean;
    onNewChat: () => void;
    onLoadSession: (id: string) => void;
    onDeleteSession: (id: string) => void;
    onClearAll: () => void;
  }

  let { sessions, currentId, open, onNewChat, onLoadSession, onDeleteSession, onClearAll }: Props = $props();
</script>

<aside
  class="flex flex-col flex-shrink-0
         border-r border-black/[.08] dark:border-white/[.06]
         bg-gray-50 dark:bg-[#0f0f14]
         transition-all duration-300 overflow-hidden"
  style="width: {open ? '248px' : '0px'}"
>
  <!-- Header -->
  <div class="flex items-center justify-between px-4 py-4 border-b border-black/[.06] dark:border-white/[.06]">
    <span class="mono text-[10px] tracking-[.14em] uppercase text-zinc-800 dark:text-white">History</span>
    <button
      onclick={onNewChat}
      class="w-7 h-7 rounded-lg bg-violet-600 hover:bg-violet-500 flex items-center justify-center
             text-white text-lg leading-none transition-colors"
      title="New chat (⌘K)"
    >+</button>
  </div>

  <!-- Session list -->
  <nav class="flex-1 overflow-y-auto p-2 space-y-0.5">
    {#if sessions.length === 0}
      <p class="mono text-[11px] text-black/25 dark:text-white/20 text-center px-4 py-10">No chats yet</p>
    {/if}
    {#each sessions as s (s.id)}
      <div class="group relative">
        <button
          onclick={() => onLoadSession(s.id)}
          class="w-full text-left flex items-center gap-2.5 px-3 py-2 pr-8 rounded-xl text-[13px]
                 transition-all duration-150 border
                 {s.id === currentId
                   ? 'bg-violet-600/10 dark:bg-violet-600/15 text-violet-700 dark:text-white border-violet-400/20 dark:border-violet-500/20'
                   : 'text-black/40 dark:text-white/40 hover:text-black/80 dark:hover:text-white/80 hover:bg-black/4 dark:hover:bg-white/4 border-transparent'}"
        >
          <svg class="shrink-0 opacity-50" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
          <span class="truncate">{s.title}</span>
        </button>

        <!-- Delete button (appears on hover) -->
        <button
          onclick={(e) => { e.stopPropagation(); onDeleteSession(s.id); }}
          title="Delete chat"
          class="absolute right-1.5 top-1/2 -translate-y-1/2
                 w-6 h-6 rounded-lg flex items-center justify-center
                 opacity-0 group-hover:opacity-100 transition-all duration-150
                 text-black/25 dark:text-white/25
                 hover:text-rose-500 dark:hover:text-rose-400
                 hover:bg-rose-50 dark:hover:bg-rose-500/10"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/>
          </svg>
        </button>
      </div>
    {/each}
  </nav>

  <!-- Footer -->
  <div class="p-3 border-t border-black/6 dark:border-white/6">
    <button
      onclick={onClearAll}
      class="w-full mono text-[10px] tracking-wide
             text-zinc-800 dark:text-white
             hover:text-rose-500 dark:hover:text-rose-400
             border border-zinc-300 dark:border-white
             hover:border-rose-400/30 dark:hover:border-rose-500/30
             rounded-lg py-2 transition-all duration-150"
    >Clear all</button>
  </div>
</aside>