<script lang="ts">
  interface Props {
    sidebarOpen: boolean;
    isDark: boolean;
    onToggleSidebar: () => void;
    onNewChat: () => void;
    onToggleTheme: () => void;
  }

  let { sidebarOpen, isDark, onToggleSidebar, onNewChat, onToggleTheme }: Props = $props();
</script>

<header
  class="h-14 flex items-center justify-between px-5 flex-shrink-0
         border-b border-black/[.08] dark:border-white/[.06]
         bg-white/80 dark:bg-[#09090d]/80 backdrop-blur-xl"
>
  <div class="flex items-center gap-3">
    <button
      onclick={onToggleSidebar}
      class="w-8 h-8 rounded-lg flex items-center justify-center
             text-black/30 dark:text-white/30
             hover:text-black/70 dark:hover:text-white/80
             hover:bg-black/[.05] dark:hover:bg-white/[.05] transition-all"
      title="Toggle sidebar"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
        <path d="M9 3v18"/>
      </svg>
    </button>

    <a
      href="/"
      onclick={(e) => { e.preventDefault(); onNewChat(); }}
      class="flex items-center gap-2 no-underline"
    >
      <div class="w-2 h-2 rounded-full bg-violet-500 shadow-[0_0_8px_2px_rgba(139,92,246,.6)]"></div>
      <span class="text-[15px] font-semibold tracking-tight text-black/80 dark:text-white/90">AI Chat</span>
    </a>
  </div>

  <div class="flex items-center gap-3">
    <span class="mono text-[10px] tracking-widest text-black/25 dark:text-white/20 uppercase">Gemini</span>

    <button
      onclick={onToggleTheme}
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      class="w-8 h-8 rounded-lg flex items-center justify-center
             text-black/40 dark:text-white/40
             hover:text-black/80 dark:hover:text-white/80
             hover:bg-black/[.05] dark:hover:bg-white/[.05] transition-all"
    >
      {#if isDark}
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="5"/>
          <line x1="12" y1="1" x2="12" y2="3"/>
          <line x1="12" y1="21" x2="12" y2="23"/>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
          <line x1="1" y1="12" x2="3" y2="12"/>
          <line x1="21" y1="12" x2="23" y2="12"/>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
      {:else}
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
      {/if}
    </button>
  </div>
</header>