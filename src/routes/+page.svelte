<script lang="ts">
  import { tick } from 'svelte';
  import Sidebar from '$lib/components/Sidebar.svelte';
  import Header from '$lib/components/Header.svelte';
  import Messages from '$lib/components/Messages.svelte';

  type Role = 'user' | 'assistant';
  interface Message { id: string; role: Role; content: string; timestamp?: number }
  interface Session { id: string; title: string; messages: Message[] }

  // ── State ──────────────────────────────────────────────────────
  let sessions: Session[] = $state(
    JSON.parse(typeof localStorage !== 'undefined' ? localStorage.getItem('chat_sessions') ?? '[]' : '[]')
  );
  let currentId: string | null = $state(
    typeof localStorage !== 'undefined' && JSON.parse(localStorage.getItem('chat_sessions') ?? '[]').length > 0
      ? JSON.parse(localStorage.getItem('chat_sessions') ?? '[]')[0].id
      : null
  );

  // Validate that currentId exists in sessions
  $effect(() => {
    if (currentId && !sessions.find(s => s.id === currentId)) {
      currentId = sessions[0]?.id ?? null;
    }
  });
  let input = $state('');
  let isLoading = $state(false);
  let sidebarOpen = $state(true);
  let isDark = $state(
    typeof localStorage !== 'undefined' ? localStorage.getItem('theme') !== 'light' : true
  );
  let abortController: AbortController | null = null;

  // sync dark class on <html> whenever isDark changes
  $effect(() => {
    const html = document.documentElement;
    html.classList.add('dark-transitioning');
    html.classList.toggle('dark', isDark);
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    
    // Remove transition-blocking class after a frame
    requestAnimationFrame(() => {
      html.classList.remove('dark-transitioning');
    });
  });

  function toggleTheme() { isDark = !isDark; }

  // ── Derived ────────────────────────────────────────────────────
  let currentSession = $derived(sessions.find(s => s.id === currentId) ?? null);
  let messages = $derived(currentSession?.messages ?? []);
  let canSend = $derived(input.trim().length > 0 && !isLoading);

  // ── Refs ───────────────────────────────────────────────────────
  let messagesEl: HTMLDivElement | undefined = $state(undefined);
  let textareaEl: HTMLTextAreaElement;

  // ── Suggestions ────────────────────────────────────────────────
  const suggestions = [
    { label: 'Explain a concept',  prompt: 'What is a serverless function?' },
    { label: 'Summarize an article', prompt: 'Summarize the following for a beginner:\n' },
    { label: 'Draft an email',     prompt: 'Draft an email to my boss about:\n' },
    { label: 'Write code',         prompt: 'Write a TypeScript function that:\n' },
  ];

  // ── Helpers ────────────────────────────────────────────────────
  function uid() { return Math.random().toString(36).slice(2, 9); }

  function save() {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('chat_sessions', JSON.stringify(sessions));
    }
  }

  async function scrollBottom() {
    await tick();
    messagesEl?.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
  }

  function autoResize() {
    if (!textareaEl) return;
    textareaEl.style.height = 'auto';
    textareaEl.style.height = Math.min(textareaEl.scrollHeight, 200) + 'px';
  }

  // ── Session management ─────────────────────────────────────────
  function newChat() {
    currentId = null;
    input = '';
    if (textareaEl) textareaEl.style.height = 'auto';
  }

  function loadSession(id: string) {
    currentId = id;
    scrollBottom();
  }

  function clearAll() {
    if (!confirm('Clear all chat history?')) return;
    sessions = [];
    currentId = null;
    save();
  }

  function deleteSession(id: string) {
    sessions = sessions.filter(s => s.id !== id);
    if (currentId === id) currentId = sessions[0]?.id ?? null;
    save();
  }

  function getOrCreate(firstMsg: string): Session {
    if (currentId) return sessions.find(s => s.id === currentId)!;
    const s: Session = {
      id: uid(),
      title: firstMsg.slice(0, 45) + (firstMsg.length > 45 ? '…' : ''),
      messages: [],
    };
    sessions = [s, ...sessions];
    currentId = s.id;
    save();
    return s;
  }

  // ── Send (streaming) ───────────────────────────────────────────
  async function send() {
    const text = input.trim();
    if (!text || isLoading) return;

    const session = getOrCreate(text);
    const userMsg: Message = { id: uid(), role: 'user', content: text, timestamp: Date.now() };
    
    // Update session with user message
    sessions = sessions.map(s => 
      s.id === session.id 
        ? { ...s, messages: [...s.messages, userMsg] }
        : s
    );

    input = '';
    if (textareaEl) textareaEl.style.height = 'auto';
    await scrollBottom();

    isLoading = true;
    abortController = new AbortController();

    const aiMsg: Message = { id: uid(), role: 'assistant', content: '', timestamp: Date.now() };
    
    // Update session with empty AI message
    sessions = sessions.map(s =>
      s.id === session.id
        ? { ...s, messages: [...s.messages, aiMsg] }
        : s
    );

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: currentSession?.messages.slice(0, -1).map(m => ({ role: m.role, content: m.content })) ?? [],
        }),
        signal: abortController.signal,
      });

      if (res.ok && res.body) {
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          aiMsg.content += dec.decode(value, { stream: true });
          sessions = sessions.map(s =>
            s.id === currentId
              ? { ...s, messages: [...s.messages.slice(0, -1), { ...aiMsg }] }
              : s
          );
          await scrollBottom();
        }
      } else {
        throw new Error('no stream');
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        await mockStream(aiMsg, text);
      } else if (!aiMsg.content) {
        aiMsg.content = '_(generation stopped)_';
        sessions = sessions.map(s =>
          s.id === currentId
            ? { ...s, messages: [...s.messages.slice(0, -1), { ...aiMsg }] }
            : s
        );
      }
    }

    save();
    isLoading = false;
    abortController = null;
    await scrollBottom();
  }

  // ── Mock fallback (remove when /api/chat is ready) ─────────────
  async function mockStream(aiMsg: Message, prompt: string) {
    const text = `This is a mock response for: "${prompt.slice(0, 40)}…"\n\nConnect \`src/routes/api/chat/+server.ts\` to replace this with real streaming output from OpenAI or another provider.`;
    for (let i = 0; i < text.length; i++) {
      aiMsg.content = text.slice(0, i + 1);
      sessions = sessions.map(s =>
        s.id === currentId
          ? { ...s, messages: [...s.messages.slice(0, -1), { ...aiMsg }] }
          : s
      );
      if (i % 4 === 0) {
        await new Promise(r => setTimeout(r, 14));
        await scrollBottom();
      }
    }
  }

  // ── Stop / Regenerate ──────────────────────────────────────────
  function stop() {
    abortController?.abort();
  }

  async function regenerate() {
    if (!currentSession || !currentId) return;
    
    // Remove last AI message if exists
    if (currentSession.messages.at(-1)?.role === 'assistant') {
      sessions = sessions.map(s =>
        s.id === currentId
          ? { ...s, messages: s.messages.slice(0, -1) }
          : s
      );
    }
    
    const last = currentSession.messages.findLast(m => m.role === 'user');
    if (!last) return;

    isLoading = true;
    abortController = new AbortController();
    const aiMsg: Message = { id: uid(), role: 'assistant', content: '', timestamp: Date.now() };
    
    // Add new AI message
    sessions = sessions.map(s =>
      s.id === currentId
        ? { ...s, messages: [...s.messages, aiMsg] }
        : s
    );

    await mockStream(aiMsg, last.content);
    save();
    isLoading = false;
    abortController = null;
  }

  // ── Utilities ──────────────────────────────────────────────────
  async function copyMsg(content: string, btn: HTMLButtonElement) {
    await navigator.clipboard.writeText(content);
    btn.textContent = 'Copied!';
    setTimeout(() => (btn.textContent = 'Copy'), 1600);
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }
</script>

<!-- ─── Fonts + global styles ─────────────────────────── -->
<svelte:head>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=Geist+Mono:wght@300;400;500&display=swap" rel="stylesheet" />
  <style>
    :root { font-family: 'Syne', sans-serif; }
    .mono { font-family: 'Geist Mono', monospace; }
    html, body { height: 100%; overflow: hidden; background: #f8f8fc; }
    html.dark, html.dark body { background: #09090d; }
    ::-webkit-scrollbar-thumb { background: #d0d0e0; border-radius: 3px; }
    html.dark ::-webkit-scrollbar-thumb { background: #2a2a35; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb:hover { background: #b8b8d0; }
    html.dark ::-webkit-scrollbar-thumb:hover { background: #3a3a45; }
    
    /* Disable transitions during dark mode toggle */
    html.dark-transitioning * {
      transition: none !important;
    }
    
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse-dot {
      0%, 80%, 100% { opacity: .2; transform: scale(.75); }
      40%            { opacity: 1;  transform: scale(1); }
    }
    @keyframes slideIn {
      from { opacity: 0; transform: translateX(-8px); }
      to   { opacity: 1; transform: translateX(0); }
    }
    
    .anim-up { animation: fadeUp .35s cubic-bezier(.2,0,.1,1) both; }
    .dot-1 { animation: pulse-dot 1.2s infinite; }
    .dot-2 { animation: pulse-dot 1.2s .2s infinite; }
    .dot-3 { animation: pulse-dot 1.2s .4s infinite; }
    
    .msg-bubble { white-space: pre-wrap; word-break: break-word; }
    .msg-actions { opacity: 0; transition: opacity .2s ease-out; }
    .message-row:hover .msg-actions { opacity: 1; }
    
    /* Better hover effects */
    @media (prefers-reduced-motion: no-preference) {
      button { transition: all .2s cubic-bezier(.2,0,.1,1); }
    }
  </style>
</svelte:head>

<!-- ─── App shell ──────────────────────────────────────── -->
<div class="flex h-screen bg-[#f8f8fc] dark:bg-[#09090d] text-[#1a1a2e] dark:text-[#e2e2ee] overflow-hidden">

  <!-- Sidebar component -->
  <Sidebar
    {sessions}
    {currentId}
    open={sidebarOpen}
    onNewChat={newChat}
    onLoadSession={loadSession}
    onDeleteSession={deleteSession}
    onClearAll={clearAll}
  />

  <div class="flex flex-col flex-1 min-w-0">

    <!-- Header component -->
    <Header
      {sidebarOpen}
      onToggleSidebar={() => sidebarOpen = !sidebarOpen}
      onNewChat={newChat}
      {isDark}
      onToggleTheme={toggleTheme}
    />

    <!-- Messages component (shown only when there are messages) -->
    {#if messages.length > 0}
      <Messages
        bind:el={messagesEl}
        {messages}
        {isLoading}
        onRegenerate={regenerate}
        onCopy={copyMsg}
      />

    <!-- Empty screen (inline, shown when no messages) -->
    {:else}
      <div class="flex-1 overflow-y-auto px-5 py-8">
        <div class="max-w-2xl mx-auto pt-8 anim-up">
          <div class="mb-2">
            <h1 class="text-4xl font-bold tracking-tight text-black/90 dark:text-white/95 mb-2">
              What can I help you with?
            </h1>
            <p class="text-[15px] text-black/50 dark:text-white/40 leading-relaxed">
              Ask me anything – from coding questions to creative writing. Start with one of the suggestions below or type your own.
            </p>
          </div>
        </div>
      </div>
    {/if}

    <!-- Input panel (inline) -->
    <div class="shrink-0 px-5 pb-6 pt-3 bg-linear-to-t from-[#f8f8fc] dark:from-[#09090d] to-transparent">
      <div class="max-w-2xl mx-auto">

        <!-- Stop button -->
        <div class="flex justify-center mb-3 min-h-8">
          {#if isLoading}
            <button
              onclick={stop}
              class="flex items-center gap-2 mono text-[11px]
                     text-black/50 dark:text-white/50
                     hover:text-black/80 dark:hover:text-white/80
                     px-4 py-1.5 rounded-full border
                     border-black/8 dark:border-white/8
                     hover:border-black/20 dark:hover:border-white/20
                     bg-white dark:bg-[#141419] transition-all"
            >
              <span class="w-2 h-2 rounded-sm bg-black/40 dark:bg-white/50 inline-block"></span>
              Stop generating
            </button>
          {/if}
        </div>

        <!-- Input box -->
        <div class="flex items-end gap-2 rounded-2xl px-4 py-2 transition-all
                    bg-white dark:bg-[#141419]
                    border border-black/8 dark:border-white/8
                    focus-within:border-violet-500/60 focus-within:shadow-lg focus-within:shadow-violet-500/10
                    dark:focus-within:shadow-violet-500/20">
          <textarea
            bind:this={textareaEl}
            bind:value={input}
            oninput={autoResize}
            onkeydown={handleKeydown}
            placeholder="Send a message…"
            rows={1}
            class="mono flex-1 bg-transparent border-none outline-none resize-none
                   text-[14px] text-black/85 dark:text-white/85
                   placeholder:text-black/25 dark:placeholder:text-white/20
                   py-2.5 min-h-11 max-h-50 leading-relaxed"
          ></textarea>
          <button
            onclick={send}
            disabled={!canSend}
            title="Send message (Enter)"
            class="w-9 h-9 mb-1 rounded-xl flex items-center justify-center shrink-0
                   transition-all duration-150
                   {canSend
                     ? 'bg-violet-600 hover:bg-violet-500 text-white hover:scale-110 active:scale-95'
                     : 'bg-white/5 text-white/20 cursor-not-allowed'}"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <path d="M5 12h14M12 5l7 7-7 7"/>
            </svg>
          </button>
        </div>

        <p class="mono text-center text-[10px] text-black/20 dark:text-white/15 mt-3 tracking-wide">
          Enter ↵ to send &nbsp;·&nbsp; Shift+Enter for newline &nbsp;·&nbsp; ⌘K new chat
        </p>
      </div>
    </div>

  </div>
</div>

<!-- ─── Global shortcuts ───────────────────────────────── -->
<svelte:window
  onkeydown={(e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); newChat(); }
  }}
/>