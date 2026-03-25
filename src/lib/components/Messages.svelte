<script lang="ts">
  interface Message { id: string; role: 'user' | 'assistant'; content: string }

  interface Props {
    messages: Message[];
    isLoading: boolean;
    el?: HTMLDivElement;
    onRegenerate: () => void;
    onCopy: (content: string, btn: HTMLButtonElement) => void;
  }

  let { messages, isLoading, el = $bindable(), onRegenerate, onCopy }: Props = $props();
</script>

<div bind:this={el} class="flex-1 overflow-y-auto px-5 py-8">
  <div class="max-w-2xl mx-auto">
    {#each messages as msg, i (msg.id)}
      <div class="message-row mb-8 anim-up" style="animation-delay:{i * 0.04}s">

        <!-- User bubble -->
        {#if msg.role === 'user'}
          <div class="flex flex-col items-end gap-1.5">
            <span class="mono text-[10px] uppercase tracking-widest text-black/25 dark:text-white/20">You</span>
            <div class="max-w-[82%] bg-violet-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 text-[14px] leading-relaxed msg-bubble">
              {msg.content}
            </div>
            <div class="msg-actions">
              <button
                onclick={(e) => onCopy(msg.content, e.currentTarget)}
                class="mono text-[10px] px-2 py-1 rounded-lg border transition-colors
                       text-black/30 dark:text-white/25 border-black/[.08] dark:border-white/[.08]
                       hover:text-black/70 dark:hover:text-white/60"
              >Copy</button>
            </div>
          </div>

        <!-- Assistant bubble -->
        {:else}
          <div class="flex gap-3 items-start">
            <div class="w-8 h-8 rounded-xl flex items-center justify-center text-violet-500 text-base shrink-0 mt-0.5
                        bg-violet-50 dark:bg-[#1a1a24] border border-violet-200 dark:border-white/[.08]">✦</div>
            <div class="flex flex-col gap-1.5 flex-1 min-w-0">
              <span class="mono text-[10px] uppercase tracking-widest text-black/25 dark:text-white/20">Assistant</span>
              <div class="border rounded-2xl rounded-tl-sm px-4 py-3 text-[14px] leading-relaxed msg-bubble
                          bg-white dark:bg-[#141419] border-black/[.07] dark:border-white/[.06]
                          text-black/80 dark:text-white/85">
                {#if msg.content === '' && isLoading}
                  <div class="flex gap-1.5 items-center py-1">
                    <div class="dot-1 w-1.5 h-1.5 rounded-full bg-violet-400"></div>
                    <div class="dot-2 w-1.5 h-1.5 rounded-full bg-violet-400"></div>
                    <div class="dot-3 w-1.5 h-1.5 rounded-full bg-violet-400"></div>
                  </div>
                {:else}
                  {msg.content}
                {/if}
              </div>

              {#if msg.content}
                <div class="msg-actions flex gap-1.5">
                  <button
                    onclick={(e) => onCopy(msg.content, e.currentTarget)}
                    class="mono text-[10px] px-2 py-1 rounded-lg border transition-colors
                           text-black/30 dark:text-white/25 border-black/[.08] dark:border-white/[.08]
                           hover:text-black/70 dark:hover:text-white/60"
                  >Copy</button>
                  {#if i === messages.length - 1 && !isLoading}
                    <button
                      onclick={onRegenerate}
                      class="mono text-[10px] px-2 py-1 rounded-lg border transition-colors
                             text-black/30 dark:text-white/25 border-black/[.08] dark:border-white/[.08]
                             hover:text-violet-600 dark:hover:text-violet-400
                             hover:border-violet-300 dark:hover:border-violet-500/30"
                    >↺ Regen</button>
                  {/if}
                </div>
              {/if}
            </div>
          </div>
        {/if}

      </div>
    {/each}
  </div>
</div>