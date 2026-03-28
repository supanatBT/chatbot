<script lang="ts">
  interface Message { id: string; role: 'user' | 'assistant'; content: string; timestamp?: number; image?: string }

  interface Props {
    messages: Message[];
    isLoading: boolean;
    el?: HTMLDivElement;
    onRegenerate: () => void;
    onCopy: (content: string, btn: HTMLButtonElement) => void;
  }

  let { messages, isLoading, el = $bindable(), onRegenerate, onCopy }: Props = $props();

  let selectedImage: string | null = $state(null);

  function formatTime(timestamp?: number): string {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    
    if (isToday) {
      return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
    }
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }

  function shouldShowTimestamp(i: number): boolean {
    if (i === 0) return true;
    const prevMsg = messages[i - 1];
    const currMsg = messages[i];
    if (!prevMsg.timestamp || !currMsg.timestamp) return false;
    return Math.abs(currMsg.timestamp - prevMsg.timestamp) > 5 * 60 * 1000; // 5 minutes
  }

  function isImageContent(content: string): boolean {
    return content.startsWith('<img');
  }

  function extractImageSrc(htmlContent: string): string | null {
    const match = htmlContent.match(/src="([^"]+)"/);
    return match ? match[1] : null;
  }

  function openImageModal(htmlContent: string) {
    const src = extractImageSrc(htmlContent);
    if (src) {
      selectedImage = src;
    }
  }

  function closeImageModal() {
    selectedImage = null;
  }
</script>

<div bind:this={el} class="flex-1 overflow-y-auto px-5 py-8">
  <div class="max-w-2xl mx-auto">
    {#each messages as msg, i (msg.id)}
    

      <div class="message-row mb-6 anim-up" style="animation-delay:{i * 0.04}s">

        <!-- User bubble -->
        {#if msg.role === 'user'}
          <div class="flex flex-col items-end gap-1.5">
            <span class="mono text-[9px] uppercase tracking-widest text-black/35 dark:text-white/25 font-medium">You</span>
            
            <!-- Image thumbnail (if present) -->
            {#if msg.image}
              <button
                onclick={() => {
                  if (msg.image) {
                    selectedImage = msg.image;
                  }
                }}
                class="max-w-[82%] rounded-2xl rounded-tr-sm overflow-hidden shadow-sm border border-violet-300 dark:border-violet-500/30
                       hover:shadow-lg hover:border-violet-400 dark:hover:border-violet-500 transition-all cursor-pointer"
                title="Click to view full size"
              >
                <img
                  src={msg.image}
                  alt="User uploaded image"
                  class="w-full h-auto max-h-64 object-cover"
                />
              </button>
            {/if}
            
            <!-- Text content (if present) -->
            {#if msg.content && msg.content !== '📷'}
              <div class="max-w-[82%] bg-violet-600 hover:bg-violet-700 text-white rounded-2xl rounded-tr-sm px-4 py-3 text-[14px] leading-relaxed msg-bubble shadow-sm transition-colors">
                {msg.content}
              </div>
            {/if}
            
            <div class="msg-actions">
              <button
                onclick={(e) => onCopy(msg.content, e.currentTarget)}
                class="mono text-[10px] px-2 py-1 rounded-lg border transition-all
                       text-black/30 dark:text-white/25 border-black/8 dark:border-white/8
                       hover:text-black/70 dark:hover:text-white/60 hover:border-black/15 dark:hover:border-white/15"
              >Copy</button>
            </div>
          </div>

        <!-- Assistant bubble -->
        {:else}
          <div class="flex gap-3 items-start">
            <div class="w-8 h-8 rounded-xl flex items-center justify-center text-violet-500 text-base shrink-0 mt-0.5
                        bg-violet-50 hover:bg-violet-100 dark:bg-[#1a1a24] dark:hover:bg-[#212229] border border-violet-200 dark:border-white/8 transition-colors">✦</div>
            <div class="flex flex-col gap-1.5 flex-1 min-w-0">
              <span class="mono text-[9px] uppercase tracking-widest text-black/35 dark:text-white/25 font-medium">Assistant</span>
              
              <!-- Render image content -->
              {#if isImageContent(msg.content)}
                <button
                  onclick={() => openImageModal(msg.content)}
                  class="cursor-pointer rounded-2xl rounded-tl-sm overflow-hidden shadow-sm border border-black/7 dark:border-white/6
                         hover:shadow-lg hover:border-violet-400 dark:hover:border-violet-500 transition-all"
                  title="Click to view full size"
                >
                  {@html msg.content}
                </button>
              <!-- Render text content -->
              {:else}
                <div class="border rounded-2xl rounded-tl-sm px-4 py-3 text-[14px] leading-relaxed msg-bubble shadow-sm
                            bg-white hover:bg-gray-50 dark:bg-[#141419] dark:hover:bg-[#18181e] border-black/7 dark:border-white/6
                            text-black/80 dark:text-white/85 transition-colors">
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
              {/if}

              {#if msg.content}
                <div class="msg-actions flex gap-1.5">
                  <button
                    onclick={(e) => onCopy(msg.content, e.currentTarget)}
                    class="mono text-[10px] px-2 py-1 rounded-lg border transition-all
                           text-black/30 dark:text-white/25 border-black/8 dark:border-white/8
                           hover:text-black/70 dark:hover:text-white/60 hover:border-black/15 dark:hover:border-white/15"
                  >Copy</button>
                  {#if i === messages.length - 1 && !isLoading}
                    <button
                      onclick={onRegenerate}
                      class="mono text-[10px] px-2 py-1 rounded-lg border transition-all
                             text-black/30 dark:text-white/25 border-black/8 dark:border-white/8
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

<!-- Image Modal -->
{#if selectedImage}
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
    onclick={closeImageModal}
    role="button"
    tabindex="0"
    onkeydown={(e) => e.key === 'Escape' && closeImageModal()}
  >
    <div
      class="relative max-w-4xl max-h-[90vh] bg-white dark:bg-[#141419] rounded-2xl shadow-2xl overflow-hidden"
      onclick={(e) => e.stopPropagation()}
      role="dialog"
    >
      <!-- Close button -->
      <button
        onclick={closeImageModal}
        class="absolute top-4 right-4 z-10 w-10 h-10 rounded-full bg-black/50 hover:bg-black/70 
               text-white flex items-center justify-center transition-colors"
        title="Close (Esc)"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>

      <!-- Image -->
      <img
        src={selectedImage}
        alt="Technical drawing"
        class="w-full h-full object-contain"
      />

      <!-- Download button -->
      <a
        href={selectedImage}
        download
        class="absolute bottom-4 left-4 px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-700 
               text-white text-sm font-medium transition-colors"
        onclick={(e) => e.stopPropagation()}
      >
        ⬇ Download
      </a>
    </div>
  </div>
{/if}