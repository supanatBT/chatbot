import { Client } from '@gradio/client';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatResponse {
  chatbotUi: ChatMessage[];
  chatState: ChatMessage[];
  rawRefs: string;
  lastPidsState: string[];
  sessionState: string;
  lastVitState: any;
  lastGraphState: any;
}

export interface ChatState {
  chatHistory: ChatMessage[];
  lastPids: string[];
  sessionId: string;
  lastVit: any;
  lastGraph: any;
}

// Initialize client (lazy load)
let clientInstance: Client | null = null;

async function getClient(): Promise<Client> {
  if (!clientInstance) {
    const apiUrl = import.meta.env.VITE_GRADIO_URL || 'http://localhost:7860';
    clientInstance = await Client.connect(apiUrl);
  }
  return clientInstance;
}

/**
 * Send a message to the chatbot and get a response
 * @param userText - The user's message
 * @param state - Current chat state (history, PIDs, session ID, etc.)
 * @param imageFile - Optional image file to send with the message
 * @returns Updated chat state and response
 */
export async function sendChatMessage(
  userText: string,
  state: ChatState,
  imageFile?: File
): Promise<ChatState & { response: ChatResponse }> {
  try {
    const client = await getClient();

    console.log('🔄 Sending to Gradio:', {
      userText,
      chatHistoryLength: state.chatHistory.length,
      sessionId: state.sessionId,
    });

    // Gradio Blocks ต้องส่งเป็น positional arguments (array) ไม่ใช่ keyword arguments
    // order ต้องตรงกับ _inputs ใน chatbot_v4.py:
    // [txt_input, img_input_ui, chat_state, last_pids_state, session_state, last_vit_state, last_graph_state]
    const result = await client.predict('chat_interaction', [
      userText,                    // txt_input
      imageFile || null,           // img_input_ui
      state.chatHistory,           // chat_state
      state.lastPids,              // last_pids_state
      state.sessionId,             // session_state
      state.lastVit,               // last_vit_state
      state.lastGraph,             // last_graph_state
    ]);

    console.log('✅ Response from Gradio:', {
      dataLength: result.data?.length,
      chatbotUiLength: result.data?.[0]?.length,
    });

    const [
      chatbotUi,
      chatStateOut,
      rawRefs,
      lastPidsState,
      sessionState,
      lastVitState,
      lastGraphState,
    ] = result.data as any[];

    // Convert response into unified ChatMessage format
    const formattedMessages: ChatMessage[] = (chatbotUi || []).map((msg: any) => ({
      role: msg.role || 'assistant',
      content: msg.content || '',
    }));

    return {
      chatHistory: chatStateOut || state.chatHistory,
      lastPids: lastPidsState || [],
      sessionId: sessionState || '',
      lastVit: lastVitState,
      lastGraph: lastGraphState,
      response: {
        chatbotUi: formattedMessages,
        chatState: chatStateOut,
        rawRefs,
        lastPidsState,
        sessionState,
        lastVitState,
        lastGraphState,
      },
    };
  } catch (error) {
    console.error('❌ Chat service error:', error);
    const errorMsg = error instanceof Error 
      ? error.message 
      : typeof error === 'object' && error !== null && 'detail' in error
        ? (error as any).detail
        : 'Unknown error';
    
    throw new Error(`Failed to connect to backend: ${errorMsg}\n\nMake sure:\n1. Backend is running: python chatbot_v4.py\n2. Port 7860 is available\n3. No firewall blocking localhost:7860`);
  }
}

/**
 * Reset the chat state for a new conversation
 */
export function createNewChatState(): ChatState {
  return {
    chatHistory: [],
    lastPids: [],
    sessionId: '',
    lastVit: null,
    lastGraph: null,
  };
}

/**
 * Check if a message content is an image (HTML img tag)
 */
export function isImageContent(content: string): boolean {
  return content.startsWith('<img');
}

/**
 * Disconnect the client
 */
export async function disconnectClient(): Promise<void> {
  if (clientInstance) {
    // Gradio client doesn't have explicit disconnect, but we can nullify it
    clientInstance = null;
  }
}
