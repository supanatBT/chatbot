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

    // แปลง File → Blob ก่อนส่ง Gradio (gr.Image รับ Blob/URL ไม่รับ File object)
    let imagePayload: Blob | null = null;
    if (imageFile) {
      imagePayload = new Blob([await imageFile.arrayBuffer()], { type: imageFile.type });
    }

    const result = await client.predict('chat_interaction', [
      userText,                    // txt_input
      imagePayload,                // img_input_ui
      state.chatHistory,           // chat_state
      state.lastPids,              // last_pids_state
      state.sessionId,             // session_state
      state.lastVit,               // last_vit_state
      state.lastGraph,             // last_graph_state
    ]);

    console.log('✅ Response from Gradio:', {
      dataLength: (result.data as any[])?.length,
      chatbotUiLength: ((result.data as any[])?.[0])?.length,
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

export function createNewChatState(): ChatState {
  return {
    chatHistory: [],
    lastPids: [],
    sessionId: '',
    lastVit: null,
    lastGraph: null,
  };
}

export function isImageContent(content: string): boolean {
  return content.startsWith('<img');
}

export async function disconnectClient(): Promise<void> {
  if (clientInstance) {
    clientInstance = null;
  }
}