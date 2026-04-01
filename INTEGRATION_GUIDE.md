# Frontend-Backend Integration Guide

บทสรุปการเชื่อม Frontend Svelte กับ Backend Gradio โดยทำตามแนวทางใน FRONTEND_GUIDE.md ✅

## 📋 สิ่งที่ได้ทำ

### 1️⃣ ติดตั้ง Gradio Client Package
```bash
npm install @gradio/client
```

### 2️⃣ สร้าง Chat Service (`src/lib/services/chatService.ts`)
- ไฟล์ที่จัดการการเชื่อมต่อ API ทั้งหมด
- รับผิดชอบการส่งข้อความไปยัง Gradio backend
- เก็บ state ต่าง ๆ: chat history, product IDs, session ID, VIT signals, graph results
- ให้ฟังก์ชั่นหลักสำหรับการใช้งาน:
  - `sendChatMessage()` - ส่งข้อความไปบ็อ
  - `createNewChatState()` - สร้าง state ใหม่สำหรับการสนทนาใหม่
  - `isImageContent()` - ตรวจสอบว่า content เป็นรูปภาพ

### 3️⃣ อัพเดท Frontend Components
- **`src/routes/+page.svelte`**
  - เพิ่ม state ใหม่: `backendStates` เพื่อติดตามการเชื่อมต่อ
  - แทนที่ mock API ด้วยการเรียก API จริงจาก Gradio
  - บันทึก state ทุกครั้งหลังจากสนทนา

- **`src/lib/components/Messages.svelte`**
  - สามารถแสดงรูปภาพ HTML (datasheet จาก backend)
  - ใช้ `{@html}` สำหรับ render images จาก base64

### 4️⃣ ตั้งค่า Vite Proxy (`vite.config.ts`)
```ts
server: {
  proxy: {
    '/api': 'http://localhost:7860',
    '/upload': 'http://localhost:7860',
  }
}
```

### 5️⃣ ตั้งค่า Environment Variables (`.env.local`)
```
VITE_GRADIO_URL=http://localhost:7860
```

---

## 🚀 วิธีการใช้งาน

### เริ่มต้นใช้งาน

#### Terminal 1 - Backend:
```bash
cd /Users/jame/Documents/GitHub/chatbot
python chatbot_v4.py
```
Backend จะรันที่: `http://localhost:7860`

#### Terminal 2 - Frontend:
```bash
cd /Users/jame/Documents/GitHub/chatbot
npm run dev
```
Frontend จะรันที่: `http://localhost:5173`

### การทำงาน
1. ผู้ใช้พิมพ์ข้อความ → Frontend ส่งไปยัง Backend
2. Backend ประมวลผล (RAG, Vision, Gemini) → ส่งกลับคำตอบ
3. Frontend แสดงคำตอบ และ **เก็บ state ไว้**
4. ในรอบถัดไป state เก่าจะถูกส่งกลับไปให้ backend รู้บริบท

---

## 📊 State Management (สิ่งที่สำคัญ!)

แต่ละ session มี backend state เก็บ:

```ts
backendStates[sessionId] = {
  chatHistory: [],      // ประวัติการสนทนา
  lastPids: [],         // Product IDs จากการค้นหาครั้งก่อน
  sessionId: "",        // Session ID สำหรับ backend
  lastVit: null,        // ข้อมูล Vision model จากรอบก่อน
  lastGraph: null,      // ข้อมูล Graph จากรอบก่อน
}
```

### ตัวอย่างการส่ง Message

```ts
import { sendChatMessage, createNewChatState } from '$lib/services/chatService';

// Turn 1
let state = createNewChatState();
const result = await sendChatMessage("แนะนำโซ่", state);
state = result; // ได้ state ใหม่กลับมา

// Turn 2: state เก่าจะถูกใช้อัตโนมัติ
const result2 = await sendChatMessage("แนะนำเพิ่มเติม", state);
stage = result2; // ต่อเนื่องไป...
```

---

## 🖼️ แสดงรูปภาพ (Datasheet)

Backend จะส่ง datasheet เป็น 2 messages แยกกัน:
1. Text message (อธิบาย)
2. Image message (`<img src="data:image/png;base64,..." />`)

Frontend จะตรวจสอบ `content.startsWith('<img')` แล้วใช้ `{@html}` เพื่อ render

```svelte
{#if content.startsWith('<img')}
  <div>{@html content}</div>
{:else}
  <div>{content}</div>
{/if}
```

---

## 🔧 Troubleshooting

### ❌ "Failed to send message"
- ตรวจสอบว่า Backend รันอยู่: `python chatbot_v4.py`
- ตรวจสอบ port: `http://localhost:7860`

### ❌ CORS Error
- ถ้าใช้ build production ต้องตั้ง CORS ใน `chatbot_v4.py`:
```python
demo.launch(
    share=True,
    server_name="0.0.0.0",  # อนุญาต cross-origin
)
```

### ❌ Image ไม่แสดง
- ตรวจสอบ Browser console ว่า base64 encode ถูกต้อง
- ตรวจสอบ `<img>` tag มี `src` attribute

---

## 📁 Files ที่ได้แก้ไข/สร้าง

### ✨ Files ที่สร้างใหม่:
- `src/lib/services/chatService.ts` - Chat API service

### 🔄 Files ที่แก้ไข:
- `src/routes/+page.svelte` - Main chat page
- `src/lib/components/Messages.svelte` - Message display component
- `vite.config.ts` - Proxy configuration
- `package.json` - Added @gradio/client

### 📝 Environment:
- `.env.local` - Already configured with backend URL

---

## ✅ Ready to Use!

Frontend ปัจจุบันใช้งานกับ Backend ไม่ต้องทำอะไรเพิ่มเติม!

หากต้องการอัพเดท:
- ตกลง API endpoint ได้ที่ `http://localhost:7860/info`
- ใช้ @gradio/client library ที่ต่อมา
- ทุกอย่าง handle state อัตโนมัติแล้ว 🎉
