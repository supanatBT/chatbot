# Frontend Integration Guide — Movex Sales AI Chatbot

คู่มือสำหรับ frontend developer ที่ต้องการเชื่อมต่อกับ chatbot backend

---

## ภาพรวม

Backend รันด้วย **Gradio** ซึ่ง expose REST API ให้เรียกได้เลย
ไม่ต้องเขียน backend เพิ่ม — เรียก Gradio API ตรง ๆ จาก frontend ได้เลย

```
Frontend (React / Vue / etc.)
        ↕ HTTP / WebSocket
Gradio API (localhost:7860 หรือ share URL)
        ↕
Chatbot Pipeline (Gemini + RAG + Vision)
```

---

## 1. เริ่มต้น Backend

```bash
python chatbot_v4.py
```

Backend จะรันที่:
- **Local:** `http://localhost:7860`
- **Public (share):** URL จะแสดงใน terminal เช่น `https://xxxx.gradio.live`

---

## 2. API Endpoint

### 2.1 เรียกผ่าน Gradio Client (แนะนำ)

ติดตั้ง:
```bash
npm install @gradio/client
```

ตัวอย่าง (JavaScript):
```js
import { Client } from "@gradio/client";

const client = await Client.connect("http://localhost:7860");

const result = await client.predict("/chat_interaction", {
  user_text: "แนะนำโซ่ที่รับโหลดได้มากกว่า 5000N",
  img_input: null,          // null ถ้าไม่มีรูป
  chat_history: [],         // ประวัติการสนทนา (array)
  last_pids: [],
  session_id: "",
  last_vit_signal: null,
  last_graph_result: null,
});

console.log(result.data);
```

---

### 2.2 เรียกผ่าน REST API โดยตรง

```
POST http://localhost:7860/api/predict
Content-Type: application/json
```

Request body:
```json
{
  "fn_index": 0,
  "data": [
    "user_text",
    null,
    [],
    [],
    "",
    null,
    null
  ]
}
```

> ดู fn_index ที่ถูกต้องได้ที่ `http://localhost:7860/info`

---

## 3. Request Format

### Input Parameters (ตามลำดับ)

| # | ชื่อ | Type | คำอธิบาย |
|---|------|------|----------|
| 1 | `user_text` | `string` | ข้อความที่ user พิมพ์ |
| 2 | `img_input` | `File \| null` | รูปภาพ (PIL → base64 ใน Gradio client) หรือ `null` |
| 3 | `chat_history` | `array` | ประวัติ chat (**ต้องส่งกลับทุกรอบ**) |
| 4 | `last_pids` | `array` | product IDs จาก turn ก่อน (ส่ง `[]` ครั้งแรก) |
| 5 | `session_id` | `string` | session ID (ส่ง `""` ครั้งแรก ระบบจะสร้างให้) |
| 6 | `last_vit_signal` | `any \| null` | carry-over จาก turn ก่อน (ส่ง `null` ครั้งแรก) |
| 7 | `last_graph_result` | `any \| null` | carry-over จาก turn ก่อน (ส่ง `null` ครั้งแรก) |

---

## 4. Response Format

### Output (ตามลำดับ)

| # | ชื่อ | Type | คำอธิบาย |
|---|------|------|----------|
| 1 | `chatbot_ui` | `array` | chat history สำหรับแสดงผล |
| 2 | `chat_state` | `array` | **เก็บไว้ส่งใน turn ถัดไป** |
| 3 | `raw_refs` | `string` | Markdown debug info (retrieval scores) |
| 4 | `last_pids_state` | `array` | **เก็บไว้ส่งใน turn ถัดไป** |
| 5 | `session_state` | `string` | **เก็บไว้ส่งใน turn ถัดไป** |
| 6 | `last_vit_state` | `any` | **เก็บไว้ส่งใน turn ถัดไป** |
| 7 | `last_graph_state` | `any` | **เก็บไว้ส่งใน turn ถัดไป** |

### chat_history format

```json
[
  { "role": "user",      "content": "แนะนำโซ่ series 820" },
  { "role": "assistant", "content": "แนะนำ LF820_K325 ครับ ..." }
]
```

### กรณีที่มี Datasheet (Drawing)

text และ image จะถูกส่งเป็น **2 message แยกกัน**:

```json
[
  { "role": "user",      "content": "ขอ drawing LF820_K325" },
  { "role": "assistant", "content": "นี่คือ datasheet ของ LF820_K325 ครับ ..." },
  { "role": "assistant", "content": "<img src=\"data:image/png;base64,...\" />" }
]
```

ตรวจสอบว่า message เป็นรูปภาพได้จาก:
```js
const isImage = msg.role === "assistant" && msg.content.startsWith("<img");
```

> **หมายเหตุ:** `content` ที่เป็น `<img>` ต้อง render ด้วย `dangerouslySetInnerHTML` (React) หรือ `v-html` (Vue)

---

## 5. ตัวอย่าง Implementation (React)

```jsx
import { useState } from "react";
import { Client } from "@gradio/client";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState(null);

  // State ที่ต้องส่งกลับทุก turn
  const [chatState, setChatState] = useState([]);
  const [lastPids, setLastPids] = useState([]);
  const [sessionId, setSessionId] = useState("");
  const [lastVit, setLastVit] = useState(null);
  const [lastGraph, setLastGraph] = useState(null);

  const sendMessage = async () => {
    if (!input.trim() && !image) return;

    const client = await Client.connect("http://localhost:7860");

    const result = await client.predict("/chat_interaction", {
      user_text: input,
      img_input: image,       // File object จาก <input type="file">
      chat_history: chatState,
      last_pids: lastPids,
      session_id: sessionId,
      last_vit_signal: lastVit,
      last_graph_result: lastGraph,
    });

    const [newChatUI, newChatState, , newPids, newSession, newVit, newGraph] = result.data;

    setMessages(newChatUI);
    setChatState(newChatState);
    setLastPids(newPids);
    setSessionId(newSession);
    setLastVit(newVit);
    setLastGraph(newGraph);
    setInput("");
    setImage(null);
  };

  return (
    <div>
      {messages.map((msg, i) => (
        <div key={i} className={msg.role}>
          {/* ใช้ dangerouslySetInnerHTML เพราะ content อาจมี <img> datasheet */}
          <div dangerouslySetInnerHTML={{ __html: msg.content }} />
        </div>
      ))}

      <input value={input} onChange={e => setInput(e.target.value)} />
      <input type="file" accept="image/*" onChange={e => setImage(e.target.files[0])} />
      <button onClick={sendMessage}>ส่ง</button>
    </div>
  );
}
```

---

## 6. การส่งรูปภาพ

Gradio Client จัดการ encode/decode ให้อัตโนมัติ — ส่ง `File` object ตรง ๆ:

```js
// จาก <input type="file">
const file = e.target.files[0];
await client.predict("/chat_interaction", {
  user_text: "ดูรูปนี้คือสินค้าอะไร",
  img_input: file,   // ← File object
  ...
});
```

ถ้าเรียกผ่าน REST API ตรง ๆ ต้อง upload file ก่อน:

```js
// 1. Upload file
const form = new FormData();
form.append("files", file);
const uploadRes = await fetch("http://localhost:7860/upload", {
  method: "POST",
  body: form,
});
const [{ name: tmpPath }] = await uploadRes.json();

// 2. ใช้ path ใน predict
fetch("http://localhost:7860/api/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    fn_index: 0,
    data: ["ดูรูปนี้", { path: tmpPath }, [], [], "", null, null]
  })
});
```

---

## 7. Chatbot Capabilities (สิ่งที่ chatbot ทำได้)

| ความสามารถ | ตัวอย่าง query |
|------------|----------------|
| ค้นหาสินค้าจากข้อความ | "โซ่ plate width ไม่เกิน 80mm" |
| ค้นหาสินค้าจากรูปภาพ | อัปโหลดรูปโซ่/สเปรอคเก็ต |
| เปรียบเทียบสินค้า | "เปรียบเทียบ LF820_K325 กับ LF820_K750" |
| กรองตาม spec | "รับโหลดมากกว่า 8000N" |
| แนะนำ compatible parts | "สเปรอคเก็ตที่ใช้กับ LF820_K325 ได้" |
| สร้าง datasheet | "ขอ drawing ของ LF820_K325" |
| สรุปสินค้าทั้งหมด | "มีโซ่ series อะไรบ้าง" |

---

## 8. ล้าง Session

เมื่อ user กดปุ่ม "เริ่มใหม่" ให้ reset state ทั้งหมดกลับเป็นค่าเริ่มต้น:

```js
const resetChat = () => {
  setMessages([]);
  setChatState([]);
  setLastPids([]);
  setSessionId("");
  setLastVit(null);
  setLastGraph(null);
};
```

---

## 9. CORS (สำหรับ local dev)

ถ้า frontend รันคนละ port ต้องแก้ `demo.launch()` ใน `chatbot_v4.py`:

```python
demo.launch(
    share=True,
    debug=True,
    server_name="0.0.0.0",   # รับ request จากทุก host
)
```

หรือใช้ proxy ใน Vite/Next.js:

```js
// vite.config.js
export default {
  server: {
    proxy: {
      "/api": "http://localhost:7860",
      "/upload": "http://localhost:7860",
    }
  }
}
```

---

## 10. Error Handling

Response ปกติจะมี `content` เป็น text เสมอ
ถ้า backend error chatbot จะตอบกลับ:

```
❌ เกิดข้อผิดพลาด: <error message>
```

ให้ handle ฝั่ง frontend ด้วยการตรวจสอบว่า content ขึ้นต้นด้วย `❌` หรือไม่
