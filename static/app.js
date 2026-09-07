/* ==========================================================================
   RESEARCH AGENT WEB DASHBOARD CLIENT SCRIPT
   ========================================================================== */

let activeSessionId = localStorage.getItem("active_session_id") || generateUUID();
let activeSessions = JSON.parse(localStorage.getItem("saved_sessions") || "[]");

document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  renderSessionList();
  selectSession(activeSessionId);
});

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

function setupEventListeners() {
  const sendBtn = document.getElementById("btn-send");
  const promptInput = document.getElementById("prompt-input");
  const newChatBtn = document.getElementById("btn-new-chat");
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");

  sendBtn.addEventListener("click", handleSendPrompt);
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendPrompt();
    }
  });

  newChatBtn.addEventListener("click", createNewSession);

  // File Drag and Drop
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      uploadDocument(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      uploadDocument(e.target.files[0]);
    }
  });
}

function useSuggestion(promptText) {
  document.getElementById("prompt-input").value = promptText;
  handleSendPrompt();
}

async function handleSendPrompt() {
  const inputEl = document.getElementById("prompt-input");
  const prompt = inputEl.value.trim();
  if (!prompt) return;

  inputEl.value = "";

  // Hide empty state
  const emptyState = document.getElementById("empty-state");
  if (emptyState) emptyState.remove();

  // Append User Message
  appendMessage("user", prompt);

  // Append Loading State
  const loadingRow = appendLoadingIndicator();

  try {
    const response = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        session_id: activeSessionId
      })
    });

    const data = await response.json();
    loadingRow.remove();

    if (response.ok) {
      activeSessionId = data.session_id;
      saveSessionToList(activeSessionId, prompt);
      appendMessage("agent", data.answer);
    } else {
      appendMessage("agent", `⚠️ Error: ${data.detail || "Server error"}`);
    }
  } catch (err) {
    loadingRow.remove();
    appendMessage("agent", `⚠️ Network connection error: ${err.message}`);
  }
}

function appendMessage(role, text) {
  const feed = document.getElementById("chat-feed");

  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (role === "agent") {
    bubble.innerHTML = marked.parse(text);
  } else {
    bubble.textContent = text;
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  feed.appendChild(row);
  feed.scrollTop = feed.scrollHeight;
  return row;
}

function appendLoadingIndicator() {
  const feed = document.getElementById("chat-feed");
  const row = document.createElement("div");
  row.className = "msg-row agent";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<span style="opacity: 0.7;">🔍 Researching & processing query...</span>`;

  row.appendChild(avatar);
  row.appendChild(bubble);
  feed.appendChild(row);
  feed.scrollTop = feed.scrollHeight;
  return row;
}

async function uploadDocument(file) {
  const dropZone = document.getElementById("drop-zone");
  const originalText = dropZone.querySelector(".drop-text").textContent;
  dropZone.querySelector(".drop-text").textContent = `Uploading ${file.name}...`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/ingest", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (res.ok) {
      alert(`✅ ${data.message}\nTotal indexed chunks: ${data.total_chunks_indexed}`);
    } else {
      alert(`❌ Upload failed: ${data.detail}`);
    }
  } catch (err) {
    alert(`❌ Upload error: ${err.message}`);
  } finally {
    dropZone.querySelector(".drop-text").textContent = originalText;
  }
}

function createNewSession() {
  activeSessionId = generateUUID();
  selectSession(activeSessionId);
}

function selectSession(sessionId) {
  activeSessionId = sessionId;
  localStorage.setItem("active_session_id", sessionId);
  document.getElementById("current-session-id-display").textContent = sessionId.slice(0, 8);

  const feed = document.getElementById("chat-feed");
  feed.innerHTML = `
    <div class="empty-state" id="empty-state">
      <div class="empty-icon">🧠</div>
      <div class="empty-title">How can I assist your research today?</div>
      <p style="font-size: 0.9rem; max-width: 500px;">
        Ask questions, search internal documents via FAISS RAG, or retrieve live real-time API data.
      </p>
      <div class="prompt-suggestions">
        <div class="suggestion-card" onclick="useSuggestion('Summarize the RAG architecture from internal documents')">
          📖 <strong>Internal Knowledge</strong><br>
          Summarize RAG architecture from local docs
        </div>
        <div class="suggestion-card" onclick="useSuggestion('Get the live current weather in Tokyo and Jodhpur')">
          🌤 <strong>Live Weather API</strong><br>
          Get current weather in Tokyo & Jodhpur
        </div>
        <div class="suggestion-card" onclick="useSuggestion('Search Wikipedia for Quantum Computing concepts')">
          🌐 <strong>Wikipedia Search</strong><br>
          Search Wikipedia for Quantum Computing
        </div>
      </div>
    </div>
  `;

  loadSessionHistory(sessionId);
  renderSessionList();
}

async function loadSessionHistory(sessionId) {
  try {
    const res = await fetch(`/api/session/${sessionId}`);
    if (res.ok) {
      const data = await res.json();
      if (data.history && data.history.length > 0) {
        const emptyState = document.getElementById("empty-state");
        if (emptyState) emptyState.remove();

        data.history.forEach(item => {
          appendMessage(item.role === "user" ? "user" : "agent", item.content);
        });
      }
    }
  } catch (e) {
    console.log("No remote session history found.");
  }
}

function saveSessionToList(sessionId, previewTitle) {
  if (!activeSessions.some(s => s.id === sessionId)) {
    activeSessions.unshift({
      id: sessionId,
      title: previewTitle.length > 25 ? previewTitle.slice(0, 25) + "..." : previewTitle
    });
    localStorage.setItem("saved_sessions", JSON.stringify(activeSessions));
    renderSessionList();
  }
}

function renderSessionList() {
  const container = document.getElementById("sessions-list");
  container.innerHTML = "";

  activeSessions.forEach(s => {
    const item = document.createElement("div");
    item.className = `session-item ${s.id === activeSessionId ? 'active' : ''}`;
    item.onclick = () => selectSession(s.id);

    const titleSpan = document.createElement("span");
    titleSpan.textContent = s.title;

    const delBtn = document.createElement("button");
    delBtn.className = "btn-del-session";
    delBtn.innerHTML = "✖";
    delBtn.onclick = (e) => {
      e.stopPropagation();
      deleteSession(s.id);
    };

    item.appendChild(titleSpan);
    item.appendChild(delBtn);
    container.appendChild(item);
  });
}

async function deleteSession(sessionId) {
  activeSessions = activeSessions.filter(s => s.id !== sessionId);
  localStorage.setItem("saved_sessions", JSON.stringify(activeSessions));

  try {
    await fetch(`/api/session/${sessionId}`, { method: "DELETE" });
  } catch (e) {}

  if (activeSessionId === sessionId) {
    createNewSession();
  } else {
    renderSessionList();
  }
}
