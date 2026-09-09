/* ==========================================================================
   BMW M MOTORSPORT RESEARCH AGENT WEB DASHBOARD CLIENT
   ========================================================================== */

let activeSessionId = localStorage.getItem("active_session_id") || generateUUID();
let activeSessions = JSON.parse(localStorage.getItem("saved_sessions") || "[]");

document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  checkSystemHealth();
  renderSessionList();
  loadToolsList();
  loadIngestedDocuments();
  selectSession(activeSessionId);
});

async function checkSystemHealth() {
  const badgeText = document.querySelector(".status-badge span");
  const statusDot = document.querySelector(".status-dot");
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      if (badgeText) badgeText.textContent = `SYSTEM ONLINE (v${data.version || '1.0'})`;
      if (statusDot) statusDot.style.background = "#00FF66";
    } else {
      if (badgeText) badgeText.textContent = "SYSTEM DEGRADED";
      if (statusDot) statusDot.style.background = "#FFCC00";
    }
  } catch (err) {
    if (badgeText) badgeText.textContent = "SYSTEM OFFLINE";
    if (statusDot) statusDot.style.background = "#FF2D55";
  }
}

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
      uploadDocuments(e.dataTransfer.files);
    }
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      uploadDocuments(e.target.files);
    }
  });

  // Agents Dropdown Popover Listener
  const agentsPill = document.getElementById("agents-pill");
  const agentsDropdown = document.getElementById("agents-dropdown");

  if (agentsPill && agentsDropdown) {
    agentsPill.addEventListener("click", (e) => {
      e.stopPropagation();
      agentsDropdown.classList.toggle("active");
    });

    document.addEventListener("click", (e) => {
      if (!agentsPill.contains(e.target)) {
        agentsDropdown.classList.remove("active");
      }
    });
  }
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
    appendMessage("agent", `⚠️ Connection error: ${err.message}`);
  }
}

function appendMessage(role, text) {
  const feed = document.getElementById("chat-feed");

  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "USR" : "///M";

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
  avatar.textContent = "///M";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<span style="opacity: 0.7; font-family: var(--font-display); letter-spacing: 1px; text-transform: uppercase;">[EXECUTING TELEMETRY & VECTOR SEARCH...]</span>`;

  row.appendChild(avatar);
  row.appendChild(bubble);
  feed.appendChild(row);
  feed.scrollTop = feed.scrollHeight;
  return row;
}

async function uploadDocuments(filesList) {
  const files = Array.from(filesList);
  if (files.length === 0) return;

  const dropZone = document.getElementById("drop-zone");
  const originalText = dropZone.querySelector(".drop-text").textContent;
  dropZone.querySelector(".drop-text").textContent = `INGESTING ${files.length} FILE(S)...`;

  const formData = new FormData();
  files.forEach(file => {
    formData.append("files", file);
  });

  try {
    const res = await fetch("/api/ingest", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (res.ok) {
      alert(`✅ ${data.message}\nTotal indexed chunks: ${data.total_chunks_indexed}`);
      loadIngestedDocuments();
    } else {
      alert(`❌ Ingestion failed: ${data.detail}`);
    }
  } catch (err) {
    alert(`❌ Upload error: ${err.message}`);
  } finally {
    dropZone.querySelector(".drop-text").textContent = originalText;
    const fileInput = document.getElementById("file-input");
    if (fileInput) fileInput.value = "";
  }
}

async function loadToolsList() {
  const container = document.getElementById("tools-list");
  if (!container) return;
  try {
    const res = await fetch("/api/tools");
    if (res.ok) {
      const data = await res.json();
      container.innerHTML = "";
      if (data.tools && data.tools.length > 0) {
        data.tools.forEach(tool => {
          const item = document.createElement("div");
          item.className = "session-item";
          item.style.cursor = "default";
          item.style.display = "flex";
          item.style.flexDirection = "column";
          item.style.gap = "2px";

          item.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
              <span style="font-weight: 600; color: #FFFFFF; font-size: 0.75rem;">🟢 ${tool.name}</span>
              <span style="font-size: 0.60rem; background: rgba(0,102,177,0.2); border: 1px solid var(--primary); color: var(--primary); padding: 1px 4px; font-family: var(--font-display);">${tool.badge}</span>
            </div>
            <span style="font-size: 0.65rem; color: var(--muted); line-height: 1.2;">${tool.description}</span>
          `;
          container.appendChild(item);
        });
      }
    }
  } catch (e) {
    console.log("Could not load active agent modules.");
  }
}

async function loadIngestedDocuments() {
  const container = document.getElementById("documents-list");
  if (!container) return;
  try {
    const res = await fetch("/api/documents");
    if (res.ok) {
      const data = await res.json();
      container.innerHTML = "";
      if (data.documents && data.documents.length > 0) {
        data.documents.forEach(doc => {
          const item = document.createElement("div");
          item.className = "session-item";
          item.style.cursor = "default";

          const ext = doc.split('.').pop().toLowerCase();
          let icon = "📄";
          if (ext === "pdf") icon = "📕";
          if (ext === "md") icon = "📝";

          item.innerHTML = `<span style="font-size: 0.78rem; text-transform: uppercase;">${icon} ${doc}</span>`;
          container.appendChild(item);
        });
      } else {
        container.innerHTML = `<div style="font-size: 0.72rem; color: var(--muted); padding: 0.5rem; text-transform: uppercase;">NO DOCUMENTS INGESTED</div>`;
      }
    }
  } catch (e) {
    console.log("Could not load ingested documents.");
  }
}

function createNewSession() {
  activeSessionId = generateUUID();
  selectSession(activeSessionId);
}

function selectSession(sessionId) {
  activeSessionId = sessionId;
  localStorage.setItem("active_session_id", sessionId);
  document.getElementById("current-session-id-display").textContent = sessionId.slice(0, 8).toUpperCase();

  const feed = document.getElementById("chat-feed");
  feed.innerHTML = `
    <div class="empty-state" id="empty-state">
      <div class="empty-headline">AUTONOMOUS RESEARCH CORE</div>
      <p class="empty-subtext">
        High-performance research platform with real-time external API tools and FAISS vector intelligence.
      </p>
      <div class="prompt-suggestions">
        <div class="suggestion-card" onclick="useSuggestion('Summarize the RAG architecture from internal documents')">
          <div class="suggestion-tag">VECTOR DATABASE</div>
          <div class="suggestion-title">Summarize RAG architecture from internal documents</div>
        </div>
        <div class="suggestion-card" onclick="useSuggestion('Get the live current weather in Tokyo and Jodhpur')">
          <div class="suggestion-tag">LIVE METRICS API</div>
          <div class="suggestion-title">Get current weather telemetry in Tokyo & Jodhpur</div>
        </div>
        <div class="suggestion-card" onclick="useSuggestion('Search Wikipedia for Quantum Computing concepts')">
          <div class="suggestion-tag">GLOBAL KNOWLEDGE</div>
          <div class="suggestion-title">Search Wikipedia for Quantum Computing concepts</div>
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
      title: previewTitle.length > 22 ? previewTitle.slice(0, 22).toUpperCase() + "..." : previewTitle.toUpperCase()
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
    delBtn.innerHTML = "✕";
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
