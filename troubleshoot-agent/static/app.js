/**
 * ARIA frontend: voice recording, API calls, UI state.
 * Vanilla JS, no framework.
 */

const API_BASE = ''; // same origin when served by FastAPI

function getOrCreateSessionId() {
  try {
    const stored = localStorage.getItem('aria_session_id');
    if (stored) return stored;
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      const id = crypto.randomUUID();
      localStorage.setItem('aria_session_id', id);
      return id;
    }
    return 'aria-' + Date.now() + '-' + Math.random().toString(36).slice(2, 11);
  } catch (_) {
    return 'aria-' + Date.now() + '-' + Math.random().toString(36).slice(2, 11);
  }
}
let sessionId = getOrCreateSessionId();

let mediaRecorder = null;
let audioChunks = [];
let stream = null;
let currentImageBase64 = null;

// --- DOM (filled when ready) ---
let conversation, textInput, sendBtn, holdSpeak, voiceStatus;
let docsUpload, fileInput, docsList, attachImage, imageInput;
let issueBadge, stepsList, sourcesList, escalationAlert, processStepsList;

// --- Opening message ---
function showOpeningMessage() {
  const msg = "Hello, I'm ARIA. What iPhone issue can I help you troubleshoot today?";
  appendAriaMessage(msg, [], null);
}

function setVoiceStatus(text) {
  if (voiceStatus) voiceStatus.textContent = text;
}

// --- Messages ---
function appendUserMessage(text, imageUrl = null) {
  if (!conversation) return;
  const div = document.createElement('div');
  div.className = 'message user';
  let html = `<div class="message-text">${escapeHtml(text)}</div>`;
  if (imageUrl) {
    html += `<img class="image-preview" src="${imageUrl}" alt="Attached" />`;
  }
  div.innerHTML = html;
  conversation.appendChild(div);
  conversation.scrollTop = conversation.scrollHeight;
}

function renderMarkdown(text) {
  if (!text) return '';
  if (typeof marked !== 'undefined') {
    marked.setOptions({ gfm: true, breaks: true });
    return marked.parse(text);
  }
  return escapeHtml(text);
}

function appendAriaMessage(text, sources = [], audioBase64 = null) {
  if (!conversation) return;
  const div = document.createElement('div');
  div.className = 'message aria';
  let html = '<span class="message-avatar">◈</span>';
  html += '<div class="message-text markdown-body">' + renderMarkdown(text) + '</div>';
  if (sources && sources.length) {
    html += '<div class="message-sources">';
    sources.forEach(s => {
      const url = typeof s === 'string' ? s : (s && (s.url || s));
      const href = url && String(url).startsWith('http') ? url : '#';
      const label = typeof s === 'string' ? s : (s && (s.title || s.url) || '');
      html += `<a class="source-pill" href="${escapeHtml(href)}" target="_blank" rel="noopener">${escapeHtml(String(label))}</a>`;
    });
    html += '</div>';
  }
  if (audioBase64) {
    html += '<div class="audio-player-wrap">';
    html += `<audio controls src="data:audio/mpeg;base64,${audioBase64}"></audio>`;
    html += '</div>';
  }
  div.innerHTML = html;
  conversation.appendChild(div);
  conversation.scrollTop = conversation.scrollHeight;
}

function appendTyping() {
  if (!conversation) return;
  const div = document.createElement('div');
  div.className = 'message aria';
  div.id = 'typing-indicator';
  div.innerHTML = '<span class="message-avatar">◈</span><div class="typing-dots"><span></span><span></span><span></span></div>';
  conversation.appendChild(div);
  conversation.scrollTop = conversation.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

// --- API ---
async function apiTranscribe(audioBlob) {
  const form = new FormData();
  form.append('audio', audioBlob, 'recording.webm');
  const res = await fetch(`${API_BASE}/api/transcribe`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiChat(message, imageBase64 = null) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message, image_base64: imageBase64 }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiUploadDocument(file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/api/upload-documents`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiSession() {
  const res = await fetch(`${API_BASE}/api/session/${sessionId}`);
  if (!res.ok) return { history: [] };
  return res.json();
}

async function apiDocuments() {
  const res = await fetch(`${API_BASE}/api/documents`);
  if (!res.ok) return { documents: [] };
  return res.json();
}

// --- Chat flow ---
async function sendMessage(text, imageBase64 = null) {
  if (!text.trim() && !imageBase64) return;
  const msg = text.trim() || '(Sent an image)';
  let imageUrl = null;
  if (imageBase64) {
    imageUrl = `data:image/png;base64,${imageBase64}`;
    currentImageBase64 = imageBase64;
  }
  appendUserMessage(msg, imageUrl);
  if (textInput) textInput.value = '';
  removeTyping();
  appendTyping();
  setVoiceStatus('Thinking...');

  try {
    const data = await apiChat(msg, imageBase64 || currentImageBase64 || undefined);
    removeTyping();
    setVoiceStatus('');
    appendAriaMessage(data.text || '', data.sources || [], data.audio_base64 || null);
    updateContext(data);
    currentImageBase64 = null;
  } catch (e) {
    removeTyping();
    setVoiceStatus('');
    appendAriaMessage('Sorry, something went wrong. Please try again.', []);
    console.error(e);
  }
}

function updateContext(data) {
  if (processStepsList && data.steps && Array.isArray(data.steps)) {
    processStepsList.innerHTML = '';
    data.steps.forEach(function(step) {
      const li = document.createElement('li');
      li.className = 'process-step process-step-' + (step.phase || 'observe');
      const phaseLabel = (step.phase === 'think' ? 'Think' : step.phase === 'act' ? 'Act' : 'Observe') + ': ';
      li.textContent = phaseLabel + (step.text || '');
      processStepsList.appendChild(li);
    });
  }
  if (sourcesList && data.sources && data.sources.length) {
    sourcesList.innerHTML = data.sources.map(s => `<a href="${s.startsWith('http') ? s : '#'}" target="_blank" rel="noopener">${escapeHtml(s)}</a>`).join('');
  }
  if (stepsList && data.text) {
    const li = document.createElement('li');
    li.textContent = data.text.slice(0, 60) + (data.text.length > 60 ? '…' : '');
    stepsList.appendChild(li);
  }
}

// --- Voice recording ---
async function startRecording() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.ondataavailable = e => { if (e.data.size) audioChunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      setVoiceStatus('Transcribing...');
      try {
        const result = await apiTranscribe(blob);
        if (result.text) await sendMessage(result.text, null);
      } catch (err) {
        console.error(err);
        setVoiceStatus('Transcription failed.');
      }
      setVoiceStatus('');
    };
    mediaRecorder.start();
    if (holdSpeak) holdSpeak.classList.add('recording');
    setVoiceStatus('Listening...');
  } catch (err) {
    console.error('Microphone access denied', err);
    setVoiceStatus('Microphone access needed for voice.');
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    if (holdSpeak) holdSpeak.classList.remove('recording');
  }
}

function setupListeners() {
  // --- Voice recording ---
  if (holdSpeak) {
    holdSpeak.addEventListener('mousedown', (e) => { e.preventDefault(); startRecording(); });
    holdSpeak.addEventListener('mouseup', stopRecording);
    holdSpeak.addEventListener('mouseleave', stopRecording);
    holdSpeak.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
    holdSpeak.addEventListener('touchend', (e) => { e.preventDefault(); stopRecording(); });
  }

  // --- Send button & Enter ---
  if (sendBtn) sendBtn.addEventListener('click', () => sendMessage(textInput ? textInput.value.trim() : '', null));
  if (textInput) {
    textInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(textInput.value.trim(), null);
      }
    });
  }

  // --- Document upload (label "for=file-input" handles click natively; we add drag-drop) ---
  if (docsUpload) {
    docsUpload.addEventListener('dragover', (e) => { e.preventDefault(); docsUpload.classList.add('drag-over'); });
    docsUpload.addEventListener('dragleave', () => docsUpload.classList.remove('drag-over'));
    docsUpload.addEventListener('drop', async (e) => {
      e.preventDefault();
      docsUpload.classList.remove('drag-over');
      const files = Array.from(e.dataTransfer.files).filter(f => /\.(pdf|md|txt)$/i.test(f.name));
      for (const file of files) await uploadDoc(file);
    });
  }
  if (fileInput) {
    fileInput.addEventListener('change', async () => {
      const files = Array.from(fileInput.files || []);
      fileInput.value = '';
      for (const file of files) await uploadDoc(file);
    });
  }

async function uploadDoc(file) {
  try {
    const result = await apiUploadDocument(file);
    if (!docsList) return;
    const item = document.createElement('div');
    item.className = 'doc-item';
    item.innerHTML = `<span>${escapeHtml(result.filename)} <span class="badge">${result.chunks_created} chunks</span></span><span class="delete-doc" data-filename="${escapeHtml(result.filename)}">✕</span>`;
    docsList.appendChild(item);
  } catch (e) {
    console.error(e);
  }
}

  // --- Image attach ---
  if (attachImage && imageInput) {
    attachImage.addEventListener('click', () => imageInput.click());
    imageInput.addEventListener('change', () => {
      const file = imageInput.files && imageInput.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = reader.result.split(',')[1];
        if (base64) sendMessage('(See attached image)', base64);
      };
      reader.readAsDataURL(file);
      imageInput.value = '';
    });
  }

  // --- Load session & docs on load ---
  init();
}

async function init() {
  if (!conversation) return;
  try {
    const session = await apiSession();
    if (session.history && session.history.length > 0) {
      session.history.forEach((m) => {
        if (m.role === 'user') appendUserMessage(m.content);
        else appendAriaMessage(m.content, []);
      });
    } else {
      showOpeningMessage();
    }
    const docs = await apiDocuments();
    if (docsList && (docs.documents || []).length) {
      (docs.documents || []).forEach(d => {
        const item = document.createElement('div');
        item.className = 'doc-item';
        item.innerHTML = `<span>${escapeHtml(d.filename)} <span class="badge">${d.chunks_created} chunks</span></span>`;
        docsList.appendChild(item);
      });
    }
  } catch (e) {
    console.error('Init failed (is the server running at http://localhost:8000?)', e);
    appendAriaMessage('Could not reach the server. Start it with: python main.py', []);
  }
}

function bindDom() {
  conversation = document.getElementById('conversation');
  textInput = document.getElementById('text-input');
  sendBtn = document.getElementById('send-btn');
  holdSpeak = document.getElementById('hold-speak');
  voiceStatus = document.getElementById('voice-status');
  docsUpload = document.getElementById('docs-upload');
  fileInput = document.getElementById('file-input');
  docsList = document.getElementById('docs-list');
  attachImage = document.getElementById('attach-image');
  imageInput = document.getElementById('image-input');
  issueBadge = document.getElementById('issue-badge');
  stepsList = document.getElementById('steps-list');
  sourcesList = document.getElementById('sources-list');
  escalationAlert = document.getElementById('escalation-alert');
  processStepsList = document.getElementById('process-steps-list');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', runApp);
} else {
  runApp();
}

function runApp() {
  try {
    if (window.location.protocol === 'file:') {
      var w = document.getElementById('file-protocol-warning');
      if (w) w.style.display = 'block';
      return;
    }
    bindDom();
    setupListeners();
    document.body.setAttribute('data-aria-ready', 'true');
  } catch (err) {
    console.error('ARIA init error:', err);
    var msg = document.getElementById('conversation');
    if (msg) {
      msg.innerHTML = '<div class="message aria" style="color: var(--error);"><strong>Script error:</strong> ' + String(err.message) + '</div>';
    }
  }
}

function onReady() {
  runApp();
}
