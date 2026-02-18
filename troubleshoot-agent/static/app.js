/**
 * ARIA frontend: voice recording, API calls, UI state.
 * Vanilla JS, no framework.
 */

const API_BASE = ''; // same origin when served by FastAPI

let sessionId = localStorage.getItem('aria_session_id') || crypto.randomUUID();
localStorage.setItem('aria_session_id', sessionId);

let mediaRecorder = null;
let audioChunks = [];
let stream = null;
let currentImageBase64 = null;

// --- DOM ---
const conversation = document.getElementById('conversation');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const holdSpeak = document.getElementById('hold-speak');
const voiceStatus = document.getElementById('voice-status');
const docsUpload = document.getElementById('docs-upload');
const fileInput = document.getElementById('file-input');
const docsList = document.getElementById('docs-list');
const attachImage = document.getElementById('attach-image');
const imageInput = document.getElementById('image-input');
const issueBadge = document.getElementById('issue-badge');
const stepsList = document.getElementById('steps-list');
const sourcesList = document.getElementById('sources-list');
const escalationAlert = document.getElementById('escalation-alert');

// --- Opening message ---
function showOpeningMessage() {
  const msg = "Hello, I'm ARIA. What iPhone issue can I help you troubleshoot today?";
  appendAriaMessage(msg, [], null);
}

function setVoiceStatus(text) {
  voiceStatus.textContent = text;
}

// --- Messages ---
function appendUserMessage(text, imageUrl = null) {
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

function appendAriaMessage(text, sources = [], audioBase64 = null) {
  const div = document.createElement('div');
  div.className = 'message aria';
  let html = '<span class="message-avatar">◈</span>';
  html += `<div class="message-text">${escapeHtml(text)}</div>`;
  if (sources && sources.length) {
    html += '<div class="message-sources">';
    sources.forEach(s => {
      const href = s.startsWith('http') ? s : '#';
      html += `<a class="source-pill" href="${href}" target="_blank" rel="noopener">${escapeHtml(s)}</a>`;
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
  textInput.value = '';
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
  if (data.sources && data.sources.length) {
    sourcesList.innerHTML = data.sources.map(s => `<a href="${s.startsWith('http') ? s : '#'}" target="_blank" rel="noopener">${escapeHtml(s)}</a>`).join('');
  }
  const steps = stepsList.querySelectorAll('li');
  if (data.text) {
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
    holdSpeak.classList.add('recording');
    setVoiceStatus('Listening...');
  } catch (err) {
    console.error('Microphone access denied', err);
    setVoiceStatus('Microphone access needed for voice.');
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    holdSpeak.classList.remove('recording');
  }
}

holdSpeak.addEventListener('mousedown', (e) => { e.preventDefault(); startRecording(); });
holdSpeak.addEventListener('mouseup', stopRecording);
holdSpeak.addEventListener('mouseleave', stopRecording);
holdSpeak.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
holdSpeak.addEventListener('touchend', (e) => { e.preventDefault(); stopRecording(); });

// --- Send button & Enter ---
sendBtn.addEventListener('click', () => sendMessage(textInput.value.trim(), null));
textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage(textInput.value.trim(), null);
  }
});

// --- Document upload ---
docsUpload.addEventListener('click', () => fileInput.click());
docsUpload.addEventListener('dragover', (e) => { e.preventDefault(); docsUpload.classList.add('drag-over'); });
docsUpload.addEventListener('dragleave', () => docsUpload.classList.remove('drag-over'));
docsUpload.addEventListener('drop', async (e) => {
  e.preventDefault();
  docsUpload.classList.remove('drag-over');
  const files = Array.from(e.dataTransfer.files).filter(f => /\.(pdf|md|txt)$/i.test(f.name));
  for (const file of files) await uploadDoc(file);
});

fileInput.addEventListener('change', async () => {
  const files = Array.from(fileInput.files || []);
  fileInput.value = '';
  for (const file of files) await uploadDoc(file);
});

async function uploadDoc(file) {
  try {
    const result = await apiUploadDocument(file);
    const item = document.createElement('div');
    item.className = 'doc-item';
    item.innerHTML = `<span>${escapeHtml(result.filename)} <span class="badge">${result.chunks_created} chunks</span></span><span class="delete-doc" data-filename="${escapeHtml(result.filename)}">✕</span>`;
    docsList.appendChild(item);
  } catch (e) {
    console.error(e);
  }
}

// --- Image attach ---
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

// --- Load session & docs on load ---
async function init() {
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
  (docs.documents || []).forEach(d => {
    const item = document.createElement('div');
    item.className = 'doc-item';
    item.innerHTML = `<span>${escapeHtml(d.filename)} <span class="badge">${d.chunks_created} chunks</span></span>`;
    docsList.appendChild(item);
  });
}

init();
