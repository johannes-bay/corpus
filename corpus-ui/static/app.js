// ---------------------------------------------------------------------------
// Corpus UI — vanilla JS SPA
// ---------------------------------------------------------------------------

const AXES = ['bpm', 'key', 'spectral', 'temporal', 'provenance'];
const DEFAULT_WEIGHTS = { bpm: 0.8, key: 0.9, spectral: 0.3, temporal: 0.5, provenance: 0.2 };

// State
let ws = null;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  initNav();
  initAxesSliders();
  initCompose();
  initSearch();
  initModal();
  initPlayer();
  loadStats();
});

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

function initNav() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('view-' + btn.dataset.view).classList.add('active');
      if (btn.dataset.view === 'stats') loadStats();
    });
  });
}

// ---------------------------------------------------------------------------
// Axes sliders
// ---------------------------------------------------------------------------

function initAxesSliders() {
  const container = document.getElementById('axes-sliders');
  AXES.forEach(name => {
    const val = DEFAULT_WEIGHTS[name] || 0.5;
    const row = document.createElement('div');
    row.className = 'axis-row';
    row.innerHTML = `
      <label>${name}</label>
      <input type="range" min="0" max="1" step="0.05" value="${val}" data-axis="${name}" />
      <span class="axis-val">${val.toFixed(2)}</span>
    `;
    row.querySelector('input').addEventListener('input', e => {
      row.querySelector('.axis-val').textContent = parseFloat(e.target.value).toFixed(2);
    });
    container.appendChild(row);
  });
}

function getAxesWeights() {
  const weights = {};
  document.querySelectorAll('#axes-sliders input[type="range"]').forEach(el => {
    const v = parseFloat(el.value);
    if (v > 0) weights[el.dataset.axis] = v;
  });
  return weights;
}

// ---------------------------------------------------------------------------
// Compose
// ---------------------------------------------------------------------------

function initCompose() {
  document.getElementById('btn-compose').addEventListener('click', doCompose);
  document.getElementById('seed-path').addEventListener('keydown', e => {
    if (e.key === 'Enter') doCompose();
  });
}

async function doCompose() {
  const seed = document.getElementById('seed-path').value.trim();
  if (!seed) return;

  const axes = getAxesWeights();
  const count = parseInt(document.getElementById('compose-count').value) || 10;
  const btn = document.getElementById('btn-compose');
  btn.disabled = true;
  btn.textContent = 'Composing...';

  try {
    const res = await fetch('/api/compose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seed, axes, count }),
    });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    renderCompose(data);
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Compose';
  }
}

function renderCompose(data) {
  // Seed info
  const seedEl = document.getElementById('seed-info');
  if (data.seed) {
    seedEl.classList.remove('hidden');
    let propsHtml = '';
    if (data.seed_properties && data.seed_properties.length) {
      propsHtml = data.seed_properties.map(p => {
        const v = p.value_num != null ? p.value_num : (p.value_txt || '');
        return `<span>[${p.domain}.${p.key}] = ${v}</span>`;
      }).join(' &middot; ');
    }
    seedEl.innerHTML = `
      <h3>${data.seed.filename}</h3>
      <div class="props">${data.seed.path}<br/>${propsHtml}</div>
    `;
  } else {
    seedEl.classList.add('hidden');
  }

  // Matches
  const list = document.getElementById('matches-list');
  if (!data.matches || !data.matches.length) {
    list.innerHTML = '<p style="color:var(--text-dim)">No matches found. Has the seed file been enriched?</p>';
    return;
  }

  list.innerHTML = data.matches.map((m, i) => {
    const explainHtml = (m.explanation || []).map(e => `<div>${e}</div>`).join('');
    const isAudio = isAudioFile(m.file.extension);
    const isImage = isImageFile(m.file.extension);
    return `
      <div class="match-card" data-path="${esc(m.file.path)}">
        <div class="match-header">
          <div>
            <span class="match-rank">#${i + 1}</span>
            <span class="match-name">${esc(m.file.filename)}</span>
          </div>
          <span class="match-score">${m.score.toFixed(3)}</span>
        </div>
        <div class="match-explain">${explainHtml}</div>
        <div class="match-actions">
          <button onclick="event.stopPropagation(); showFileDetail('${esc(m.file.path)}')">Info</button>
          ${isAudio ? `<button onclick="event.stopPropagation(); playFile('${esc(m.file.path)}', '${esc(m.file.filename)}')">Play</button>` : ''}
          ${isImage ? `<button onclick="event.stopPropagation(); previewImage('${esc(m.file.path)}')">Preview</button>` : ''}
          <button onclick="event.stopPropagation(); useSeed('${esc(m.file.path)}')">Use as seed</button>
        </div>
      </div>
    `;
  }).join('');
}

function useSeed(path) {
  document.getElementById('seed-path').value = path;
  doCompose();
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

function initSearch() {
  document.getElementById('btn-search').addEventListener('click', doSearch);
  document.getElementById('search-q').addEventListener('keydown', e => {
    if (e.key === 'Enter') doSearch();
  });
}

async function doSearch() {
  const q = document.getElementById('search-q').value.trim();
  const ext = document.getElementById('search-ext').value.trim();
  const params = new URLSearchParams();
  if (q) params.set('q', q);
  if (ext) params.set('ext', ext);
  params.set('limit', '50');

  try {
    const res = await fetch('/api/search?' + params.toString());
    const data = await res.json();
    renderSearch(data);
  } catch (e) {
    document.getElementById('search-results').innerHTML = '<p>Error: ' + e.message + '</p>';
  }
}

function renderSearch(files) {
  const el = document.getElementById('search-results');
  if (!files.length) {
    el.innerHTML = '<p style="color:var(--text-dim)">No results</p>';
    return;
  }
  el.innerHTML = files.map(f => `
    <div class="file-row" onclick="showFileDetail('${esc(f.path)}')">
      <span class="ext">${esc(f.extension || '')}</span>
      <span class="name">${esc(f.filename)}</span>
      <span class="size">${formatSize(f.size_bytes)}</span>
    </div>
  `).join('');
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

async function loadStats() {
  const el = document.getElementById('stats-content');
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    renderStats(data);
  } catch (e) {
    el.innerHTML = '<p>Failed to load stats</p>';
  }
}

function renderStats(data) {
  const el = document.getElementById('stats-content');
  const enrichKeys = Object.keys(data.enrichment || {});
  const enrichRows = enrichKeys.map(k =>
    `<tr><td>${k}</td><td>${data.enrichment[k].toLocaleString()}</td></tr>`
  ).join('');

  el.innerHTML = `
    <div class="stat-grid">
      <div class="stat-card"><div class="label">Total files</div><div class="value">${data.total_files.toLocaleString()}</div></div>
      <div class="stat-card"><div class="label">Audio files</div><div class="value">${data.audio_files.toLocaleString()}</div></div>
      <div class="stat-card"><div class="label">Image files</div><div class="value">${data.image_files.toLocaleString()}</div></div>
    </div>
    ${enrichKeys.length ? `
    <h3 style="color:var(--text-dim); margin-bottom: 8px;">Enrichment</h3>
    <table class="enrichment-table">
      <thead><tr><th>Property</th><th>Count</th></tr></thead>
      <tbody>${enrichRows}</tbody>
    </table>
    ` : ''}
  `;
}

// ---------------------------------------------------------------------------
// File detail modal
// ---------------------------------------------------------------------------

function initModal() {
  document.querySelector('.modal-close').addEventListener('click', closeModal);
  document.getElementById('file-modal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeModal();
  });
}

function closeModal() {
  document.getElementById('file-modal').classList.add('hidden');
}

async function showFileDetail(path) {
  const modal = document.getElementById('file-modal');
  const body = document.getElementById('modal-body');
  body.innerHTML = '<p style="color:var(--text-dim)">Loading...</p>';
  modal.classList.remove('hidden');

  try {
    const res = await fetch('/api/file?path=' + encodeURIComponent(path));
    const data = await res.json();
    renderFileDetail(data);
  } catch (e) {
    body.innerHTML = '<p>Error loading file info</p>';
  }
}

function renderFileDetail(data) {
  const body = document.getElementById('modal-body');
  if (!data.file) {
    body.innerHTML = '<p>File not found</p>';
    return;
  }

  const f = data.file;
  let html = `<h3 style="margin-bottom:12px">${esc(f.filename)}</h3>`;
  html += `<div class="meta-section"><div class="meta-row"><span class="key">Path</span><span>${esc(f.path)}</span></div>`;
  html += `<div class="meta-row"><span class="key">Size</span><span>${formatSize(f.size_bytes)}</span></div>`;
  if (f.modified_date) html += `<div class="meta-row"><span class="key">Modified</span><span>${esc(f.modified_date)}</span></div>`;
  html += `</div>`;

  // Preview
  if (isImageFile(f.extension)) {
    html += `<img class="preview-img" src="/api/file/preview?path=${encodeURIComponent(f.path)}" />`;
  }
  if (isAudioFile(f.extension)) {
    html += `<audio controls style="width:100%;margin-bottom:12px"><source src="/api/file/preview?path=${encodeURIComponent(f.path)}"></audio>`;
  }

  // Typed metadata
  if (data.audio_meta) html += renderMetaSection('Audio', data.audio_meta);
  if (data.photo_meta) html += renderMetaSection('Photo', data.photo_meta);
  if (data.video_meta) html += renderMetaSection('Video', data.video_meta);
  if (data.document_meta) html += renderMetaSection('Document', data.document_meta);
  if (data.font_meta) html += renderMetaSection('Font', data.font_meta);

  // Properties
  if (data.properties && data.properties.length) {
    html += '<div class="meta-section"><h4>Enriched Properties</h4>';
    data.properties.forEach(p => {
      const v = p.value_num != null ? p.value_num : (p.value_txt || '');
      html += `<div class="meta-row"><span class="key">${p.domain}.${p.key}</span><span>${v}</span></div>`;
    });
    html += '</div>';
  }

  body.innerHTML = html;
}

function renderMetaSection(title, meta) {
  let html = `<div class="meta-section"><h4>${title}</h4>`;
  for (const [k, v] of Object.entries(meta)) {
    if (v == null || v === '') continue;
    html += `<div class="meta-row"><span class="key">${k}</span><span>${v}</span></div>`;
  }
  html += '</div>';
  return html;
}

// ---------------------------------------------------------------------------
// Audio player
// ---------------------------------------------------------------------------

function initPlayer() {
  document.getElementById('player-close').addEventListener('click', () => {
    const player = document.getElementById('player');
    player.pause();
    player.src = '';
    document.getElementById('audio-player').classList.add('hidden');
  });
}

function playFile(path, name) {
  const bar = document.getElementById('audio-player');
  const player = document.getElementById('player');
  const nameEl = document.getElementById('player-name');
  player.src = '/api/file/preview?path=' + encodeURIComponent(path);
  nameEl.textContent = name;
  bar.classList.remove('hidden');
  player.play();
}

function previewImage(path) {
  const modal = document.getElementById('file-modal');
  const body = document.getElementById('modal-body');
  body.innerHTML = `<img class="preview-img" src="/api/file/preview?path=${encodeURIComponent(path)}" style="max-width:100%"/>`;
  modal.classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function esc(s) {
  if (!s) return '';
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function formatSize(bytes) {
  if (bytes == null) return '';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}

const AUDIO_EXTS = ['.wav', '.mp3', '.m4a', '.aif', '.aiff', '.flac', '.ogg', '.aac'];
const IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.webp'];

function isAudioFile(ext) {
  return ext && AUDIO_EXTS.includes(ext.toLowerCase());
}

function isImageFile(ext) {
  return ext && IMAGE_EXTS.includes(ext.toLowerCase());
}
