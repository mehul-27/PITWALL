/* PitWall — Chat Logic & Animations */
(function () {
  'use strict';

  const chatMessages = document.getElementById('chatMessages');
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const emptyState = document.getElementById('emptyState');

  // Sidebar meta elements
  const metaMode = document.getElementById('metaMode');
  const metaTime = document.getElementById('metaTime');
  const metaConfBar = document.getElementById('metaConfBar');
  const metaTopics = document.getElementById('metaTopics');
  const metaQueryTime = document.getElementById('metaQueryTime');
  const metaDataPoints = document.getElementById('metaDataPoints');

  let isWaiting = false;

  // ── Ticker ───────────────────────────────────────────────
  (function initTicker() {
    const items = [
      { sym: 'PITWALL AI', val: 'RACE ENGINEER', pts: '' },
      { sym: 'SEASONS', val: '2022-25', pts: '' },
      { sym: 'DATA', val: 'QUALIFYING', pts: 'FP2 \u00b7 RACE' },
      { sym: 'DRIVERS', val: 'ALL GRID', pts: '' },
      { sym: 'TYRES', val: 'STRATEGY', pts: 'PACE' },
    ];
    const mk = i =>
      `<span class="tick"><span class="sym">${i.sym}</span> <span class="val">${i.val}</span>${i.pts ? ` <span class="pts">${i.pts}</span>` : ''}</span><span class="tick tick-dot">\u25c6</span>`;
    const half = items.map(mk).join('');
    document.getElementById('tkTrack').innerHTML = half + half;
  })();

  // ── Helpers ──────────────────────────────────────────────
  function timestamp() {
    const d = new Date();
    return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    });
  }

  function hideEmptyState() {
    if (emptyState) emptyState.style.display = 'none';
  }

  // ── Speed line fire ──────────────────────────────────────
  function fireSpeedLines() {
    document.querySelectorAll('.speed-line').forEach((el, i) => {
      el.classList.remove('active');
      void el.offsetWidth; // reflow
      setTimeout(() => el.classList.add('active'), i * 400);
    });
  }

  function sourceLabelFromMeta(meta) {
    const m = {
      fastf1_valid: 'FastF1 Query',
      data_unavailable: 'Data Unavailable',
      model_knowledge: 'Model Knowledge',
      no_data_found: 'No Data Found',
    };
    if (!meta || !meta.source) return m.model_knowledge;
    return m[meta.source] || m.model_knowledge;
  }

  // ── Create message element ───────────────────────────────
  function createMessage(role, text, mode, meta) {
    const div = document.createElement('div');
    const modeClass = mode === 'telemetry' ? 'mode-telemetry' : 'mode-general';
    div.className = `msg ${role} ${role === 'ai' ? modeClass : ''}`;

    let kicker = '';
    if (role === 'user') {
      kicker = '<span class="msg-kicker">YOU</span>';
    } else if (mode === 'telemetry') {
      kicker = '<span class="msg-kicker">TELEMETRY</span>';
    } else {
      kicker = '<span class="msg-kicker">RACE ENGINEER</span>';
    }

    let footer = `<span class="msg-timestamp">${timestamp()}</span>`;
    if (role === 'ai') {
      const src = sourceLabelFromMeta(meta);
      footer = `<span class="msg-source">Source \u00b7 ${src}</span>`;
      if (meta && meta.source === 'fastf1_valid' && meta.telemetry && meta.telemetry.query_ms !== undefined) {
        footer += `<span class="msg-query-time">${meta.telemetry.query_ms}ms</span>`;
      }
      if (meta && meta.telemetry) {
        (meta.telemetry.drivers || []).forEach(d => {
          footer += `<span class="msg-chip">${d}</span>`;
        });
        if (meta.telemetry.circuit) footer += `<span class="msg-chip">${meta.telemetry.circuit}</span>`;
        if (meta.telemetry.year && meta.telemetry.session) {
          footer += `<span class="msg-chip">${meta.telemetry.year} ${meta.telemetry.session}</span>`;
        }
      }
      footer += `<span class="msg-timestamp">${timestamp()}</span>`;
    }

    const bodyHtml = role === 'ai' ? renderAssistantMarkdown(text) : escapeHtml(text);

    div.innerHTML = `
      ${kicker}
      <div class="msg-body ${role === 'ai' ? 'msg-body-md' : ''}">${bodyHtml}</div>
      <div class="msg-footer">${footer}</div>
    `;
    return div;
  }

  function escapeHtml(t) {
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
  }

  /** If the model echoed the same telemetry block twice, keep the first (compare-query bugfix). */
  function stripDuplicateTelemetryBlock(raw) {
    if (!raw || typeof raw !== 'string') return raw;
    const m = '=== TELEMETRY DATA ===';
    const first = raw.indexOf(m);
    if (first < 0) return raw;
    const second = raw.indexOf(m, first + m.length);
    if (second < 0) return raw;
    return raw.slice(0, second).trimEnd();
  }

  /** Model replies: markdown -> safe HTML. User text stays plain escaped. */
  function renderAssistantMarkdown(raw) {
    const cleaned = stripDuplicateTelemetryBlock(raw);
    if (!cleaned) return '';
    if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
      return escapeHtml(cleaned).replace(/\n/g, '<br>');
    }
    const html = marked.parse(cleaned, { breaks: true, gfm: true });
    return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
  }

  // ── Thinking indicator ───────────────────────────────────
  function showThinking(mode) {
    const div = document.createElement('div');
    const mc = mode === 'telemetry' ? 'mode-telemetry' : 'mode-general';
    div.className = `thinking ${mc}`;
    div.id = 'thinkingIndicator';

    const label = mode === 'telemetry' ? 'Querying telemetry...' : 'Consulting race engineer...';
    let extra = '';
    if (mode === 'telemetry') {
      extra = `
        <div style="flex-basis:100%">
          <div class="thinking-bar"><div class="thinking-bar-fill"></div></div>
          <div class="thinking-bar-label">FastF1 fetch</div>
        </div>`;
    }

    div.innerHTML = `
      <div class="thinking-dots"><span></span><span></span><span></span></div>
      <span class="thinking-text">${label}</span>
      ${extra}
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
  }

  function hideThinking() {
    const el = document.getElementById('thinkingIndicator');
    if (el) el.remove();
  }

  // ── Update sidebar meta ──────────────────────────────────
  function updateMeta(meta) {
    if (!meta) return;

    // Mode badge
    if (metaMode) {
      metaMode.className = `meta-badge ${meta.mode || 'general'}`;
      metaMode.textContent = (meta.mode || 'general').toUpperCase();
    }

    // Response time
    if (metaTime) metaTime.textContent = `${meta.response_time_ms || 0}ms`;

    // Confidence bar
    if (metaConfBar) {
      metaConfBar.style.width = `${(meta.confidence || 0) * 100}%`;
      metaConfBar.classList.remove('animate');
      void metaConfBar.offsetWidth;
      metaConfBar.classList.add('animate');
    }

    // Topics
    if (metaTopics) {
      metaTopics.innerHTML = (meta.topics || [])
        .map(t => `<span class="meta-chip">${t.toUpperCase()}</span>`)
        .join('');
    }

    // Telemetry-specific
    if (metaQueryTime) {
      if (meta.mode === 'telemetry' && meta.telemetry && meta.telemetry.query_ms !== undefined) {
        metaQueryTime.parentElement.style.display = '';
        metaQueryTime.textContent = `${meta.telemetry.query_ms}ms`;
      } else {
        metaQueryTime.parentElement.style.display = 'none';
      }
    }
    if (metaDataPoints) {
      if (meta.mode === 'telemetry' && meta.telemetry) {
        const parts = [];
        if (meta.telemetry.lap) parts.push(`Lap ${meta.telemetry.lap}`);
        (meta.telemetry.drivers || []).forEach(d => parts.push(d));
        if (meta.telemetry.circuit) parts.push(meta.telemetry.circuit);
        if (meta.telemetry.year && meta.telemetry.session) parts.push(`${meta.telemetry.year} ${meta.telemetry.session}`);
        metaDataPoints.parentElement.style.display = parts.length ? '' : 'none';
        metaDataPoints.textContent = parts.join(' \u00b7 ');
      } else {
        metaDataPoints.parentElement.style.display = 'none';
      }
    }
  }

  // ── Send message ─────────────────────────────────────────
  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || isWaiting) return;

    isWaiting = true;
    sendBtn.disabled = true;
    hideEmptyState();

    // Client-side guess for the thinking label only (server intent.py is authoritative).
    const strategyScene = /\b(undercut|overcut|stay out|cover the|safety car|VSC|track position|car behind|seconds back|pitted?|pitting|stint|hards?|mediums?|softers?|do we|should we|or pit)\b/i.test(text);
    const wantsSqlLap = /\btelemetry|from (the |our )?database|fastest lap|qualifying lap|show me the|what (was|is) the|lap time|sector [123]|\bcompare\s+.+\b(to|vs|versus)\b/i.test(text);
    const isTelemetry = (!strategyScene || wantsSqlLap) && (/\b(?:lap\s*\d+|\d+(?:st|nd|rd|th)?\s*lap)\b|\b(compare|vs|versus)\b|\bsector\s*[123]|\b(Q[123]|FP[123])\b|\b(fastest lap|telemetry|speed trap)\b/i.test(text));
    const thinkingMode = isTelemetry ? 'telemetry' : 'general';

    // Append user message
    chatMessages.appendChild(createMessage('user', text, 'general', null));
    chatInput.value = '';
    chatInput.style.height = 'auto';
    scrollToBottom();

    // Show thinking
    showThinking(thinkingMode);

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();

      hideThinking();

      // Append AI message
      const mode = (data.metadata && data.metadata.mode) || 'general';
      chatMessages.appendChild(createMessage('ai', data.response, mode, data.metadata));
      scrollToBottom();

      // Update sidebar
      updateMeta(data.metadata);

      // Fire speed lines on telemetry
      if (mode === 'telemetry') fireSpeedLines();

    } catch (err) {
      hideThinking();
      chatMessages.appendChild(createMessage('ai', 'Connection error. Please try again.', 'general', null));
      scrollToBottom();
    }

    isWaiting = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }

  // ── Event listeners ──────────────────────────────────────
  sendBtn.addEventListener('click', sendMessage);

  chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-resize textarea
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
  });

  // Quick prompts
  document.querySelectorAll('.prompt-card').forEach(card => {
    card.addEventListener('click', () => {
      chatInput.value = card.textContent.trim();
      chatInput.focus();
      chatInput.dispatchEvent(new Event('input'));
    });
  });

})();
