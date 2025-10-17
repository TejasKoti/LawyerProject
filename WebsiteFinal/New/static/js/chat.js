// chat.js — Optimized Chat Logic + Expandable Advanced Search (fixed search behavior)

(function (global) {
  const S = window.StorageAPI;

  /* === DOM References === */
  const timeline      = document.getElementById('timeline');
  const contentScroll = document.getElementById('contentScroll');
  const input         = document.getElementById('composerInput');
  const sendBtn       = document.getElementById('sendBtn');
  const btnShare      = document.getElementById('btnShare');
  const btnMore       = document.getElementById('btnMore');
  const moreMenu      = document.getElementById('moreMenu');
  const addToFolderMi = document.getElementById('miAddToFolder');

  /* --- Search bar elements (expandable) --- */
  const searchToggle  = document.getElementById("searchToggle");
  const searchBar     = document.getElementById("searchBar");
  const searchInput   = document.getElementById("searchInput");
  const searchFilter = document.getElementById("searchFilterSelect");
  const matchCounter  = document.getElementById("matchCounter");
  const nextMatchBtn  = document.getElementById("nextMatch");
  const prevMatchBtn  = document.getElementById("prevMatch");
  const clearSearchBtn= document.getElementById("clearSearch");

  /* === Chat Virtualization (Enhanced with Formatting) === */
  const RENDER_WINDOW = 80;
  let renderedCount = 0;

  // 🧩 Formatter for chatbot responses (Markdown → HTML)
  function formatResponse(text) {
  if (!text) return "";

  let formatted = text
    // --- Headings ---
    .replace(/^\s*#{3}\s?(.*)$/gim, "<h3>$1</h3>")    // ### Heading
    .replace(/^\s*#{2}\s?(.*)$/gim, "<h2>$1</h2>")    // ## Heading
    .replace(/^\s*#\s?(.*)$/gim, "<h1>$1</h1>")       // # Heading

    // --- Bold / Italic ---
    .replace(/\*\*(.*?)\*\*/gim, "<b>$1</b>")
    .replace(/\*(.*?)\*/gim, "<i>$1</i>")

    // --- Bullet Points ---
    .replace(/^\s*[-•]\s+(.*)$/gim, "• $1")

    // --- Numbered Lists ---
    .replace(/^\s*(\d+)\.\s+(.*)$/gim, "<div class='list-item'><b>$1.</b> $2</div>")

    // --- “Next Actions” and similar bold section labels on their own line ---
    .replace(/(Next Actions:)/gi, "<br><br><b>$1</b><br>")

    // --- Cleanup excessive spacing ---
    .replace(/\n{3,}/g, "\n\n")
    .replace(/\n/g, "<br>")
    .replace(/<br>\s*<h2>/g, "<h2>")  // remove unwanted gaps before headings
    .replace(/<\/h2><br>/g, "</h2>"); // remove unwanted gaps after headings

  return formatted.trim();
}

  // Creates a chat message DOM node
  function appendNode(role, text) {
    const el = document.createElement('div');
    el.className = 'msg ' + (role === 'user' ? 'user' : 'bot');

    // 🔹 Use textContent for user (safe), innerHTML for bot (to allow formatting)
    if (role === 'bot') {
      el.innerHTML = formatResponse(text);
    } else {
      el.textContent = text;
    }
    return el;
  }

  function renderTail(conv) {
    const total = conv.msgs.length;
    const start = Math.max(0, total - RENDER_WINDOW);
    timeline.innerHTML = '';
    const frag = document.createDocumentFragment();
    for (let i = start; i < total; i++) {
      const m = conv.msgs[i];
      frag.appendChild(appendNode(m.role, m.text));
    }
    timeline.appendChild(frag);
    renderedCount = total;
    requestAnimationFrame(() => {
      contentScroll.scrollTop = contentScroll.scrollHeight;
    });
  }

  function loadCurrent() {
    const conv = S.ensureConversation();
    renderTail(conv);
  }

  function appendMsg(role, text) {
    const conv = S.ensureConversation();
    S.pushMsg(role, text);
    const el = appendNode(role, text);
    const total = conv.msgs.length;
    if (total <= RENDER_WINDOW) {
      timeline.appendChild(el);
    } else {
      if (timeline.childNodes.length >= RENDER_WINDOW) {
        timeline.removeChild(timeline.firstChild);
      }
      timeline.appendChild(el);
    }
    requestAnimationFrame(() => {
      contentScroll.scrollTop = contentScroll.scrollHeight;
    });
  }

  function typingOn() {
    const t = document.createElement('div');
    t.id = 'typing';
    t.className = 'msg bot';
    t.textContent = '…';
    timeline.appendChild(t);
    requestAnimationFrame(() => {
      contentScroll.scrollTop = contentScroll.scrollHeight;
    });
  }

  function typingOff() {
    const t = document.getElementById('typing');
    if (t) t.remove();
  }

  /* === Backend send === */
  function sendNow() {
    const q = input.value.trim();
    if (!q) return;
    input.value = '';
    appendMsg('user', q);
    typingOn();

    $.post("/ask", { question: q }, function (data) {
      typingOff();
      const resp = data && data.response ? String(data.response) : "Error: no response";
      appendMsg('bot', resp);
    }).fail(function () {
      typingOff();
      appendMsg('bot', "Network error.");
    });
  }

  sendBtn.addEventListener('click', sendNow);
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendNow();
    }
  });

  /* === Three-dots menu === */
  btnMore.addEventListener('click', e => {
    e.stopPropagation();
    const open = moreMenu.style.display === 'block';
    moreMenu.style.display = open ? 'none' : 'block';
    btnMore.setAttribute('aria-expanded', open ? 'false' : 'true');
  }, { passive: true });

  document.addEventListener('click', () => {
    moreMenu.style.display = 'none';
    btnMore.setAttribute('aria-expanded', 'false');
  }, { passive: true });

  addToFolderMi.addEventListener('click', () => {
    moreMenu.style.display = 'none';
    global.UI.openAssignFolderModal();
  });

  /* === Expandable Search System === */
  let matches = [], currentIdx = -1, searchActive = false;

  function clearHighlights() {
    document.querySelectorAll("#timeline .msg mark.hl").forEach(m => {
      const parent = m.parentNode;
      parent.replaceChild(document.createTextNode(m.textContent), m);
      parent.normalize();
    });
    matches = [];
    currentIdx = -1;
    matchCounter.textContent = "0 / 0";
  }

  function highlightMatches(query) {
    clearHighlights();
    if (!query.trim()) return;
    const filter = searchFilter ? searchFilter.dataset.value : "all";
    const nodes = Array.from(document.querySelectorAll(".msg")).filter(n => {
      if (filter === "user") return n.classList.contains("user");
      if (filter === "bot") return n.classList.contains("bot");
      return true;
    });

    const re = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
    nodes.forEach(node => {
      const txt = node.textContent;
      if (re.test(txt)) {
        node.innerHTML = txt.replace(re, m => `<mark class="hl">${m}</mark>`);
      }
    });

    matches = Array.from(document.querySelectorAll("mark.hl"));
    if (matches.length) {
      currentIdx = 0;
      scrollToMatch();
    }
    matchCounter.textContent = matches.length
      ? `${currentIdx + 1} / ${matches.length}`
      : "0 / 0";

    searchActive = !!query;
  }

  function scrollToMatch() {
    if (!matches.length) return;
    matches.forEach(m => m.style.background = "#fff59d");
    const m = matches[currentIdx];
    if (!m) return;
    m.scrollIntoView({ behavior: "smooth", block: "center" });
    matches.forEach(x => (x.style.background = "#fff59d"));
    m.style.background = "#a5d6ff";
    matchCounter.textContent = `${currentIdx + 1} / ${matches.length}`;
  }

  nextMatchBtn.addEventListener("click", () => {
    if (!matches.length) return;
    currentIdx = (currentIdx + 1) % matches.length;
    scrollToMatch();
  });

  prevMatchBtn.addEventListener("click", () => {
    if (!matches.length) return;
    currentIdx = (currentIdx - 1 + matches.length) % matches.length;
    scrollToMatch();
  });

  searchInput.addEventListener("input", e => highlightMatches(e.target.value));
  searchFilter.addEventListener("change", () => highlightMatches(searchInput.value));

  clearSearchBtn.addEventListener("click", stopSearch);
  searchToggle.addEventListener("click", () => {
    if (searchBar.classList.contains("active")) stopSearch();
    else {
      searchBar.classList.add("active");
      searchInput.focus();
    }
  });

  function stopSearch() {
    clearHighlights();
    searchBar.classList.remove("active");
    searchInput.value = "";
    searchActive = false;
  }

  /* === Export Transcript as Text (Auto-detect message structure) === */
btnShare.addEventListener('click', () => {
  // Try different possible message structures
  const selectors = ['.message', '.msg', '.bubble', '.chat-msg', '.timeline-item'];
  let messages = [];
  for (const sel of selectors) {
    const found = document.querySelectorAll(sel);
    if (found.length) {
      messages = found;
      break;
    }
  }

  if (!messages.length) {
    alert('⚠️ No chat messages to export.');
    return;
  }

  let transcript = '=== LawyerAI Chat Transcript ===\n\n';

  messages.forEach(msg => {
    const text = msg.innerText.trim();
    if (!text) return;

    // Detect sender (based on class)
    const isUser = msg.classList.contains('user') || msg.classList.contains('sent');
    const isBot  = msg.classList.contains('bot')  || msg.classList.contains('received');

    // Find timestamp if exists, otherwise fallback to now
    const timeElem = msg.querySelector('.timestamp, time');
    const time = timeElem ? timeElem.textContent.trim() : new Date().toLocaleTimeString();

    if (isUser) transcript += `[User @ ${time}]: ${text}\n`;
    else if (isBot) transcript += `[Bot @ ${time}]: ${text}\n`;
    else transcript += `[System @ ${time}]: ${text}\n`;
  });

  // Create downloadable .txt file
  const blob = new Blob([transcript], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `transcript_${new Date().toISOString().split('T')[0]}.txt`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 500);
});

  /* === Public Interface === */
  global.ChatUI = { appendMsg, typingOn, typingOff, loadCurrent, sendNow };

})(window);