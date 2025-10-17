// storage.js
// Optimized localStorage with in-memory cache + idle debounced flush.
// Prevents CPU spikes and reduces JSON churn.

(function (global) {
  const LS_CONVS   = "lawyerai_conversations"; // [{id, title, msgs:[{role,text,ts}]}]
  const LS_CURRENT = "lawyerai_current";
  const LS_FOLDERS = "lawyerai_folders";       // {folderName:[convIds]}

  let cache = { convs: null, current: null, folders: null };
  const idle = window.requestIdleCallback || (fn => setTimeout(fn, 120));
  let saveQueued = false;

  function safeParse(str, fallback) {
    try { return str ? JSON.parse(str) : fallback; } catch { return fallback; }
  }
  function writeJSON(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); }
    catch (e) { console.warn('localStorage write failed:', e); }
  }
  function queueFlush() {
    if (saveQueued) return;
    saveQueued = true;
    idle(() => {
      if (cache.convs !== null) writeJSON(LS_CONVS, cache.convs);
      if (cache.current !== null) localStorage.setItem(LS_CURRENT, cache.current);
      if (cache.folders !== null) writeJSON(LS_FOLDERS, cache.folders);
      saveQueued = false;
    });
  }

  const S = {
    init() {
      if (cache.convs   === null) cache.convs   = safeParse(localStorage.getItem(LS_CONVS), []);
      if (cache.current === null) cache.current = localStorage.getItem(LS_CURRENT) || '';
      if (cache.folders === null) cache.folders = safeParse(localStorage.getItem(LS_FOLDERS), {});
    },

    getConvs(){ this.init(); return cache.convs; },
    setConvs(v){ cache.convs = v; queueFlush(); },

    getCurrent(){ this.init(); return cache.current; },
    setCurrent(id){ cache.current = id; queueFlush(); },

    getFolders(){ this.init(); return cache.folders; },
    setFolders(v){ cache.folders = v; queueFlush(); },

    ensureConversation() {
      this.init();
      let id = cache.current;
      let conv = cache.convs.find(c => c.id === id);
      if (!conv) {
        id = 'c' + Date.now();
        conv = { id, title: 'Chat ' + (cache.convs.length + 1), msgs: [] };
        cache.convs.push(conv);
        cache.current = id;
        queueFlush();
      }
      return conv;
    },

    pushMsg(role, text) {
      this.init();
      const id = cache.current;
      const idx = cache.convs.findIndex(c => c.id === id);
      if (idx === -1) return;
      cache.convs[idx].msgs.push({ role, text, ts: Date.now() });
      queueFlush();
    }
  };

  // Pause writes when tab hidden to avoid churn on tab switch
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) queueFlush();
  }, { passive: true });

  global.StorageAPI = S;
})(window);