// ui.js
// Efficient rendering + event delegation + modal lifecycle.
// Added: safeguard so global click handler ignores .search-filter dropdown.

(function (global) {
  const S = window.StorageAPI;

  // Refs
  const libraryList   = document.getElementById('libraryList');
  const foldersList   = document.getElementById('foldersList');
  const folderModal   = document.getElementById('folderModal');
  const folderChecks  = document.getElementById('folderChecks');
  const closeFolder   = document.getElementById('closeFolderModal');
  const saveFolder    = document.getElementById('saveFolderAssign');
  const renameModal   = document.getElementById('renameModal');
  const renameTitleEl = document.getElementById('renameTitle');
  const renameInput   = document.getElementById('renameInput');
  const renameCancel  = document.getElementById('cancelRename');
  const renameConfirm = document.getElementById('confirmRename');
  const btnCreateFolder = document.getElementById('btnCreateFolder');

  let renameTarget = null; // chat id OR 'FOLDER::name'

  // Helpers
  function span(cls, txt) { const s=document.createElement('span'); if(cls) s.className=cls; if(txt!=null) s.textContent=txt; return s; }
  function btn(icon, title, cls=''){ const b=document.createElement('button'); b.className=cls; b.title=title; b.type='button'; b.style.border='none'; b.style.background='transparent'; b.style.cursor='pointer'; b.innerHTML = `<i class="${icon}"></i>`; return b; }

  // Inline editing
  function makeInlineEditable(targetSpan, initial, onSave) {
    const input = document.createElement('input');
    input.type = 'text'; input.value = initial; input.className = 'inline-edit';
    targetSpan.replaceWith(input);
    input.focus({ preventScroll: true });

    const commit = () => {
      const v = input.value.trim() || initial;
      const s = document.createElement('span'); s.textContent = v;
      input.replaceWith(s);
      onSave(v, s);
    };
    const onKey = (e)=>{ if(e.key==='Enter') commit(); if(e.key==='Escape'){ input.value=initial; commit(); } };
    const onBlur = ()=> commit();

    input.addEventListener('keydown', onKey);
    input.addEventListener('blur', onBlur, { passive: true });
  }

  /* ---------------- Library ---------------- */
  function renderLibrary() {
    const convs = S.getConvs();
    libraryList.innerHTML = '';
    const frag = document.createDocumentFragment();

    convs.forEach(c => {
      const row = document.createElement('div');
      row.className = 'row';

      const left = document.createElement('div');
      left.className = 'left';
      const icon = document.createElement('i'); icon.className = 'ri-chat-1-line';
      const titleSpan = document.createElement('span');
      titleSpan.className = 'chat-title';
      titleSpan.textContent = c.title;
      titleSpan.dataset.id = c.id;
      titleSpan.style.cursor = 'pointer';

      left.append(icon, titleSpan);

      const actions = document.createElement('div');
      const rename = btn('ri-edit-2-line', 'Rename', 'rename'); rename.dataset.id = c.id;
      const del    = btn('ri-delete-bin-6-line', 'Delete', 'del'); del.dataset.id = c.id;
      actions.append(rename, del);

      row.append(left, actions);
      frag.appendChild(row);
    });

    libraryList.appendChild(frag);
  }

  // Delegation for library
  libraryList.addEventListener('click', (e)=>{
    const target = e.target.closest('button, .chat-title');
    if (!target) return;

    if (target.classList.contains('del')) {
      const id = target.dataset.id;
      const convs = S.getConvs().filter(c => c.id !== id);
      S.setConvs(convs);
      const f = S.getFolders();
      Object.keys(f).forEach(k => f[k] = f[k].filter(x => x !== id));
      S.setFolders(f);
      if (S.getCurrent() === id) S.setCurrent(convs[0]?.id || '');
      renderLibrary(); renderFolders(); global.ChatUI.loadCurrent();
      return;
    }

    if (target.classList.contains('rename')) {
      renameTarget = target.dataset.id;
      const conv = S.getConvs().find(c => c.id === renameTarget);
      openRenameModal('Rename chat', 'Confirm', conv?.title || '');
      return;
    }

    if (target.classList.contains('chat-title')) {
      S.setCurrent(target.dataset.id);
      global.ChatUI.loadCurrent();
    }
  }, { passive: true });

  // Inline rename on dblclick
  libraryList.addEventListener('dblclick', (e)=>{
    const s = e.target.closest('.chat-title');
    if (!s) return;
    const convs = S.getConvs();
    const conv = convs.find(c => c.id === s.dataset.id);
    makeInlineEditable(s, conv.title, (v, newSpan)=>{
      conv.title = v;
      S.setConvs(convs);
      newSpan.className = 'chat-title';
      newSpan.dataset.id = conv.id;
      newSpan.style.cursor = 'pointer';
    });
  });

  /* ---------------- Folders ---------------- */
  function renderFolders() {
    const f = S.getFolders();
    const convs = S.getConvs();
    foldersList.innerHTML = '';
    const frag = document.createDocumentFragment();

    Object.keys(f).forEach(name => {
      const folder = document.createElement('div');
      folder.className = 'folder';

      const left = document.createElement('div');
      left.className = 'left';
      const icon = document.createElement('i'); icon.className = 'ri-folder-2-line';
      const nameSpan = document.createElement('span'); nameSpan.textContent = name; nameSpan.style.cursor='pointer';
      left.append(icon, nameSpan);

      const right = document.createElement('div');
      right.className = 'right';
      const countEl = span(null, String(f[name].length)); countEl.style.opacity='.7';
      const rename = btn('ri-edit-2-line', 'Rename folder', 'rename-folder'); rename.dataset.name = name;
      const chev = document.createElement('span'); chev.className='chev'; chev.innerHTML='<i class="ri-arrow-down-s-line"></i>';
      right.append(countEl, rename, chev);

      folder.append(left, right);

      const sub = document.createElement('div'); sub.className='subchats';
      const subFrag = document.createDocumentFragment();
      f[name].forEach(id => {
        const c = convs.find(x => x.id === id);
        if (!c) return;
        const row = document.createElement('div'); row.className = 'sub';
        const icon = document.createElement('i'); icon.className='ri-chat-1-line';
        const title = document.createElement('span'); title.className='sub-open'; title.textContent = c.title; title.dataset.id = id;
        row.append(icon, title);
        subFrag.appendChild(row);
      });
      sub.appendChild(subFrag);

      frag.append(folder, sub);

      // Toggle
      let open = false;
      chev.addEventListener('click', ()=> {
        open = !open;
        sub.style.display = open ? 'flex' : 'none';
        chev.innerHTML = open ? '<i class="ri-arrow-up-s-line"></i>' : '<i class="ri-arrow-down-s-line"></i>';
      }, { passive:true });

      // Inline rename folder
      nameSpan.addEventListener('dblclick', ()=> {
        makeInlineEditable(nameSpan, name, (val)=> {
          if (!val || val === name) return;
          const folders = S.getFolders();
          if (!folders[val]) folders[val] = [];
          folders[val] = Array.from(new Set([...(folders[val]||[]), ...(folders[name]||[])]));
          delete folders[name];
          S.setFolders(folders);
          renderFolders();
        });
      });
    });

    foldersList.appendChild(frag);
  }

  // Delegation for folders
  foldersList.addEventListener('click', (e)=>{
    const openSub = e.target.closest('.sub-open');
    if (openSub) {
      S.setCurrent(openSub.dataset.id);
      global.ChatUI.loadCurrent();
      return;
    }
    const rename = e.target.closest('.rename-folder');
    if (rename) {
      renameTarget = 'FOLDER::' + rename.dataset.name;
      openRenameModal('Rename folder', 'Confirm', rename.dataset.name);
    }
  }, { passive: true });

  // Create folder button
  btnCreateFolder.addEventListener('click', ()=> {
    renameTarget = null;
    openRenameModal('Create folder','Create','');
  });
  btnCreateFolder.addEventListener('keydown', (e)=>{ if(e.key==='Enter'||e.key===' ') btnCreateFolder.click(); });

  /* ---------------- Modals ---------------- */
  function openAssignFolderModal() {
    const f = S.getFolders();
    const id = S.getCurrent();
    folderChecks.innerHTML = '';
    const frag = document.createDocumentFragment();

    if (Object.keys(f).length === 0) {
      const row = document.createElement('div');
      row.className = 'row';
      row.textContent = 'No folders yet. Click the plus next to “Folders” to create one.';
      frag.appendChild(row);
    } else {
      Object.keys(f).forEach(name => {
        const row = document.createElement('div'); row.className='row';
        const cb = document.createElement('input'); cb.type='checkbox'; cb.id='cb_'+name; cb.checked = f[name].includes(id);
        const lab = document.createElement('label'); lab.setAttribute('for','cb_'+name); lab.textContent = name;
        row.append(cb, lab); frag.appendChild(row);
      });
    }
    folderChecks.appendChild(frag);
    folderModal.style.display = 'flex';
    folderModal.setAttribute('aria-hidden', 'false');
  }
  function closeAssignFolderModal() {
    folderModal.style.display = 'none';
    folderModal.setAttribute('aria-hidden', 'true');
  }
  closeFolder.addEventListener('click', closeAssignFolderModal);
  saveFolder.addEventListener('click', ()=> {
    const f = S.getFolders(); const id = S.getCurrent();
    Object.keys(f).forEach(name => {
      const cb = document.getElementById('cb_'+name);
      if (!cb) return;
      if (cb.checked && !f[name].includes(id)) f[name].push(id);
      if (!cb.checked) f[name] = f[name].filter(x => x !== id);
    });
    S.setFolders(f); renderFolders(); closeAssignFolderModal();
  });
  folderModal.addEventListener('click', (e)=> { if(e.target === folderModal) closeAssignFolderModal(); }, { passive:true });

  function openRenameModal(title, cta, value) {
    renameTitleEl.textContent = title;
    renameConfirm.textContent = cta;
    renameInput.value = value || '';
    renameModal.style.display = 'flex';
    renameModal.setAttribute('aria-hidden','false');
    setTimeout(()=> renameInput.focus({ preventScroll: true }), 30);
  }
  function closeRenameModal() {
    renameModal.style.display = 'none';
    renameModal.setAttribute('aria-hidden','true');
  }
  renameCancel.addEventListener('click', closeRenameModal, { passive:true });
  renameModal.addEventListener('click', (e)=> { if(e.target === renameModal) closeRenameModal(); }, { passive:true });

  renameConfirm.addEventListener('click', ()=> {
    const val = (renameInput.value || '').trim();
    if (!val && renameTarget !== null) { closeRenameModal(); return; }

    if (renameTarget === null) {
      const f = S.getFolders();
      if (!f[val]) f[val] = [];
      S.setFolders(f); renderFolders(); closeRenameModal(); return;
    }

    if (String(renameTarget).startsWith('FOLDER::')) {
      const oldName = String(renameTarget).slice(8);
      const f = S.getFolders();
      if (!f[oldName]) { closeRenameModal(); return; }
      if (oldName !== val) {
        if (!f[val]) f[val] = [];
        f[val] = Array.from(new Set([...(f[val]||[]), ...(f[oldName]||[])]));
        delete f[oldName];
      }
      S.setFolders(f); renderFolders(); closeRenameModal(); return;
    }

    const convs = S.getConvs();
    const conv = convs.find(c => c.id === renameTarget);
    if (conv) { conv.title = val; S.setConvs(convs); renderLibrary(); }
    closeRenameModal();
  });
/* ---------------- Three-dots menu (portal + top-layer positioning) ---------------- */
const btnMore = document.getElementById('btnMore');
const moreMenu = document.getElementById('moreMenu');

function positionMenuToButton() {
  if (!btnMore || !moreMenu) return;

  // Ensure the menu is portaled to <body> so it's outside blurred/z-index contexts
  if (moreMenu.parentElement !== document.body) {
    document.body.appendChild(moreMenu);
  }

  const rect = btnMore.getBoundingClientRect();
  const offset = 8; // distance below button
  moreMenu.style.top = `${rect.bottom + offset}px`;

  // Measure menu width for proper alignment
  moreMenu.style.visibility = 'hidden';
  moreMenu.style.display = 'flex';
  const width = moreMenu.offsetWidth;
  moreMenu.style.display = 'none';
  moreMenu.style.visibility = 'visible';
  moreMenu.style.left = `${rect.right - width}px`;
}

if (btnMore && moreMenu) {
  btnMore.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOpen = moreMenu.classList.contains('open');

    // Close other open menus
    document.querySelectorAll('.menu.open').forEach(m => m.classList.remove('open'));

    if (!isOpen) {
      positionMenuToButton();
      moreMenu.classList.add('open');
      btnMore.setAttribute('aria-expanded', 'true');
    } else {
      moreMenu.classList.remove('open');
      btnMore.setAttribute('aria-expanded', 'false');
    }
  });

  // Reposition on resize/scroll if open
  window.addEventListener('resize', () => {
    if (moreMenu.classList.contains('open')) positionMenuToButton();
  });
  window.addEventListener('scroll', () => {
    if (moreMenu.classList.contains('open')) positionMenuToButton();
  }, { passive: true });

  // Close when clicking outside
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#btnMore') && !e.target.closest('#moreMenu')) {
      moreMenu.classList.remove('open');
      btnMore.setAttribute('aria-expanded', 'false');
    }
  });
}

/* ---------------- More Menu Item Actions ---------------- */
const miAddToFolder = document.getElementById('miAddToFolder');
if (miAddToFolder) {
  miAddToFolder.addEventListener('click', (e) => {
    e.stopPropagation();
    moreMenu.classList.remove('open');
    btnMore.setAttribute('aria-expanded', 'false');
    if (window.UI && typeof UI.openAssignFolderModal === 'function') {
      UI.openAssignFolderModal();
    }
  });
}

/* ---------------- Global click safety ---------------- */
document.addEventListener('click', (e) => {
  if (
    e.target.closest('.search-filter') ||
    e.target.closest('#moreMenu') ||
    e.target.closest('#btnMore') ||
    e.target.closest('#folderModal') ||
    e.target.closest('#renameModal')
  ) return;

  try {
    document.querySelectorAll('.menu.open, #userMenu.open')
      .forEach(m => m.classList.remove('open'));
  } catch (err) {
    console.warn(err);
  }
});

// Public API
global.UI = {
  renderLibrary,
  renderFolders,
  openAssignFolderModal,
  openCreateFolderModal: () => { renameTarget = null; openRenameModal('Create folder','Create',''); },
  setRenameTarget: (id) => { renameTarget = id; }
};
})(window);