// main.js
// Boot + bindings. Passive listeners and no work when hidden.

(function () {
  const S = window.StorageAPI;
  const UI = window.UI;
  const Chat = window.ChatUI;

  const btnNewChat   = document.getElementById('btnNewChat');
  const userMenuBtn  = document.getElementById('userMenuBtn');
  const userMenu     = document.getElementById('userMenu');
  const logoutBtn    = document.getElementById('logoutBtn');

  // Ensure state exists
  S.ensureConversation();

  // Initial renders
  UI.renderLibrary();
  UI.renderFolders();
  Chat.loadCurrent();

  // ===== New Chat =====
  btnNewChat.addEventListener('click', () => {
    const convs = S.getConvs();
    const newId = 'c' + Date.now();
    const title = 'Chat ' + (convs.length + 1);
    convs.push({ id: newId, title, msgs: [] });
    S.setConvs(convs);
    S.setCurrent(newId);
    UI.renderLibrary();
    UI.renderFolders();
    Chat.loadCurrent();
  }, { passive: true });

  // ===== User Dropdown Menu =====
  if (userMenuBtn && userMenu && logoutBtn) {
    userMenuBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const isVisible = userMenu.style.display === 'flex';
      userMenu.style.display = isVisible ? 'none' : 'flex';
      userMenuBtn.setAttribute('aria-expanded', !isVisible);
    });

    document.addEventListener('click', (e) => {
      if (!userMenu.contains(e.target) && !userMenuBtn.contains(e.target)) {
        userMenu.style.display = 'none';
        userMenuBtn.setAttribute('aria-expanded', 'false');
      }
    }, { passive: true });

    logoutBtn.addEventListener('click', () => {
      userMenu.style.display = 'none';
      window.location.href = '/logout';
    });
  }

  // ===== Scroll Performance =====
  const scrollArea = document.getElementById('contentScroll');
  if (scrollArea) {
    scrollArea.addEventListener('wheel', () => {}, { passive: true });
  }

  // ===== Visibility Safety Net =====
  document.addEventListener('visibilitychange', () => {}, { passive: true });

  
  // ===== Custom Select (search filter) =====
  const customSelects = document.querySelectorAll('.custom-select');
  customSelects.forEach(sel => {
    const selected = sel.querySelector('.selected');
    const options = sel.querySelector('.options');

    sel.addEventListener('click', (e) => {
      e.stopPropagation();
      sel.classList.toggle('open');
    });

    options.querySelectorAll('li').forEach(opt => {
      opt.addEventListener('click', (e) => {
        e.stopPropagation();
        options.querySelectorAll('li').forEach(o => o.classList.remove('active'));
        opt.classList.add('active');
        selected.textContent = opt.textContent;
        sel.dataset.value = opt.dataset.value;
        sel.classList.remove('open');
      });
    });
  });
  
document.addEventListener("DOMContentLoaded", () => {
  const dropdown = document.querySelector("#searchFilterSelect");
  if (!dropdown) return;

  const selected = dropdown.querySelector(".selected");
  const options = dropdown.querySelector(".options");
  const items = options.querySelectorAll("li");

  selected.addEventListener("click", (e) => {
    e.stopPropagation();
    dropdown.classList.toggle("open");
  });

  items.forEach((li) => {
    li.addEventListener("click", (e) => {
      e.stopPropagation();
      items.forEach((i) => i.classList.remove("active"));
      li.classList.add("active");
      selected.textContent = li.textContent + " ▾";
      dropdown.dataset.value = li.dataset.value;
      dropdown.classList.remove("open");

      const input = document.querySelector("#searchInput");
      if (input) input.dispatchEvent(new Event("input"));
    });
  });

  document.addEventListener("click", (e) => {
    if (!dropdown.contains(e.target)) dropdown.classList.remove("open");
  });
});


  document.addEventListener('click', () => {
    document.querySelectorAll('.custom-select.open').forEach(s => s.classList.remove('open'));
  });
})();