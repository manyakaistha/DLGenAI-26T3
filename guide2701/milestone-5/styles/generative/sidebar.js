/* Shared progressive enhancement for every resource page with a sidebar. */
(() => {
  'use strict';
  const current = document.currentScript;
  if (!current || !document.querySelector('.layout > .sidebar')) return;
  const stylesheet = document.createElement('link');
  const url = new URL('sidebar.css', current.src);
  url.search = new URL(current.src).search;
  stylesheet.rel = 'stylesheet';
  stylesheet.href = url.href;
  // Leave normal navigation intact if its enhancement stylesheet cannot load.
  stylesheet.addEventListener('load', mount, {once: true});
  document.head.append(stylesheet);

  function mount() {
  const root = document.documentElement;
  const sidebar = document.querySelector('.layout > .sidebar');
  if (!sidebar || sidebar.querySelector('.sidebar-toggle')) return;
  const content = document.createElement('div');
  content.className = 'sidebar-body';
  content.id = 'guide-sidebar-content';
  while (sidebar.firstChild) content.append(sidebar.firstChild);
  const controls = document.createElement('div');
  controls.className = 'sidebar-controls';
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'sidebar-toggle';
  button.setAttribute('aria-controls', content.id);
  button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"></rect><path d="M9 4v16"></path><path class="sidebar-toggle-chevron" d="m15 9-3 3 3 3"></path></svg><span class="sidebar-toggle-label"></span>';
  controls.append(button);
  sidebar.append(controls, content);

  const key = 'guide-sidebar-state';
  const compact = matchMedia('(max-width: 1099px)');
  const mobile = matchMedia('(max-width: 850px)');
  let preference = null;
  try {
    const saved = localStorage.getItem(key);
    if (saved === 'collapsed' || saved === 'expanded') preference = saved;
  } catch { /* The disclosure also works when storage is unavailable. */ }
  let collapsed = preference ? preference === 'collapsed' : compact.matches;

  function render() {
    if (collapsed && content.contains(document.activeElement)) button.focus({preventScroll: true});
    root.dataset.sidebar = collapsed ? 'collapsed' : 'expanded';
    content.hidden = collapsed;
    button.setAttribute('aria-expanded', String(!collapsed));
    const label = collapsed ? 'Show sidebar' : 'Hide sidebar';
    button.setAttribute('aria-label', label);
    button.title = label;
    button.querySelector('.sidebar-toggle-label').textContent = label;
    // On phones, opening the disclosure exposes the links in a single step.
    const menu = content.querySelector('.mobile-menu');
    if (menu && mobile.matches) menu.open = !collapsed;
  }
  function choose(value) {
    collapsed = value;
    preference = value ? 'collapsed' : 'expanded';
    if (collapsed && content.contains(document.activeElement)) {
      button.focus({preventScroll: true});
    }
    try { localStorage.setItem(key, preference); } catch {}
    render();
  }
  button.addEventListener('click', () => choose(!collapsed));
  sidebar.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !collapsed && !event.defaultPrevented) {
      event.preventDefault();
      choose(true);
      button.focus({preventScroll: true});
    }
  });
  function resize() {
    if (preference === null) collapsed = compact.matches;
    render();
  }
  compact.addEventListener('change', resize);
  mobile.addEventListener('change', resize);
  addEventListener('storage', event => {
    if (event.key !== key && event.key !== null) return;
    preference = ['collapsed', 'expanded'].includes(event.newValue) ? event.newValue : null;
    collapsed = preference ? preference === 'collapsed' : compact.matches;
    if (collapsed && content.contains(document.activeElement)) button.focus({preventScroll: true});
    render();
  });
  render();
  button.hidden = false;
  root.classList.add('guide-sidebar-ready');
  }
})();
