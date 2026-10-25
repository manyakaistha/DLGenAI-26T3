
const inputs = document.querySelectorAll('.search');
let index;
inputs.forEach(input => input.addEventListener('input', async () => {
  const results = input.nextElementSibling;
  const query = input.value.trim().toLowerCase();
  results.replaceChildren();
  if (query.length < 2) return;
  try {
    index ||= await (await fetch('search.json')).json();
    const matches = index.filter(p => (p.title+' '+p.text).toLowerCase().includes(query));
    for (const p of matches) {
      const a = document.createElement('a'); a.href=p.url; a.textContent=p.title; results.append(a);
    }
    if (!matches.length) results.textContent='No matching pages.';
  } catch { results.textContent='Search unavailable. Use the guide navigation below.'; }
}));
document.querySelectorAll('article table').forEach(table => {
 const wrapper=document.createElement('div'); wrapper.className='table-scroll';
 wrapper.tabIndex=0; wrapper.setAttribute('role','region'); wrapper.setAttribute('aria-label','Scrollable table');
 table.before(wrapper); wrapper.append(table);
});
