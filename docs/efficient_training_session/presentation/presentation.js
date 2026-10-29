/* Interaction contract: fixed, deterministic toy data; native controls; semantic
   diagrams. Only user-started convolution scans and scalar updates animate.
   A finite timeline settles on navigation, interruption, or reduced motion. */
(() => {
  'use strict';
  const $ = (s, root=document) => root.querySelector(s);
  const $$ = (s, root=document) => [...root.querySelectorAll(s)];
  const notesBase = '../';
  const escape = v => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const btn = (text, action, cls='') => `<button type="button" data-action="${action}" class="${cls}">${text}</button>`;
  const jump = (text, id) => `<button type="button" data-jump="${id}">${text}</button>`;
  const reset = () => btn('Reset', 'reset');
  const panel = (content, cls='') => `<div class="panel ${cls}">${content}</div>`;
  const callout = text => `<div class="callout">${text}</div>`;
  const quiz = (question, options, correct, explanation) => `<div data-lab="quiz" data-correct="${correct}" data-explanation="${escape(explanation)}"><div class="question">${question}</div><div class="quiz-options" style="margin-top:18px">${options.map((text,i)=>btn(text,`answer-${i}`,'answer-button')).join('')}</div><div class="answer" role="status">Choose an answer, then explain your reasoning.</div>${reset()}</div>`;
  // Explanations have a deliberate reading order: takeaway, reasoning, then caveat.
  const lesson = (label, title, rows, foot='') => `<aside class="lesson-notes"><div class="notes-kicker">${label}</div><h3 class="notes-title">${title}</h3><dl class="notes-reasoning">${rows.map(([label,text])=>`<div><dt>${label}</dt><dd>${text}</dd></div>`).join('')}</dl>${foot?`<div class="notes-foot">${foot}</div>`:''}</aside>`;
/* SESSION_CONTENT */
  $('#slides').innerHTML=slides.map((s,i)=>`<section id="${s.id}" class="slide ${s.hero?'hero':''}" aria-labelledby="title-${i}" ${i?'hidden inert':''}>${s.hero?'':`<div class="eyebrow">${escape(s.chapter)}</div><h2 id="title-${i}" tabindex="-1">${escape(s.title)}</h2>${s.lede?`<p class="lede">${s.lede}</p>`:''}`} ${s.body}</section>`).join('');
  $('#start h1').id='title-0'; $('#start h1').tabIndex=-1;
  const chapters=[['01','Precision','precision'],['02','AMP','amp'],['03','Recover','checkpoint'],['04','Evidence','validation']];
  $('#chapter').classList.add('sr-only');
  $('#chapter').insertAdjacentHTML('afterend',`<nav class="chapter-nav" aria-label="Jump to a chapter">${chapters.map(([n,label,id])=>`<button data-jump="${id}"><span>${n}</span> ${label}</button>`).join('')}</nav>`);
  const themes=['paper','midnight','moss','butter','blush','mono'];
  const themeRoot=document.documentElement;
  function applyTheme(name){
    if(!themes.includes(name))name='paper';
    themeRoot.dataset.theme=name;
    $$('.theme-option').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.palette===name)));
    try{localStorage.setItem('guide-garden-theme',name);}catch{}
  }
  try{applyTheme(localStorage.getItem('guide-garden-theme')||themeRoot.dataset.theme||'paper');}catch{applyTheme(themeRoot.dataset.theme||'paper');}
  $$('.theme-option').forEach(b=>b.addEventListener('click',()=>applyTheme(b.dataset.palette)));
  let current=0, includeOptional=false, motionOff=matchMedia('(prefers-reduced-motion: reduce)').matches;
  let pending=null;
  const media=matchMedia('(prefers-reduced-motion: reduce)');
  function settle(){if(pending){const p=pending;pending=null;cancelAnimationFrame(p.frame);p.finish();}}
  function animate(duration,draw,finish=()=>{}){
    settle();
    if(motionOff||document.hidden){draw(1);finish();return;}
    const birth=performance.now();
    const animation={frame:0,finish:()=>{draw(1);finish();}};
    pending=animation;
    const tick=now=>{if(pending!==animation)return;const u=Math.min(1,(now-birth)/duration);draw(u);if(u<1)animation.frame=requestAnimationFrame(tick);else{pending=null;finish();}};
    animation.frame=requestAnimationFrame(tick);
  }
  const route=()=>slides.map((s,i)=>!s.optional||includeOptional?i:null).filter(i=>i!==null);
  function go(target,focus=true){
    const n=typeof target==='string'?slides.findIndex(s=>s.id===target):target;
    if(n<0||n>=slides.length)return;
    settle();current=n;
    $$('.slide').forEach((el,i)=>{el.classList.remove('is-entering');el.hidden=i!==n;el.inert=i!==n;});
    if(focus&&!motionOff) { void $$('.slide')[n].offsetWidth; $$('.slide')[n].classList.add('is-entering'); }
    const chapter=slides[n].chapter.match(/^0[1-4]/)?.[0];
    $$('.chapter-nav button').forEach((button,i)=>{if(chapters[i][0]===chapter)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');});
    $('#chapter').textContent=slides[n].chapter;
    const path=route(),position=path.indexOf(n);
    $('#counter').textContent=position<0?`Optional · ${slides.filter(s=>s.optional).findIndex(s=>s.id===slides[n].id)+1}/${slides.filter(s=>s.optional).length}`:`${String(position+1).padStart(2,'0')} / ${path.length}`;

    const next=path.find(i=>i>n),prev=path.filter(i=>i<n).at(-1);
    $('#previous').disabled=prev===undefined;$('#next').disabled=next===undefined;
    $('#next').title=next===undefined?'End of session':`Next: ${slides[next].title}`;
    $('#previous').title=prev===undefined?'Start of session':`Previous: ${slides[prev].title}`;
    $('#notes').href=notesBase+slides[n].note;
    $('#progress').style.width=`${(position<0?(n+1)/slides.length:(position+1)/path.length)*100}%`;
    try{history.replaceState(null,'',`#${slides[n].id}`);}catch{/* Embedded/offline viewers may restrict URL changes. */}
    if(focus)$(`#title-${n}`).focus({preventScroll:true});
    if(innerWidth<=700)scrollTo({top:0,behavior:'instant'});
    $$('#index-list button').forEach(b=>b.setAttribute('aria-current',b.dataset.jump===slides[n].id?'page':'false'));
  }
  function move(direction){let n=current+direction;while(n>=0&&n<slides.length&&!includeOptional&&slides[n].optional)n+=direction;go(n);}
  $('#previous').onclick=()=>move(-1);$('#next').onclick=()=>move(1);
  const dialog=$('#index-dialog');
  $('#contents').onclick=()=>{settle();$('#slide-search').value='';filterIndex();dialog.showModal();$('#slide-search').focus();};$('#close-index').onclick=()=>dialog.close();
  $('#include-optional').onchange=e=>{includeOptional=e.target.checked;go(current,false);};
  const indexGroups=[['Opening',[slides[0]]],...chapters.map(([n,label])=>[`${n} / ${label}`,slides.filter(s=>s.chapter.startsWith(n))]),['Optional / Beyond the core',slides.filter(s=>s.optional)]];
  $('#index-list').innerHTML=indexGroups.map(([label,items])=>`<section class="index-group"><h3>${label}</h3><div>${items.map(s=>`<button type="button" data-jump="${s.id}" data-search="${escape((s.title+' '+s.chapter+' '+(s.lede||'')+' '+s.body.replace(/<[^>]*>/g,' ')).toLowerCase())}" class="${s.optional?'optional':''}"><span>${String(slides.indexOf(s)+1).padStart(2,'0')}</span><strong>${escape(s.title)}</strong><i aria-hidden="true">↗</i></button>`).join('')}</div></section>`).join('');
  function filterIndex(){
    const terms=$('#slide-search').value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    $$('#index-list button').forEach(b=>b.hidden=!terms.every(term=>b.dataset.search.includes(term)));
    $$('.index-group').forEach(group=>group.hidden=!$$('button',group).some(b=>!b.hidden));
    $('#index-empty').hidden=$$('#index-list button').some(b=>!b.hidden);
  }
  $('#slide-search').addEventListener('input',filterIndex);
  document.addEventListener('click',e=>{const target=e.target.closest('[data-jump]');if(target){if(dialog.open)dialog.close();go(target.dataset.jump);}});
  function motionUI(){ themeRoot.dataset.motion=motionOff?'off':'on'; $('#motion').textContent=motionOff?'Motion off':'Motion on';$('#motion').setAttribute('aria-pressed',String(motionOff)); }
  $('#motion').onclick=()=>{motionOff=!motionOff;settle();motionUI();};
  media.addEventListener('change',e=>{motionOff=e.matches;settle();motionUI();});motionUI();
  document.addEventListener('visibilitychange',()=>{if(document.hidden)settle();});
  window.addEventListener('pagehide',settle);
  async function fullscreen(){try{if(document.fullscreenElement)await document.exitFullscreen();else await $('#viewport').requestFullscreen();}catch{$('#announcer').textContent='Fullscreen is unavailable here. Open the presentation in its own browser tab or use browser fullscreen.';}}
  $('#fullscreen').onclick=fullscreen;
  document.addEventListener('fullscreenchange',()=>{$('#fullscreen').textContent=document.fullscreenElement?'Exit fullscreen':'Fullscreen';fit();});
  document.addEventListener('keydown',e=>{
    if(e.altKey||e.ctrlKey||e.metaKey||e.isComposing||dialog.open)return;
    const palette=$('.palette-menu');
    if(palette.open){if(e.key==='Escape'){palette.open=false;$('summary',palette).focus();}return;}
    const editing=e.target.closest('input,select,textarea,[contenteditable]');
    if(editing)return;
    if(e.key===' '&&e.target.closest('button,a'))return;
    if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();move(1);}
    else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();move(-1);}
    else if(e.key==='Home'){e.preventDefault();go(0);}
    else if(e.key==='End'){e.preventDefault();go(slides.length-1);}
    else if(e.key.toLowerCase()==='i'){$('#contents').click();}
    else if(e.key.toLowerCase()==='f')fullscreen();
  });
  window.addEventListener('hashchange',()=>{const id=location.hash.slice(1);if(slides.some(s=>s.id===id))go(id,false);});
  function fit(){
    const stage=$('#stage');
    if(innerWidth<=700){for(const key of ['transform','position','left','top','marginLeft','marginTop'])stage.style[key]='';$('#viewport').style.height='';return;}
    const scale=Math.min(innerWidth/1280,innerHeight/720);
    stage.style.transform=`scale(${scale})`;
    // The unscaled stage must not enlarge the document's scrollable area.
    $('#viewport').style.height=`${innerHeight}px`;
    stage.style.position='absolute';stage.style.left='50%';stage.style.top='50%';
    stage.style.marginLeft='-640px';stage.style.marginTop='-360px';
  }
  window.addEventListener('resize',fit);fit();
  try{history.scrollRestoration='manual';}catch{}
  if(innerWidth<=700)setTimeout(()=>scrollTo(0,0),80);


  window.Presentation=Object.freeze({go, slides:slides.map(({id,title,optional})=>({id,title,optional:!!optional})),get current(){return slides[current].id;}});
  $('.index-bottom a[download]').href=location.href.split('#')[0];
  const hash=location.hash.slice(1);go(slides.some(s=>s.id===hash)?hash:0,false);
})();
// The presentation uses the same seeded garden language as the reading site:
// miniature stems, leaves and spiral blooms form a quiet footer signature.
// Geometry is generated once, then
// rendered deterministically at the current time.
(() => {
  'use strict';
  const canvas = document.querySelector('#botanical-canvas');
  const motionButton = document.querySelector('#motion');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const root = document.documentElement;
  let width = 0, height = 0, dpr = 1, visible = true, frame = 0;
  let lastNow = performance.now(), elapsed = 0;
  let palette = {};

  function mulberry32(seed) {
    let state = seed | 0;
    return () => {
      state = (state + 0x6D2B79F5) | 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const rand = mulberry32(261026);
  // A miniature specimen group, confined to the footer's empty centre.
  // Use the reading site's flower contour/spiral, at a much quieter scale.
  const plants = [[.27,.38,-.08],[.52,.24,.04],[.76,.48,.08]]
    .map(([x,y,bend]) => ({
      x, y, bend,
      curve: [[x+bend,.96],[x+bend,.72],[x-.025,.54],[x,y]],
      radius: 5 + rand() * 1.7,
      phase: rand() * Math.PI * 2,
      petals: 5 + Math.floor(rand() * 3),
      leaves: 2,
      depth: .78
    }));
  const butterflies = [[.11,.30,6.3],[.90,.24,5.7]].map(([x,y,size]) => ({
    x,y,size,phase:rand()*Math.PI*2,tilt:-.3+rand()*.4
  }));
  // Cache the course butterfly silhouette and its inlaid panels once.
  const wings = new Path2D();
  wings.moveTo(.04,-.12);
  wings.bezierCurveTo(.26,-.78,.78,-1.18,1.04,-.92);
  wings.bezierCurveTo(1.26,-.63,.88,-.1,.16,.13);
  wings.bezierCurveTo(.68,-.06,1.02,.36,.72,.7);
  wings.bezierCurveTo(.44,.97,.1,.57,.04,.17);wings.closePath();
  const wingPanels = new Path2D();
  wingPanels.moveTo(.2,-.18);wingPanels.bezierCurveTo(.4,-.62,.8,-.91,.91,-.77);
  wingPanels.bezierCurveTo(.98,-.62,.65,-.27,.2,-.18);wingPanels.closePath();
  wingPanels.moveTo(.45+.17*Math.cos(-.4),.38+.17*Math.sin(-.4));
  wingPanels.ellipse(.45,.38,.17,.23,-.4,0,Math.PI*2);
  const wingVeins = new Path2D();
  wingVeins.moveTo(.08,.02);wingVeins.quadraticCurveTo(.45,-.38,.86,-.76);
  wingVeins.moveTo(.08,.12);wingVeins.quadraticCurveTo(.3,.3,.53,.56);
  for (const p of plants) {
    p.contour = Array.from({length:73}, (_, i) => {
      const a = i / 72 * Math.PI * 2;
      const radius = p.radius * (1 + .13 * Math.sin(p.petals * a + p.phase));
      return [Math.cos(a) * radius, Math.sin(a) * radius * .84];
    });
    p.spiral = Array.from({length:86}, (_, i) => {
      const u = i / 85, a = u * Math.PI * 6 + p.phase;
      const radius = p.radius * (.06 + u * .61);
      return [Math.cos(a) * radius, Math.sin(a) * radius * .7];
    });
  }

  const clamp = (v, lo=0, hi=1) => Math.max(lo, Math.min(hi, v));
  const easeOut = t => 1 - Math.pow(1 - clamp(t), 3);
  const spring = t => 1 - Math.exp(-6 * clamp(t)) * Math.cos(10 * clamp(t));
  const cubic = (points, t) => {
    const u = 1 - t;
    return [
      u*u*u*points[0][0] + 3*u*u*t*points[1][0] + 3*u*t*t*points[2][0] + t*t*t*points[3][0],
      u*u*u*points[0][1] + 3*u*u*t*points[1][1] + 3*u*t*t*points[2][1] + t*t*t*points[3][1]
    ];
  };
  function readPalette() {
    const styles = getComputedStyle(root);
    palette = Object.fromEntries(['line','stem','flower','flower-line'].map(name => [name, styles.getPropertyValue(`--${name}`).trim()]));
  }
  function resize() {
    width = Math.max(1, canvas.clientWidth || 112);
    height = Math.max(1, canvas.clientHeight || 42);
    dpr = Math.min(devicePixelRatio || 1, 2.5);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    readPalette();
    repaint();
  }
  function drawLeaf(x, y, angle, length, growth, alpha) {
    if (growth <= 0) return;
    ctx.save();
    ctx.translate(x, y); ctx.rotate(angle); ctx.scale(growth, growth);
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.bezierCurveTo(length*.2, -length*.6, length*.8, -length*.45, length, 0);
    ctx.bezierCurveTo(length*.7, length*.3, length*.2, length*.35, 0, 0);
    ctx.fillStyle = palette.stem; ctx.fill();
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(length*.84, 0);
    ctx.strokeStyle = palette['flower-line']; ctx.lineWidth = .65; ctx.stroke();
    ctx.restore();
  }
  function drawBloom(p, x, y, scale, time, alpha) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(Math.sin(time * .00045 + p.phase) * .07);
    ctx.scale(scale, scale);
    ctx.globalAlpha = alpha;
    // A scalloped outer contour matches the reading site's inked blossoms.
    ctx.beginPath();
    for (let i = 0; i < p.contour.length; i++) {
      const [px, py] = p.contour[i];
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = palette.flower;
    ctx.strokeStyle = palette['flower-line'];
    ctx.lineWidth = 1.1;
    ctx.fill();
    // Fine spiral linework keeps the blooms from reading as flat stickers.
    ctx.beginPath();
    for (let i = 0; i < p.spiral.length; i++) {
      const [px, py] = p.spiral[i];
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = palette['flower-line']; ctx.lineWidth = .85; ctx.stroke();
    ctx.restore();
  }
  function drawButterfly(b,time,animate) {
    const flutter=animate?.76+.24*Math.pow(Math.sin(time*.005+b.phase),2):.94;
    const dx=animate?Math.sin(time*.0007+b.phase)*1.6:0;
    const dy=animate?Math.cos(time*.0009+b.phase)*1.1:0;
    ctx.save();ctx.translate(b.x*width+dx,b.y*height+dy);
    ctx.rotate(b.tilt);ctx.scale(b.size,b.size);ctx.globalAlpha=.88;
    ctx.lineJoin='round';ctx.lineCap='round';
    for(const side of [-1,1]) {
      ctx.save();ctx.scale(side*flutter,1);
      ctx.fillStyle=palette.flower;ctx.strokeStyle=palette.stem;ctx.lineWidth=.075;
      ctx.fill(wings);ctx.stroke(wings);
      ctx.fillStyle=palette['flower-line'];ctx.fill(wingPanels);
      ctx.lineWidth=.035;ctx.stroke(wingVeins);ctx.restore();
    }
    ctx.fillStyle=palette.stem;ctx.beginPath();ctx.ellipse(0,.07,.07,.47,0,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.arc(0,-.43,.09,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle=palette.stem;ctx.lineWidth=.05;ctx.beginPath();
    ctx.moveTo(-.03,-.46);ctx.quadraticCurveTo(-.1,-.75,-.27,-.78);
    ctx.moveTo(.03,-.46);ctx.quadraticCurveTo(.1,-.75,.27,-.78);ctx.stroke();ctx.restore();
  }
  function draw(now) {
    frame = 0;
    const dt = Math.min(64, Math.max(0, now - lastNow));
    lastNow = now;
    const motionOn = !reduced.matches && motionButton?.getAttribute('aria-pressed') !== 'true';
    if (motionOn && visible && !document.hidden) elapsed += dt;
    const time = motionOn ? elapsed : 0;
    const grow = motionOn ? easeOut(elapsed / 1100) : 1;
    ctx.clearRect(0, 0, width, height);

    for (const p of plants) {
      const sway = motionOn ? Math.sin(time * .0006 + p.phase) * Math.min(.55, width * .004) : 0;

      const x = p.x * width + sway;
      const y = p.y * height;
      // The normalized curve is seeded once; only a subpixel sway changes.
      const path = p.curve.map(([px, py], i) => [
        px * width + (i > 1 ? sway : sway * .35),
        py * height
      ]);
      const stemGrowth = clamp(grow * 1.14);
      ctx.globalAlpha = p.depth;
      ctx.beginPath();
      for (let i = 0; i <= 36; i++) {
        const point = cubic(path, i / 36 * stemGrowth);
        if (i === 0) ctx.moveTo(point[0], point[1]); else ctx.lineTo(point[0], point[1]);
      }
      ctx.strokeStyle = palette.stem; ctx.lineWidth = .8; ctx.lineCap = 'round'; ctx.stroke();
      for (let j = 0; j < p.leaves; j++) {
        const t = .28 + j * .27;
        if (t > stemGrowth) continue;
        const point = cubic(path, t);
        const side = j % 2 ? -.58 : -2.46;
        drawLeaf(point[0], point[1], side, 5 + j * .8, clamp((stemGrowth - t) * 5), p.depth * .72);
      }
      const bloomProgress = clamp((grow - .52) / .48);
      if (bloomProgress > 0) drawBloom(p, x, y, spring(bloomProgress), time, p.depth);
    }
    for (const b of butterflies) drawButterfly(b,time,motionOn);
    if (motionOn && visible && !document.hidden) frame = requestAnimationFrame(draw);
  }
  function repaint() {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    readPalette();
    draw(performance.now());
  }
  new ResizeObserver(resize).observe(canvas);
  new IntersectionObserver(entries => {
    visible = entries[0]?.isIntersecting ?? true;
    if (visible) repaint(); else if (frame) { cancelAnimationFrame(frame); frame = 0; }
  }, {threshold: 0}).observe(canvas);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { if (frame) cancelAnimationFrame(frame); frame = 0; }
    else if (visible) repaint();
  });
  reduced.addEventListener('change', repaint);
  new MutationObserver(repaint).observe(root, {attributes:true, attributeFilter:['data-theme']});
  if (motionButton) new MutationObserver(repaint).observe(motionButton, {attributes:true, attributeFilter:['aria-pressed']});
  window.addEventListener('resize', resize);
  resize();
})();
