(() => {
  'use strict';
  const root = document.documentElement;
  const themes = ['paper', 'midnight', 'moss', 'butter', 'blush', 'mono'];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let motion = !reduced.matches;
  try { motion = localStorage.getItem('guide-garden-motion') !== 'off' && !reduced.matches; } catch {}
  let repaint = () => {};
  const toggle = document.querySelector('.motion-toggle');
  function updateControls() {
    document.querySelectorAll('.theme-option').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.palette === root.dataset.theme)));
    toggle.textContent = reduced.matches ? 'Motion off (system preference)' : motion ? 'Pause header motion' : 'Enable header motion';
    toggle.setAttribute('aria-pressed', String(motion));
    toggle.disabled = reduced.matches;
  }
  document.querySelectorAll('.theme-option').forEach(b => b.addEventListener('click', () => {
    if (!themes.includes(b.dataset.palette)) return;
    root.dataset.theme = b.dataset.palette;
    try { localStorage.setItem('guide-garden-theme', root.dataset.theme); } catch {}
    updateControls(); repaint();
  }));
  toggle.addEventListener('click', () => {
    motion = !motion;
    try { localStorage.setItem('guide-garden-motion', motion ? 'on' : 'off'); } catch {}
    updateControls(); repaint();
  });
  reduced.addEventListener('change', () => { motion = !reduced.matches; updateControls(); repaint(); });
  updateControls();
  const progress = document.querySelector('.reading-progress');
  const updateProgress = () => {
    const total = document.documentElement.scrollHeight - innerHeight;
    progress.style.width = `${total > 0 ? Math.min(100, scrollY / total * 100) : 0}%`;
  };
  addEventListener('scroll', updateProgress, {passive:true});
  addEventListener('resize', updateProgress); updateProgress();
  const canvas = document.querySelector('.garden-hero canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  let width = 0, height = 0, visible = true, frame = 0, birth = performance.now(), palette;
  function rng(seed) {
    return () => {
      seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = (t + Math.imul(t ^ t >>> 7, 61 | t)) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  const rand = rng(260310);
  const plants = Array.from({length:7}, (_,i) => ({
    x:.51+i*.063, y:.24+rand()*.43, bend:(rand()-.5)*.16,
    radius:12+rand()*10, phase:rand()*Math.PI*2,
    petals:5+Math.floor(rand()*3), leaves:3+Math.floor(rand()*2)
  }));
  const pixels = Array.from({length:28}, () => ({x:.48+rand()*.5,y:.12+rand()*.75,size:2+rand()*4}));
  function colors() {
    const css = getComputedStyle(root);
    palette = Object.fromEntries(['ink','line','stem','flower','flower-line'].map(k=>[k,css.getPropertyValue('--'+k).trim()]));
  }
  function size() {
    const r = canvas.getBoundingClientRect(); width = r.width; height = r.height;
    const dpr = Math.min(devicePixelRatio || 1, 2.5);
    canvas.width = Math.round(width*dpr); canvas.height = Math.round(height*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0); colors(); repaint();
  }
  const cubic = (p,t) => {
    const u=1-t;
    return [u*u*u*p[0][0]+3*u*u*t*p[1][0]+3*u*t*t*p[2][0]+t*t*t*p[3][0],u*u*u*p[0][1]+3*u*u*t*p[1][1]+3*u*t*t*p[2][1]+t*t*t*p[3][1]];
  };
  function leaf(x,y,angle,length,growth) {
    ctx.save(); ctx.translate(x,y); ctx.rotate(angle); ctx.scale(growth,growth);
    ctx.beginPath(); ctx.moveTo(0,0); ctx.bezierCurveTo(length*.2,-length*.6,length*.8,-length*.45,length,0);ctx.bezierCurveTo(length*.7,length*.3,length*.2,length*.35,0,0);
    ctx.fillStyle=palette.stem;ctx.fill();ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(length*.84,0);ctx.strokeStyle=palette['flower-line'];ctx.lineWidth=.6;ctx.stroke();ctx.restore();
  }
  function bloom(p,x,y,scale,time) {
    ctx.save(); ctx.translate(x,y);ctx.rotate(Math.sin(time*.0004+p.phase)*.08);ctx.scale(scale,scale);
    ctx.beginPath();
    for(let i=0;i<=72;i++) {
      const a=i/72*Math.PI*2, r=p.radius*(1+.13*Math.sin(p.petals*a+p.phase));
      const px=Math.cos(a)*r,py=Math.sin(a)*r*.84;
      if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);
    }
    ctx.closePath();ctx.fillStyle=palette.flower;ctx.fill();ctx.strokeStyle=palette['flower-line'];ctx.lineWidth=.9;
    ctx.beginPath();
    for(let i=0;i<85;i++) {
      const u=i/84,a=u*Math.PI*6+p.phase,r=p.radius*(.06+u*.62);
      const px=Math.cos(a)*r,py=Math.sin(a)*r*.7;
      if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);
    }
    ctx.stroke();ctx.restore();
  }
  // Small visitors stay in the header margins, clear of readable text.
  function butterfly(x,y,size,phase,time,animate) {
    // Separate forewings/hindwings with patterned panels and visible veins.
    const flutter=animate?.72+.28*Math.abs(Math.sin(time*.0045+phase)):.94;
    ctx.save();ctx.translate(x,y);ctx.rotate(-.22+Math.sin(time*.0007+phase)*.12);
    ctx.scale(size,size);ctx.lineJoin='round';ctx.lineCap='round';
    for(const side of [-1,1]) {
      ctx.save();ctx.scale(side*flutter,1);
      ctx.fillStyle=palette.flower;ctx.strokeStyle=palette.stem;ctx.lineWidth=.065;
      ctx.beginPath();ctx.moveTo(.04,-.12);
      ctx.bezierCurveTo(.26,-.78,.78,-1.18,1.04,-.92);
      ctx.bezierCurveTo(1.26,-.63,.88,-.1,.16,.13);
      ctx.bezierCurveTo(.68,-.06,1.02,.36,.72,.7);
      ctx.bezierCurveTo(.44,.97,.1,.57,.04,.17);
      ctx.closePath();ctx.fill();ctx.stroke();
      // Pale inset panels, as on a painted butterfly's wings.
      ctx.fillStyle=palette['flower-line'];
      ctx.beginPath();ctx.moveTo(.2,-.18);
      ctx.bezierCurveTo(.4,-.62,.8,-.91,.91,-.77);
      ctx.bezierCurveTo(.98,-.62,.65,-.27,.2,-.18);ctx.fill();
      ctx.beginPath();ctx.ellipse(.45,.38,.17,.23,-.4,0,Math.PI*2);ctx.fill();
      ctx.strokeStyle=palette.stem;ctx.lineWidth=.035;
      ctx.beginPath();ctx.moveTo(.08,.02);ctx.quadraticCurveTo(.45,-.38,.86,-.76);
      ctx.moveTo(.42,-.36);ctx.lineTo(.65,-.68);
      ctx.moveTo(.08,.12);ctx.quadraticCurveTo(.3,.3,.53,.56);ctx.stroke();
      // Tiny edge spots echo the flower linework.
      ctx.fillStyle=palette['flower-line'];
      for(const [px,py] of [[.94,-.5],[.79,-.24],[.66,.65]]) {
        ctx.beginPath();ctx.arc(px,py,.047,0,Math.PI*2);ctx.fill();
      }
      ctx.restore();
    }
    ctx.fillStyle=palette.stem;
    ctx.beginPath();ctx.ellipse(0,.07,.07,.47,0,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.arc(0,-.43,.09,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle=palette.stem;ctx.lineWidth=.045;
    ctx.beginPath();ctx.moveTo(-.03,-.46);ctx.quadraticCurveTo(-.1,-.75,-.27,-.78);
    ctx.moveTo(.03,-.46);ctx.quadraticCurveTo(.1,-.75,.27,-.78);ctx.stroke();
    ctx.restore();
  }
  function draw(now) {
    frame=0;ctx.clearRect(0,0,width,height);
    const animate=motion&&!reduced.matches;
    const u=animate?Math.max(0,Math.min(1,(now-birth)/1300)):1;
    const grow=1-Math.pow(1-u,3),time=animate?now:0;
    ctx.strokeStyle=palette.line;ctx.lineWidth=.6;ctx.globalAlpha=.55;
    for(let i=0;i<5;i++){ctx.beginPath();ctx.moveTo(width*.48,height*(.18+i*.16));ctx.lineTo(width,height*(.18+i*.16));ctx.stroke();}
    ctx.fillStyle=palette.stem;ctx.globalAlpha=.18;
    for(const p of pixels)ctx.fillRect(p.x*width,p.y*height,p.size,p.size);
    ctx.globalAlpha=1;
    for(const p of plants) {
      const sway=animate?Math.sin(time*.0006+p.phase)*3:0;
      const x=p.x*width+sway,y=p.y*height;
      const path=[[p.x*width+p.bend*width,height+5],[x+p.bend*width,height*.64],[x-20,height*.45],[x,y]];
      ctx.beginPath();
      for(let i=0;i<=35;i++){const q=cubic(path,i/35*grow);if(i===0)ctx.moveTo(...q);else ctx.lineTo(...q);}
      ctx.strokeStyle=palette.stem;ctx.lineWidth=1.5;ctx.stroke();
      for(let j=0;j<p.leaves;j++) {
        const t=.17+j*.16;if(t>grow)continue;
        const q=cubic(path,t);leaf(q[0],q[1],j%2?-.55:-2.4,14+j*2,Math.min(1,(grow-t)*5));
      }
      const f=Math.max(0,Math.min(1,(grow-.58)/.42));
      const spring=f>=1?1:1-Math.exp(-6*f)*Math.cos(10*f);
      if(f>0)bloom(p,x,y,spring*(width<500?.8:1),time);
    }
    if(grow>.8) {
      const small=width<500?11:15;
      // One beside the title, two amongst the blossoms. Slow, bounded flight.
      const visitors=[[.425,.14,1.4],[plants[2].x+.025,plants[2].y-.1,3.2],[plants[5].x+.035,plants[5].y-.08,5.1]];
      for(const [x,y,phase] of visitors) {
        const dx=animate?Math.sin(time*.00045+phase)*8:0;
        const dy=animate?Math.cos(time*.00065+phase)*5:0;
        butterfly(width*x+dx,height*y+dy,small,phase,time,animate);
      }
    }
    if(animate&&visible&&!document.hidden)frame=requestAnimationFrame(draw);
  }
  repaint=()=>{
    if(frame)cancelAnimationFrame(frame);frame=0;colors();
    draw(performance.now());
  };
  new ResizeObserver(size).observe(canvas);
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)repaint();else if(frame){cancelAnimationFrame(frame);frame=0;}},{threshold:0}).observe(canvas);
  document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(frame);frame=0;}else if(visible)repaint();});
  size();
})();
