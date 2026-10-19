/* Botanical controls: seeded rose geometry, finite spring morphs, SVG lettering. */
(() => {
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const states=new Set(),moving=new Set();
  let frame=0,serial=0;
  const svg=(tag,attributes={})=>{const el=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attributes))el.setAttribute(k,v);return el;};
  function rng(seed){return ()=>{seed=(seed+0x6D2B79F5)|0;let t=Math.imul(seed^seed>>>15,1|seed);t=(t+Math.imul(t^t>>>7,61|t))^t;return ((t^t>>>14)>>>0)/4294967296;};}
  function path(points,close=false){return points.map(([x,y],i)=>(i?'L':'M')+x.toFixed(3)+','+y.toFixed(3)).join(' ')+(close?'Z':'');}
  // The same lobed polar curve and three-turn spiral used in the guide header.
  function rose(seed,radius=20){
    const random=rng(seed),phase=random()*Math.PI*2,petals=5+Math.floor(random()*3);
    const outline=Array.from({length:97},(_,i)=>{const a=i/96*Math.PI*2,r=radius*(1+.15*Math.sin(petals*a+phase));return [Math.cos(a)*r,Math.sin(a)*r*.84];});
    const bud=Array.from({length:97},(_,i)=>{const a=i/96*Math.PI*2,r=radius*.66*(1+.055*Math.sin(petals*a+phase));return [Math.cos(a)*r,Math.sin(a)*r*.84];});
    const spiral=Array.from({length:85},(_,i)=>{const u=i/84,a=u*Math.PI*6+phase,r=radius*(.06+u*.62);return [Math.cos(a)*r,Math.sin(a)*r*.7];});
    return {outline,bud,spiral};
  }
  function ornament(seed,radius=20){
    const g=svg('g',{'class':'ai-rose'}),leaf=svg('g',{'class':'ai-rose-leaves'});
    leaf.append(svg('path',{d:'M-8 8 C-27 7 -27 18 -25 21 C-13 22 -5 13 -8 8Z'}),svg('path',{d:'M8 8 C24 5 28 12 28 17 C16 21 9 15 8 8Z'}));
    const contour=svg('path',{'class':'ai-rose-contour'}),spiral=svg('path',{'class':'ai-rose-spiral'});
    g.append(leaf,contour,spiral);
    const geometry=rose(seed,radius);
    spiral.setAttribute('d',path(geometry.spiral));
    return {g,geometry,contour,spiral,leaf};
  }
  function apply(state,value){
    const v=Math.max(-.08,Math.min(1.16,value));state.value=v;
    state.flower.contour.setAttribute('d',path(state.flower.geometry.bud.map((p,i)=>[p[0]+(state.flower.geometry.outline[i][0]-p[0])*v,p[1]+(state.flower.geometry.outline[i][1]-p[1])*v]),true));
    state.flower.spiral.setAttribute('transform',`scale(${.66+.34*v})`);
    state.flower.leaf.setAttribute('transform',`scale(${.70+.30*v})`);
    state.flower.leaf.style.opacity=String(.35+.65*Math.max(0,v));
    state.button.style.setProperty('--bloom',v.toFixed(3));
  }
  function motionAllowed(){return !reduced.matches&&document.documentElement.dataset.askAiMotion!=='off'&&!document.hidden;}
  function finish(){cancelAnimationFrame(frame);frame=0;for(const s of moving)apply(s,s.target);moving.clear();}
  function tick(now){
    frame=0;
    if(!motionAllowed()){finish();return;}
    for(const s of moving){
      const u=Math.min(1,(now-s.birth)/s.duration);
      // Finite, normalized damped spring; snap to the exact endpoint on completion.
      const ease=s.target>0?1-Math.exp(-7*u)*Math.cos(11*u):1-Math.pow(1-u,3);
      apply(s,u===1?s.target:s.from+(s.target-s.from)*ease);
      if(u===1)moving.delete(s);
    }
    if(moving.size)frame=requestAnimationFrame(tick);
  }
  function to(state,target){
    if(state.target===target&&!moving.has(state))return;
    state.from=state.value;state.target=target;state.birth=performance.now();state.duration=target?760:440;
    if(!motionAllowed()||!state.visible){apply(state,target);moving.delete(state);return;}
    moving.add(state);if(!frame)frame=requestAnimationFrame(tick);
  }
  const observer=new IntersectionObserver(entries=>entries.forEach(e=>{const s=[...states].find(s=>s.button===e.target);if(s){s.visible=e.isIntersecting;if(!s.visible){moving.delete(s);apply(s,s.target);}}}),{threshold:.05});
  function decorateFlower(button,seed){
    const icon=svg('svg',{viewBox:'0 0 64 64','class':'ai-bloom-icon','aria-hidden':'true',focusable:'false'});
    const flower=ornament(seed),position=svg('g',{transform:'translate(32 31)'});
    position.append(flower.g);icon.append(position);
    const center=svg('text',{x:32,y:34.5,'class':'ai-bloom-monogram','text-anchor':'middle'});center.textContent='AI';
    // A quiet paper disk keeps the monogram legible through the spiral.
    icon.append(svg('circle',{cx:32,cy:31,r:7.5,'class':'ai-bloom-center'}),center);
    const label=document.createElement('span');label.className='ai-bloom-label';label.textContent='Ask AI';
    button.replaceChildren(icon,label);
    const state={button,flower,value:0,target:0,from:0,visible:true,hover:false,focus:false};states.add(state);apply(state,0);observer.observe(button);
    const update=()=>to(state,state.hover||state.focus?1:0);
    button.addEventListener('pointerenter',e=>{if(e.pointerType==='mouse'){state.hover=true;update();}});
    button.addEventListener('pointerleave',()=>{state.hover=false;update();});
    button.addEventListener('pointercancel',()=>{state.hover=false;update();});
    button.addEventListener('focus',()=>{state.focus=true;update();});
    button.addEventListener('blur',()=>{state.focus=false;update();});
    return state;
  }
  function decorateStamp(button){
    const icon=svg('svg',{viewBox:'0 0 128 128','class':'ai-stamp','aria-hidden':'true',focusable:'false'});
    const id='ask-ai-rim-'+(++serial),defs=svg('defs');
    defs.append(svg('path',{id,d:'M64 17.5 A46.5 46.5 0 1 1 63.999 17.5'}));icon.append(defs);
    const edge=[];
    for(let i=0;i<=192;i++){const a=i/192*Math.PI*2,r=59.5+1.1*Math.sin(24*a);edge.push([64+Math.cos(a)*r,64+Math.sin(a)*r]);}
    icon.append(svg('path',{d:path(edge,true),'class':'ai-stamp-paper'}),svg('circle',{cx:64,cy:64,r:55,'class':'ai-stamp-keyline'}));
    const orbit=svg('g',{'class':'ai-stamp-orbit'}),lettering=svg('text',{'class':'ai-stamp-lettering',textLength:286,lengthAdjust:'spacing'}),onPath=svg('textPath',{href:'#'+id,startOffset:'0%'});
    lettering.append(onPath);orbit.append(lettering);icon.append(orbit);
    const rosePosition=svg('g',{transform:'translate(64 57)'}),flower=ornament(2701,20);
    flower.contour.setAttribute('d',path(flower.geometry.outline,true));flower.leaf.setAttribute('transform','translate(0 2) scale(.84)');rosePosition.append(flower.g);icon.append(rosePosition);
    const word=svg('text',{x:64,y:91,'text-anchor':'middle','class':'ai-stamp-word'});word.textContent='Ask AI';icon.append(word);
    const hint=document.createElement('span');hint.className='ai-stamp-hint';hint.textContent='Explain this page';
    button.replaceChildren(icon,hint);setStamp(button,false);
  }
  function setStamp(button,selected){
    button.dataset.selection=String(selected);
    button.querySelector('textPath').textContent=selected?'ASK AI  ·  EXPLAIN MY SELECTION  ·':'ASK AI  ·  LET CURIOSITY BLOOM  ·';
    button.querySelector('.ai-stamp-hint').textContent=selected?'Explain selected text':'Explain this page';
    button.setAttribute('aria-label',selected?'Ask AI about selected text':'Ask AI to explain this page. Select text to ask about a passage.');
  }
  function syncMotion(){
    const toggle=document.querySelector('.motion-toggle');
    document.documentElement.dataset.askAiMotion=reduced.matches||(toggle&&toggle.getAttribute('aria-pressed')==='false')?'off':'on';
    if(!motionAllowed())finish();
  }
  const toggle=document.querySelector('.motion-toggle');if(toggle)new MutationObserver(syncMotion).observe(toggle,{attributes:true,attributeFilter:['aria-pressed']});
  reduced.addEventListener('change',syncMotion);
  document.addEventListener('visibilitychange',()=>{document.documentElement.dataset.askAiHidden=String(document.hidden);if(document.hidden)finish();});
  syncMotion();
  window.AskAIBotanicals={decorateFlower,decorateStamp,setStamp};
})();
