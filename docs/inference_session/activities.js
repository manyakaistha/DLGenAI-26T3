/* Deterministic arithmetic activities. No model training, network calls or benchmark claims. */
(() => {
'use strict';
const root=document;
const cell=(n,on)=>`<span class="inf-cell ${on?'active':''}">${n}</span>`;
function renderTTA(el){
  let inverted=true;
  el.innerHTML='<div class="inf-controls"><label><input type="checkbox" checked> Apply spatial inverse</label><button type="button" data-reset>Reset</button></div><div class="inf-label">Original marker at column 1</div><div data-original class="inf-grid"></div><div class="inf-label">Flipped-view prediction after selected alignment</div><div data-view class="inf-grid"></div><p class="inf-readout" role="status"></p>';
  const draw=()=>{el.querySelector('[data-original]').innerHTML=Array.from({length:8},(_,i)=>cell(i,i===1)).join('');el.querySelector('[data-view]').innerHTML=Array.from({length:8},(_,i)=>cell(i,i===(inverted?1:6))).join('');el.querySelector('[role=status]').textContent=inverted?'Both views place the defect at column 1. Their mean stays at column 1.':'Without the inverse: averaging creates 0.5 at columns 1 and 6. At threshold 0.5, both become foreground.';};
  el.querySelector('input').onchange=e=>{inverted=e.target.checked;draw();};el.querySelector('[data-reset]').onclick=()=>{inverted=true;el.querySelector('input').checked=true;draw();};draw();
}
function renderEMA(el){
  let state=0, step=0;
  el.innerHTML='<div class="inf-controls"><label>Decay β <select><option value=".5">0.5</option><option value=".9">0.9</option><option value=".99">0.99</option></select></label><button type="button" data-step>Next update</button><button type="button" data-reset>Reset</button></div><div class="inf-label">EMA scalar</div><div class="inf-scalar" aria-live="polite">0.000</div><p class="inf-readout" role="status"></p>';
  const draw=()=>{el.querySelector('.inf-scalar').textContent=state.toFixed(3);el.querySelector('[role=status]').textContent=step===3?'All updates complete. Reset or choose a different decay.':`Completed ${step}/3 updates. Next parameter: ${[2,4,6][step]}.`;el.querySelector('[data-step]').disabled=step===3;};
  const reset=()=>{state=0;step=0;draw();};
  el.querySelector('[data-step]').onclick=()=>{if(step>=3)return;const beta=+el.querySelector('select').value;state=beta*state+(1-beta)*[2,4,6][step++];draw();};el.querySelector('[data-reset]').onclick=reset;el.querySelector('select').onchange=reset;draw();
}
function renderBlend(el){
  el.innerHTML='<div class="inf-controls"><label>Weight on model B <input type="range" min="0" max="1" step=".05" value=".75"><output>0.75</output></label><button type="button">Reset</button></div><div class="inf-label">Model A = 0.20, model B = 0.80</div><div class="inf-scalar" aria-live="polite"></div><p class="inf-readout" role="status"></p>';
  const slider=el.querySelector('input');const draw=()=>{const w=+slider.value,p=(1-w)*.2+w*.8;el.querySelector('output').textContent=w.toFixed(2);el.querySelector('.inf-scalar').textContent=p.toFixed(3);el.querySelector('[role=status]').textContent=`Weights A ${(1-w).toFixed(2)}, B ${w.toFixed(2)}. At threshold 0.5: ${p>=.5?'foreground':'background'}. These are toy probabilities.`;};slider.oninput=draw;el.querySelector('button').onclick=()=>{slider.value='.75';draw();};draw();
}
function renderThreshold(el){
  el.innerHTML='<div class="inf-controls"><label>Threshold <input type="range" min=".1" max=".9" step=".1" value=".5"><output>0.5</output></label><button type="button">Reset</button></div><div class="inf-label">Probabilities 0.8, 0.6, 0.4, 0.2 · Target 1, 1, 0, 0</div><div class="inf-grid four"></div><p class="inf-readout" role="status"></p>';
  const slider=el.querySelector('input');const draw=()=>{const t=+slider.value,ps=[.8,.6,.4,.2],ys=[1,1,0,0],pred=ps.map(p=>p>=t);let tp=0,fp=0,fn=0;pred.forEach((v,i)=>{tp+=+(v&&ys[i]);fp+=+(v&&!ys[i]);fn+=+(!v&&ys[i]);});el.querySelector('output').textContent=t.toFixed(1);el.querySelector('.inf-grid').innerHTML=pred.map(v=>cell(+v,v)).join('');el.querySelector('[role=status]').textContent=`TP ${tp} · FP ${fp} · FN ${fn} · Dice ${(2*tp/(2*tp+fp+fn)).toFixed(3)}. One synthetic positive mask.`;};slider.oninput=draw;el.querySelector('button').onclick=()=>{slider.value='.5';draw();};draw();
}
for(const el of root.querySelectorAll('[data-inference]'))({tta:renderTTA,ema:renderEMA,blend:renderBlend,threshold:renderThreshold}[el.dataset.inference])(el);
// The reused slide shell supplies quiz markup. Keep its answers keyboard-accessible.
root.addEventListener('click',e=>{const b=e.target.closest('[data-lab="quiz"] button');if(!b)return;const q=b.closest('[data-lab="quiz"]'),answer=q.querySelector('.answer');if(b.dataset.action==='reset'){answer.textContent='Choose an answer, then explain your reasoning.';answer.classList.remove('wrong');q.querySelectorAll('.answer-button').forEach(x=>x.removeAttribute('aria-pressed'));return;}const selected=Number(b.dataset.action.replace('answer-','')),correct=selected===Number(q.dataset.correct);answer.textContent=(correct?'Correct. ':'Try again. ')+q.dataset.explanation;answer.classList.toggle('wrong',!correct);q.querySelectorAll('.answer-button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));});
})();
