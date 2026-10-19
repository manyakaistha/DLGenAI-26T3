/* Event-driven teaching labs. Fixed synthetic arrays, native inputs, no timers. */
(() => {
  'use strict';
  const W=72,H=48;
  const base=new Float32Array(W*H);
  for(let y=0;y<H;y++) for(let x=0;x<W;x++)
    base[y*W+x]=(((x-21)/12)**2+((y-17)/7)**2<1 || ((x-27)/8)**2+((y-24)/9)**2<1 || (x>49&&x<53&&y>29&&y<39))?1:0;
  function rng(seed){return ()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};}
  const control=(name,label,min,max,value,step=1)=>`<label>${label}<input data-key="${name}" type="range" min="${min}" max="${max}" value="${value}" step="${step}"></label>`;
  const select=(name,label,choices)=>`<label>${label}<select data-key="${name}">${choices.map(([v,t])=>`<option value="${v}">${t}</option>`).join('')}</select></label>`;
  const check=(name,label,on=false)=>`<label class="check"><input data-key="${name}" type="checkbox" ${on?'checked':''}> ${label}</label>`;
  const panel=(key,label)=>`<figure><figcaption>${label}</figcaption><canvas data-view="${key}" width="288" height="192" role="img" aria-label="${label}">${label}; see the current-values readout.</canvas></figure>`;
  function setup(el,title,html){el.innerHTML=`<h3>${title}</h3>${html}<output aria-live="polite" class="lab-readout"></output>`;return {q:s=>el.querySelector(s),value:k=>el.querySelector(`[data-key="${k}"]`).value,on:k=>el.querySelector(`[data-key="${k}"]`).checked,out:el.querySelector('output')};}
  function draw(cv,shape,{mask=null,brightness=1,maskOnly=false}={}){
    const ctx=cv.getContext('2d'),tmp=document.createElement('canvas');tmp.width=W;tmp.height=H;
    const tc=tmp.getContext('2d'),im=tc.createImageData(W,H);
    for(let y=0;y<H;y++)for(let x=0;x<W;x++){
      const i=y*W+x,noise=((x*7+y*11)%13)-6,v=shape[i];
      let color=maskOnly?[255*v,255*v,255*v]:[35+145*v+noise,90+115*v+noise,135+75*v+noise].map(a=>Math.max(0,Math.min(255,a*brightness)));
      if(mask&&mask[i]>0&&((x+y)%4<2)){
        const boundary=x===0||y===0||x===W-1||y===H-1||!mask[i-1]||!mask[i+1]||!mask[i-W]||!mask[i+W];
        if(boundary)color=[255,210,20];
      }
      im.data.set([...color,255],i*4);
    }
    tc.putImageData(im,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(tmp,0,0,cv.width,cv.height);
  }
  function transform(kind,dx=0){const a=new Float32Array(base.length);for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    const sx=kind==='hflip'?W-1-x:kind==='translate'?x-dx:x,sy=kind==='vflip'?H-1-y:y;
    a[y*W+x]=sx>=0&&sx<W?base[sy*W+sx]:0;
  }return a;}
  const labs={
    geometry(el){const s=setup(el,'Keep the boundary attached',`<div class="lab-controls">${select('kind','Spatial operation',[['hflip','Horizontal flip'],['vflip','Vertical flip'],['translate','Horizontal translation'],['none','No movement']])}${control('dx','Translation in pixels',-20,20,12)}${check('sync','Apply the same operation to masks',true)}<button type="button" data-reset>Reset</button></div><div class="lab-panels">${panel('original','Original + original mask')}${panel('result','Transformed image + supplied mask')}</div>`);
      const render=()=>{const expected=transform(s.value('kind'),+s.value('dx')),mask=s.on('sync')?expected:base;let inter=0,union=0;for(let i=0;i<mask.length;i++){inter+=!!(mask[i]&&expected[i]);union+=!!(mask[i]||expected[i]);}const score=union?inter/union:1;draw(s.q('[data-view="original"]'),base,{mask:base});draw(s.q('[data-view="result"]'),expected,{mask});s.out.textContent=`Operation: ${s.value('kind')}; translation: ${s.value('dx')} px (used only for translation). Alignment IoU: ${score.toFixed(3)}. Dashed yellow boundary = supplied mask. ${score===1?'Image and label agree.':'Label is detached from the transformed cloud.'}`;el.dataset.iou=score.toFixed(3);};
      el.addEventListener('input',render);el.addEventListener('change',render);s.q('[data-reset]').onclick=()=>{s.q('[data-key="kind"]').value='hflip';s.q('[data-key="dx"]').value=12;s.q('[data-key="sync"]').checked=true;render();};render();
    },
    brightness(el){const s=setup(el,'Appearance changes; labels stay binary',`<div class="lab-controls">${control('factor','Brightness multiplier',0.2,1.8,1,0.1)}${check('bug','Deliberate bug: multiply mask values too')}<button type="button" data-reset>Reset</button></div><div class="lab-panels">${panel('image','Image after brightness change')}${panel('mask','Target mask: black = 0, white = 1')}</div>`);
      const render=()=>{const factor=+s.value('factor'),value=s.on('bug')?factor:1,mask=Float32Array.from(base,x=>x*value);draw(s.q('[data-view="image"]'),base,{brightness:factor});draw(s.q('[data-view="mask"]'),mask,{maskOnly:true});s.out.textContent=`Brightness: ${factor.toFixed(1)}×. Mask values: {0, ${value.toFixed(1)}}. ${value===1?'Binary target preserved.':'Corrupted target: foreground must remain 1 in this hard-label lesson.'} White display clips above 1; read the numeric value.`;el.dataset.foreground=value;};
      el.addEventListener('input',render);el.addEventListener('change',render);s.q('[data-reset]').onclick=()=>{s.q('[data-key="factor"]').value=1;s.q('[data-key="bug"]').checked=false;render();};render();
    },
    interpolation(el){const s=setup(el,'Inspect a resized label pixel',`<div class="lab-controls">${select('method','Mask interpolation',[['nearest','Nearest-neighbor'],['bilinear','Bilinear (deliberate mistake)']])}${control('row','Selected output row',0,15,7)}${control('col','Selected output column',0,15,7)}<button type="button" data-reset>Reset</button></div><div class="lab-panels"><figure><figcaption>Source binary mask: 4 × 4</figcaption><canvas data-source width="256" height="256" role="img" aria-label="Four by four diagonal binary mask">Four by four identity matrix.</canvas></figure><figure><figcaption>Resized mask: 16 × 16; outline = selected pixel</figcaption><canvas data-target width="256" height="256" role="img" aria-label="Resized mask, inspect with row and column sliders">Use row and column sliders to inspect numeric output.</canvas></figure></div>`);
      const source=(x,y)=>x===y?1:0,clamp=x=>Math.max(0,Math.min(3,x));
      const render=()=>{const method=s.value('method'),r=+s.value('row'),c=+s.value('col'),values=[];
        const ctx=s.q('[data-source]').getContext('2d');for(let y=0;y<4;y++)for(let x=0;x<4;x++){ctx.fillStyle=source(x,y)?'#fff':'#12283a';ctx.fillRect(x*64,y*64,64,64);}
        const out=s.q('[data-target]').getContext('2d');for(let y=0;y<16;y++)for(let x=0;x<16;x++){
          const sx=clamp((x+0.5)/4-0.5),sy=clamp((y+0.5)/4-0.5),x0=Math.floor(sx),y0=Math.floor(sy),fx=sx-x0,fy=sy-y0;
          const v=method==='nearest'?source(Math.floor(x/4),Math.floor(y/4)):(1-fx)*(1-fy)*source(x0,y0)+fx*(1-fy)*source(clamp(x0+1),y0)+(1-fx)*fy*source(x0,clamp(y0+1))+fx*fy*source(clamp(x0+1),clamp(y0+1));
          values.push(v);const shade=Math.round(v*255);out.fillStyle=`rgb(${shade},${shade},${shade})`;out.fillRect(x*16,y*16,16,16);
        }
        out.strokeStyle='#e86435';out.lineWidth=3;out.strokeRect(c*16+1.5,r*16+1.5,13,13);
        const fraction=values.filter(v=>v>0&&v<1).length,selected=values[r*16+c];
        s.out.textContent=`Method: ${method}. Pixel [row ${r}, column ${c}] = ${selected.toFixed(4)}. Fractional-valued pixels: ${fraction} / 256. ${fraction?'Boundaries contain mixed labels.':'Every output label remains 0 or 1.'}`;el.dataset.fractional=fraction;el.dataset.selected=selected;
      };el.addEventListener('input',render);el.addEventListener('change',render);s.q('[data-reset]').onclick=()=>{s.q('[data-key="method"]').value='nearest';s.q('[data-key="row"]').value=7;s.q('[data-key="col"]').value=7;render();};render();
    },
    probability(el){const s=setup(el,'Draw decisions, not a fixed quota',`<div class="lab-controls">${control('p','Probability of horizontal flip',0,1,0.5,0.05)}<button type="button" data-draw>Draw next 20 fetches</button><button type="button" data-reset>Reset sequence</button></div><div class="decision-grid" aria-label="Twenty sample fetch decisions"></div>`);let random=rng(42),draws=0;
      const render=()=>{const p=+s.value('p');draws++;const items=Array.from({length:20},()=>random()<p);const grid=s.q('.decision-grid');grid.replaceChildren(...items.map((flip,i)=>{const b=document.createElement('span');b.className=flip?'decision flipped':'decision';b.textContent=`${i+1}: ${flip?'↔ flip':'keep'}`;return b;}));s.out.textContent=`p = ${p.toFixed(2)}. Draw ${draws}: ${items.filter(Boolean).length} of 20 flipped. Expected count = ${(20*p).toFixed(1)}, not a guaranteed count. Dataset length is unchanged.`;el.dataset.flips=items.filter(Boolean).length;};s.q('[data-draw]').onclick=render;s.q('[data-reset]').onclick=()=>{random=rng(42);draws=0;render();};el.addEventListener('input',render);render();
    },
    batches(el){const s=setup(el,'Build a batch from numbered samples',`<div class="lab-controls">${control('size','Batch size',1,8,4)}${check('drop','Drop the incomplete final batch')}${check('shuffle','Shuffle sample order')}<button type="button" data-next>New epoch</button><button type="button" data-reset>Reset</button></div><div class="batch-list"></div>`);let epoch=1;
      const render=()=>{const size=+s.value('size'),random=rng(41+epoch),order=Array.from({length:19},(_,i)=>i);if(s.on('shuffle'))for(let i=18;i>0;i--){const j=Math.floor(random()*(i+1));[order[i],order[j]]=[order[j],order[i]];}const kept=s.on('drop')?19-19%size:19,groups=[];for(let i=0;i<kept;i+=size)groups.push(order.slice(i,Math.min(i+size,kept)));const list=s.q('.batch-list');list.replaceChildren(...groups.map((g,i)=>{const div=document.createElement('div');div.className='batch-row';const label=document.createElement('strong');label.textContent=`Batch ${i+1}`;div.append(label);g.forEach(n=>{const tile=document.createElement('span');tile.className='sample-tile';tile.textContent=String(n).padStart(2,'0');div.append(tile);});return div;}));s.out.textContent=`Epoch ${epoch}: ${groups.length} batches; batch sizes ${groups.map(g=>g.length).join(', ')}. ${kept} samples kept; ${19-kept} skipped. One full image batch: [${size}, 3, 60, 90]. Mask channels always remain 4.`;el.dataset.batches=groups.length;el.dataset.kept=kept;};el.addEventListener('input',render);el.addEventListener('change',render);s.q('[data-next]').onclick=()=>{epoch++;render();};s.q('[data-reset]').onclick=()=>{epoch=1;s.q('[data-key="size"]').value=4;s.q('[data-key="drop"]').checked=false;s.q('[data-key="shuffle"]').checked=false;render();};render();
    }
  };
  document.querySelectorAll('[data-session-lab]').forEach(el=>{const start=labs[el.dataset.sessionLab];if(start)start(el);});
})();
