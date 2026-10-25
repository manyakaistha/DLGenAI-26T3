/* Exact small teaching calculations. No model training, network calls or idle RAF. */
(() => {
  'use strict';
  const range=(key,label,min,max,value,step=1)=>`<label>${label}<input data-key="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"></label>`;
  const select=(key,label,choices)=>`<label>${label}<select data-key="${key}">${choices.map(([v,t])=>`<option value="${v}">${t}</option>`).join('')}</select></label>`;
  const check=(key,label,on=false)=>`<label class="check"><input data-key="${key}" type="checkbox" ${on?'checked':''}> ${label}</label>`;
  const reset='<button type="button" data-reset>Reset lab</button>';
  const panel=(key,title)=>`<figure><figcaption>${title}</figcaption><canvas data-view="${key}" width="256" height="256" role="img" aria-label="${title}">${title}; numerical results are in the text readout.</canvas></figure>`;
  function setup(el,title,context,body){
    el.innerHTML=`<h3>${title}</h3><p class="lab-context">${context}</p>${body}<output class="mri-readout" aria-live="polite"></output>`;
    const controls=[...el.querySelectorAll('input,select')],defaults=controls.map(c=>({value:c.value,checked:c.checked}));
    const s={q:q=>el.querySelector(q),value:key=>el.querySelector(`[data-key="${key}"]`).value,on:key=>el.querySelector(`[data-key="${key}"]`).checked,out:el.querySelector('output')};
    s.listen=render=>{el.addEventListener('input',render);el.addEventListener('change',render);s.q('[data-reset]').addEventListener('click',()=>{controls.forEach((c,i)=>{c.value=defaults[i].value;c.checked=defaults[i].checked;});render();});render();};
    return s;
  }
  const labs={
    split(el){
      const s=setup(el,'Split slices or split patients?','Fictional patients, four slices each. T = training; V = validation. Case-level ratios are rounded up for validation.',`<div class="mri-controls">${select('mode','Split unit',[['patient','Whole patients'],['slice','Slices within each patient (leakage)']])}${range('patients','Number of patients',2,8,4)}${reset}</div><div class="mri-grid"></div>`);
      s.listen(()=>{const n=+s.value('patients'),mode=s.value('mode'),nVal=Math.ceil(n*.2);let train=0,val=0,overlap=0;
        const rows=[];for(let p=0;p<n;p++){
          const choices=Array.from({length:4},(_,i)=>mode==='patient'?p>=n-nVal:i===3),mixed=choices.some(Boolean)&&choices.some(x=>!x);if(mixed)overlap++;
          val+=choices.filter(Boolean).length;train+=choices.filter(x=>!x).length;
          rows.push(`<div class="mri-patient"><strong>Patient ${String.fromCharCode(65+p)}${mixed?' · appears on BOTH sides':''}</strong><div class="mri-slices">${choices.map((v,i)=>`<span class="mri-chip ${v?'val':''}">${i+1}: ${v?'V':'T'}</span>`).join('')}</div></div>`);
        }
        s.q('.mri-grid').innerHTML=rows.join('');s.out.textContent=`${train} training slices; ${val} validation slices. Patient overlap: ${overlap}. ${overlap?'Validation contains patients already seen during training. A plausible slice ratio does not make it independent.':`All ${nVal} validation patients are unseen in training. Case and slice proportions can differ in a real dataset.`}`;el.dataset.overlap=overlap;
      });
    },
    shapes(el){
      const s=setup(el,'Trace a skip connection','Exact shape arithmetic for the teaching ResNet18 U-Net. B is the batch dimension; these examples use dimensions divisible by 32.',`<div class="mri-controls">${select('size','Input height × width',[['256,256','256 × 256 MRI'],['128,128','128 × 128 classroom'],['128,800','128 × 800 industrial strip']])}${select('join','Decoder join',[['0','Deepest → s4'],['1','Next → s3'],['2','Next → s2'],['3','Next → s1']])}${reset}</div><div class="mri-stages"></div><p class="mri-equation"></p>`);
      s.listen(()=>{const [h,w]=s.value('size').split(',').map(Number),j=+s.value('join'),channels=[64,64,128,256,512],skip=3-j,upChannels=[512,256,128,64][j],outChannels=[256,128,64,32][j],shapes=channels.map((c,i)=>[c,h/2**(i+1),w/2**(i+1)]);
        s.q('.mri-stages').innerHTML=shapes.map((shape,i)=>`<div class="mri-stage ${i===skip?'active':''}"><strong>s${i+1}</strong>[B, ${shape.join(', ')}]</div>`).join('');
        const [sc,sh,sw]=shapes[skip],joined=upChannels+sc;
        s.q('.mri-equation').textContent=`Resize decoder feature to ${sh} × ${sw} → concatenate ${upChannels} + ${sc} = ${joined} channels → ConvBlock(${joined}, ${outChannels}).`;
        s.out.textContent=`Input [B, 3, ${h}, ${w}]. The selected skip is s${skip+1}: [B, ${sc}, ${sh}, ${sw}]. Concatenation changes channels, not height/width. Final model output: [B, 1, ${h}, ${w}] logits. The industrial task would instead require four output channels.`;el.dataset.joined=joined;
      });
    },
    comparison(el){
      const s=setup(el,'Audit the experiment before comparing scores','Arm A is random-initialized ResNet18 U-Net, saved patient split, 2 epochs, BCE–Dice, three sequences, threshold 0.5 and full fine-tuning. Arm B starts with ImageNet weights.',`<div class="mri-controls">${select('split','Arm B split',[['same','Same saved patients'],['different','Different patient split']])}${select('epochs','Arm B budget',[['same','Same 2 epochs'],['different','10 epochs']])}${select('loss','Arm B loss',[['same','Same BCE–Dice'],['different','Dice only']])}${select('input','Arm B input',[['same','Same three sequences'],['different','FLAIR repeated']])}${select('threshold','Arm B threshold',[['same','Same 0.5'],['different','0.3']])}${select('policy','Arm B training policy',[['same','Same full fine-tuning'],['different','Frozen encoder']])}${check('leak','Patients overlap between training and validation')}${reset}</div><div class="mri-equation"></div>`);
      s.listen(()=>{const names={split:'patient split',epochs:'training budget',loss:'loss',input:'input representation',threshold:'decision threshold',policy:'training policy'},changes=Object.keys(names).filter(k=>s.value(k)!=='same').map(k=>names[k]),leak=s.on('leak'),valid=!changes.length&&!leak;
        s.q('.mri-equation').textContent=valid?'Intended difference: encoder initialization only.':'Additional problems: '+[...changes, ...(leak?['patient leakage']:[])].join('; ')+'.';
        s.out.textContent=valid?'This design can investigate initialization under the fixed budget. You still need measured results and repeated runs for stronger evidence. No score is invented by this lab.':`Do not attribute an observed difference solely to pretraining. ${leak?'The held-out-patient claim is also invalid. ':''}Restore matching settings, then run the models.`;el.dataset.valid=String(valid);
      });
    },
    channels(el){
      const s=setup(el,'Three channels: sequence meaning and normalization','Synthetic intensities in [0,1], not patient measurements. Mean/std are the fixed torchvision ImageNet constants.',`<div class="mri-controls">${select('mode','Input representation',[['sequences','Three supplied sequences'],['repeat','Repeat FLAIR three times']])}${range('pre','Pre-contrast intensity',0,1,.25,.01)}${range('flair','FLAIR intensity',0,1,.6,.01)}${range('post','Post-contrast intensity',0,1,.4,.01)}${check('normalize','Apply channel normalization',true)}${reset}</div><div class="mri-channel-grid"></div>`);
      s.listen(()=>{const repeat=s.value('mode')==='repeat',raw=repeat?Array(3).fill(+s.value('flair')):[+s.value('pre'),+s.value('flair'),+s.value('post')],means=[.485,.456,.406],stds=[.229,.224,.225],names=repeat?['FLAIR copy 1','FLAIR copy 2','FLAIR copy 3']:['Pre-contrast','FLAIR','Post-contrast'],z=raw.map((x,i)=>s.on('normalize')?(x-means[i])/stds[i]:x);
        s.q('.mri-channel-grid').innerHTML=raw.map((x,i)=>`<div class="mri-channel"><strong>Channel ${i}: ${names[i]}</strong><div class="intensity" style="opacity:${.15+.85*x}"></div><span>Raw: ${x.toFixed(2)}</span><span>Model value: ${z[i].toFixed(3)}</span></div>`).join('');
        s.out.textContent=`${repeat?'FLAIR is copied; pre/post sliders do not affect the input.':'The three sequence slots retain different meanings.'} ${s.on('normalize')?'z=(x−mean)/std. Negative normalized values are expected. Identical raw copies can normalize differently by channel.':'No normalization applied in this toy calculation.'} The binary target is never normalized.`;el.dataset.values=z.map(x=>x.toFixed(3)).join(',');
      });
    },
    freeze(el){
      const s=setup(el,'A frozen weight can coexist with moving buffers','One illustrative SGD step: weight=1, gradient=0.4; BatchNorm running mean=0, batch mean=2, momentum=0.1. This exact scalar calculation is not a neural-network training simulator.',`<div class="mri-controls">${check('grad','Encoder weight requires gradients',true)}${check('optimizer','Encoder weight is in optimizer',true)}${check('bn','Encoder BatchNorm is in train mode',true)}${range('lr','SGD learning rate',0,.5,.1,.01)}${reset}</div><div class="mri-weight-list"></div>`);
      s.listen(()=>{const gradient=s.on('grad'),member=s.on('optimizer'),bn=s.on('bn'),lr=+s.value('lr'),weight=gradient&&member?1-lr*.4:1,mean=bn?.2:0;
        s.q('.mri-weight-list').innerHTML=`<div class="mri-weight"><strong>Encoder parameter: 1.000 → ${weight.toFixed(3)}</strong>${gradient?'Gradient computed: 0.4.':'Gradient absent (None).'} ${member?'Optimizer includes the parameter.':'Optimizer excludes the parameter.'}</div><div class="mri-weight"><strong>Running-mean buffer: 0.000 → ${mean.toFixed(3)}</strong>${bn?'0.9 × 0 + 0.1 × 2 = 0.2 in train mode.':'Evaluation mode keeps the running buffer fixed.'}</div>`;
        s.out.textContent=`Weight update ${weight===1?'does not occur':'occurs'}. Buffer update ${bn?'occurs':'does not occur'}. To keep this encoder fixed, freeze its parameters, use the intended optimizer membership and set encoder.eval() after model.train() each epoch. Real AdamW includes momentum/adaptive state and weight decay; this lab deliberately uses simple SGD arithmetic.`;el.dataset.weight=weight.toFixed(3);el.dataset.buffer=mean.toFixed(3);
      });
    },
    threshold(el){
      const s=setup(el,'Threshold a fixed probability field','Synthetic 8 × 8 arrays. Left: probabilities; right: predicted mask, with reference cells outlined. Not MRI predictions. Both-empty Dice=1; one-empty Dice=0.',`<div class="mri-controls">${range('threshold','Probability threshold',0,1,.5,.05)}${select('target','Reference target',[['positive','A small foreground region'],['empty','Empty target']])}${reset}</div><div class="mri-panels">${panel('probability','Fixed synthetic probabilities (lighter = higher)')}${panel('prediction','Binary prediction; orange outlines = target')}</div>`);
      const probability=[],target=[];for(let y=0;y<8;y++)for(let x=0;x<8;x++){probability.push(.05+.85*Math.exp(-((x-3.6)**2+(y-3.1)**2)/5));target.push(x>=2&&x<=4&&y>=2&&y<=4);}
      s.listen(()=>{const threshold=+s.value('threshold'),empty=s.value('target')==='empty',predicted=probability.map(p=>p>=threshold),truth=target.map(t=>!empty&&t);let tp=0,fp=0,fn=0;
        predicted.forEach((p,i)=>{tp+=Number(p&&truth[i]);fp+=Number(p&&!truth[i]);fn+=Number(!p&&truth[i]);});
        const denominator=2*tp+fp+fn,dice=denominator?2*tp/denominator:1;
        ['probability','prediction'].forEach(view=>{const ctx=s.q(`[data-view="${view}"]`).getContext('2d');ctx.clearRect(0,0,256,256);for(let i=0;i<64;i++){const x=(i%8)*32,y=Math.floor(i/8)*32,value=view==='probability'?probability[i]:Number(predicted[i]),shade=Math.round(20+220*value);ctx.fillStyle=`rgb(${shade},${shade},${shade})`;ctx.fillRect(x,y,32,32);ctx.strokeStyle='#808080';ctx.lineWidth=.5;ctx.strokeRect(x,y,32,32);if(view==='prediction'&&truth[i]){ctx.strokeStyle='#ef7548';ctx.lineWidth=3;ctx.strokeRect(x+2,y+2,28,28);}}});
        s.out.textContent=`Threshold ${threshold.toFixed(2)}. TP=${tp}, FP=${fp}, FN=${fn}. Hard Dice=${dice.toFixed(3)}. ${empty?'Empty target: any predicted pixel is a false positive.':'Target contains 9 foreground cells.'} Thresholding changes the decision, not the fixed probabilities or the soft training objective.`;el.dataset.dice=dice.toFixed(3);el.dataset.fp=fp;
      });
    },
    provenance(el){
      const s=setup(el,'Which weights are being reused?','Compare starting states, not brand names. The main notebook uses torchvision ImageNet encoder weights and a new decoder.',`<div class="mri-controls">${select('source','Checkpoint source',[['random','No pretraining'],['imagenet','ImageNet encoder'],['mri','MRI-trained full U-Net with unknown case split']])}${reset}</div><div class="mri-weight-list"></div>`);
      s.listen(()=>{const source=s.value('source'),items=source==='random'?['Encoder: random initialization','Decoder/head: random initialization','Check data split; no pretrained task knowledge.']:source==='imagenet'?['Encoder: learned ImageNet parameters/buffers','Decoder/head: new task-specific initialization','Same MRI architecture as the random arm; record the exact weight ID.']:['Encoder AND decoder: MRI-trained state','Patient exposure: must be investigated','Do not claim held-out MRI performance until checkpoint case provenance is known.'];
        s.q('.mri-weight-list').innerHTML=items.map(t=>`<div class="mri-weight">${t}</div>`).join('');s.out.textContent=source==='mri'?'This is not the notebook’s ImageNet-initialization experiment. Some intended validation cases may already have influenced the checkpoint.':source==='imagenet'?'Pretraining changes the starting encoder state. It does not supply a trained one-channel MRI decoder or guarantee better validation performance.':'This is the matched random-initialization control arm. Architecture and training settings still matter.';el.dataset.source=source;
      });
    }
  };
  document.querySelectorAll('[data-mri-lab]').forEach(el=>{const start=labs[el.dataset.mriLab];if(start)start(el);});
})();
