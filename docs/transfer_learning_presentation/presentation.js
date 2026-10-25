/* Interaction contract: fixed, deterministic toy data; native controls; semantic
   diagrams. Only user-started convolution scans and scalar updates animate.
   A finite timeline settles on navigation, interruption, or reduced motion. */
(() => {
  'use strict';
  const $ = (s, root=document) => root.querySelector(s);
  const $$ = (s, root=document) => [...root.querySelectorAll(s)];
  const notesBase = 'https://manyakaistha.github.io/DLGenAI-26T3/session-3/';
  const escape = v => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const btn = (text, action, cls='') => `<button type="button" data-action="${action}" class="${cls}">${text}</button>`;
  const jump = (text, id) => `<button type="button" data-jump="${id}">${text}</button>`;
  const reset = () => btn('Reset', 'reset');
  const panel = (content, cls='') => `<div class="panel ${cls}">${content}</div>`;
  const callout = text => `<div class="callout">${text}</div>`;
  const quiz = (question, options, correct, explanation) => `<div data-lab="quiz" data-correct="${correct}" data-explanation="${escape(explanation)}"><div class="question">${question}</div><div class="quiz-options" style="margin-top:18px">${options.map((text,i)=>btn(text,`answer-${i}`,'answer-button')).join('')}</div><div class="answer" role="status">Choose an answer, then explain your reasoning.</div>${reset()}</div>`;
  // Explanations have a deliberate reading order: takeaway, reasoning, then caveat.
  const lesson = (label, title, rows, foot='') => `<aside class="lesson-notes"><div class="notes-kicker">${label}</div><h3 class="notes-title">${title}</h3><dl class="notes-reasoning">${rows.map(([label,text])=>`<div><dt>${label}</dt><dd>${text}</dd></div>`).join('')}</dl>${foot?`<div class="notes-foot">${foot}</div>`:''}</aside>`;
  const slides = [
    {id:'start',chapter:'26 October 2026 · Milestone 3',title:'Transfer learning for segmentation',note:'',hero:true,body:`<div class="body two"><div class="hero-left"><div class="eyebrow">Live session 03 · 26 October 2026</div><h1>A head start<br>for <em>your U-Net.</em></h1><p>Borrow a pretrained encoder. Build a decoder for your task. Then test what that head start actually gives you.</p><div class="row" style="margin-top:24px">${jump('Let’s begin <span aria-hidden="true">↗</span>','task')}</div><p class="small muted" style="margin-top:20px">Explore the examples together. <br>Keep the study notes for the code.</p></div><figure class="cover-workbench"><figcaption><span class="tag">From an image to a pixel decision</span><span class="cover-caption">A schematic U-Net</span></figcaption><div class="cover-model"><div class="cover-bridge"><span>skip connections · spatial detail</span></div><div class="cover-unit"><div class="cover-sample">${Array.from({length:64},(_,i)=>`<i class="${[18,19,26,27,34,35,42,43].includes(i)?'sample-mark':''}"></i>`).join('')}</div><strong>Image</strong><span>What do we see?</span></div><span class="model-arrow" aria-hidden="true">→</span><div class="cover-unit"><div class="encoder-stack"><i></i><i></i><i></i><span>features</span></div><strong>Encoder → decoder</strong><span>Find patterns. Restore detail.</span></div><span class="model-arrow" aria-hidden="true">→</span><div class="cover-unit"><div class="cover-sample mask">${Array.from({length:64},(_,i)=>`<i class="${[18,19,26,27,34,35,42,43].includes(i)?'sample-mark':''}"></i>`).join('')}</div><strong>Mask</strong><span>Where is it?</span></div></div><div class="cover-key"><span><i></i> Pretrained encoder</span><span><i></i> New decoder + pixel head</span></div><p class="tiny muted">Synthetic pattern · binary example</p></figure></div><nav class="session-route" aria-label="Session roadmap">${[['01','The model','Shapes & skip connections','task'],['02','The head start','Architecture & weights','pretraining'],['03','The fine-tuning','Gradients & model state','freeze'],['04','The evidence','Losses & fair comparisons','logits']].map(([n,t,d,id])=>`<button data-jump="${id}"><span>${n}</span><strong>${t}</strong><small>${d}</small><i aria-hidden="true">↗</i></button>`).join('')}</nav>`},
    {id:'task',chapter:'01 / Encoder & decoder',title:'Classification or segmentation?',lede:"Switch the output. What information does each prediction retain?",note:'encoder.html',body:`<div class="body two">${panel(`<div data-lab="task"><div class="controls">${btn('Classification','classify')}${btn('Segmentation','segment','primary')}</div><div class="viz"></div></div>`)}${lesson("The prediction","A label says what.<br>A mask says where.",[["Encoder","Find edges, textures and increasingly complex patterns."],["Decoder","Combine features at higher spatial resolutions."],["Pixel head","Produce one logit per pixel for each binary target."]],"Today’s examples use one binary target. The industrial task needs <strong>four independent output channels.</strong>")}</div>`},
    {id:'shapes',chapter:'01 / Encoder & decoder',title:'Inside the encoder',lede:"Track height, width and channel count through a ResNet18 encoder.",note:'encoder.html',body:`<div class="body wide-left">${panel(`<div data-lab="shapes"><div class="controls"><label>Input <select name="size"><option value="256,256">256 × 256</option><option value="128,128">128 × 128</option><option value="128,800">128 × 800 · industrial aspect ratio</option></select></label>${reset()}</div><div class="viz"></div></div>`)}${lesson("Read the tensor","Less spatial detail.<br>More feature channels.",[["H × W ↓","Downsampling reduces the feature map’s height and width."],["Channels ↑","More channels let the network represent more kinds of patterns."],["Keep them distinct","Feature channels describe patterns. Output channels describe target classes."]],"Choose the <strong>128 × 800</strong> input to follow the industrial aspect ratio.")}</div>`},
    {id:'convolution',chapter:'01 / Encoder & decoder',title:'How a convolution finds a pattern',lede:"Advance the 3 × 3 filter and inspect its dot product.",note:'encoder.html',body:`<div class="body wide-left">${panel(`<div data-lab="conv"><div class="controls">${btn('Next position','step','primary')}${btn('Play one scan','play')}${reset()}</div><div class="viz"></div></div>`)}${lesson("Follow the filter","Shared weights.<br>Local responses.",[["01 / Select","Take one 3 × 3 image patch."],["02 / Multiply","Pair each pixel with the corresponding filter weight."],["03 / Sum","Add the nine products. A stronger match gives a larger response for this filter."]],"This vertical-change filter is <strong>hand-set</strong>. A real CNN learns the filter weights.")}</div>`},
    {id:'join',chapter:'01 / Encoder & decoder',title:'Building a decoder, step by step',lede:"Resize → concatenate → convolve. Watch the tensor at each operation.",note:'encoder.html',body:`<div class="body wide-left">${panel(`<div data-lab="join"><div class="controls">${btn('Next operation','step','primary')}${reset()}<span class="small muted">Input: 256 × 256</span></div><div class="viz"></div></div>`)}${lesson("Three decoder operations","Match space.<br>Then mix channels.",[["01 / Resize","Match the skip’s exact <code>H × W</code>."],["02 / Concatenate","Keep both channel sets: <strong class=\"notes-number\">512 + 256 = 768</strong>"],["03 / Convolve","Learn to mix 768 input channels into 256 output features."]],"Upsampling aligns the spatial sizes. It does not add back information that was lost.")}</div>`},
    {id:'skip',chapter:'01 / Encoder & decoder',title:'Addition and concatenation',lede:'Same word, different operation. Follow what happens to the channels.',note:'encoder.html',body:`<div class="comparison-spread"><article><div class="comparison-label">ResNet · residual block</div><h3>Add the values.</h3><div class="tensor-equation"><code>[B,64,H,W]</code><span>+</span><code>[B,64,H,W]</code><strong>= [B,64,H,W]</strong></div><p>Shapes must match at the addition. A projection can align the residual branch.</p><div class="comparison-result"><span>Channel count</span><strong>64 → 64</strong></div></article><article><div class="comparison-label">U-Net · decoder join</div><h3>Keep both feature sets.</h3><div class="tensor-equation"><code>[B,512,H,W]</code><span>concat</span><code>[B,256,H,W]</code><strong>= [B,768,H,W]</strong></div><p>Batch and spatial sizes match. Channels add; a new convolution learns how to combine them.</p><div class="comparison-result"><span>Channel count</span><strong>512 + 256 → 768</strong></div></article></div><p class="slide-takeaway">The high-resolution skip preserves access to earlier spatial detail. Upsampling alone cannot recover information that was lost.</p>`},
    {id:'shape-check',chapter:'01 / Checkpoint',title:'What shape comes next?',lede:"Use the tensors on the right. Predict the decoder’s concatenated shape.",note:'encoder.html',body:`<div class="body two">${panel(quiz('After resizing deep → (8, 50), then concatenating along channels, what is the shape?', ['[4, 768, 8, 50]','[4, 512, 12, 75]','[4, 256, 8, 50]'],0,'Yes. Batch stays 4; channels add to 768; spatial size stays 8 × 50.'))}${lesson("Work from these tensors","Spatial sizes match.<br>Channels add.",[["Deep feature","<code>[4, 512, 4, 25]</code>"],["Skip feature","<code>[4, 256, 8, 50]</code>"],["Your operation","Resize deep to <code>8 × 50</code>, then concatenate along channels."]],`These feature sizes follow a <strong>128 × 800</strong> input. Keep the industrial aspect ratio.<div class="notes-links">${jump("Optional: MRI image & mask →","mri-data")}</div>`)}</div>`},
    {id:'mri-data',chapter:'Optional / MRI application',title:'An MRI segmentation example',lede:'The image changes; the question does not: which pixels belong to the mask?',note:'dataset.html',optional:true,body:`<div class="body wide-left">${panel(`<img class="mri-image" src="__MRI_IMAGE__" alt="One supplied MRI slice shown as pre-contrast, FLAIR, post-contrast, and its binary abnormality mask."><p class="tiny muted" style="margin-top:12px">Selected teaching image: Mateusz Buda / TCIA / TCGA-LGG. CC BY-NC-SA 4.0. The mask marks FLAIR abnormality; this is not a tumor-grade classifier or a diagnostic system.</p>`)}${lesson("Prevent leakage","One patient.<br>One partition.",[["Image channels","Pre-contrast, FLAIR and post-contrast are MRI sequences."],["Split unit","Keep every slice from one patient wholly in train or wholly in validation."],["Why it matters","Related slices in both partitions would leak patient information."]],`<div class="notes-links">${jump("Continue: pretraining →","pretraining")}</div>`)}</div>`},
    {id:'pretraining',chapter:'02 / Pretraining',title:'What do we reuse from a pretrained model?',lede:"Change the encoder’s starting weights; distinguish the choices around it.",note:'transfer.html',body:`<div class="body wide-left">${panel(`<div data-lab="pretraining"><div class="controls">${btn('Random encoder','random')}${btn('ImageNet encoder','imagenet','primary')}</div><div class="viz"></div></div>`)}${lesson("Separate three decisions","The network, its start<br>and its training policy.",[["Architecture","Which operations are connected?"],["Initialization","Which values do the parameters start from?"],["Training policy","Which parameters may update, and at what learning rates?"]],"ImageNet weights change the encoder’s <strong>starting values</strong>. The new decoder and pixel head still need training.")}</div>`},
    {id:'features',chapter:'02 / Pretraining',title:'Why can pretraining help?',lede:"Explore what might transfer, and what a new task still needs to learn.",note:'transfer.html',body:`<div class="body two">${panel(`<div data-lab="features"><div class="feature-options">${btn('Earlier features','early','primary')}${btn('Later features','late')}${btn('Domain shift','shift')}</div><div class="viz"></div></div>`)}${lesson("The transfer hypothesis","Useful patterns.<br>A task still to learn.",[["What may transfer","Features useful for natural-image classification may help a new image task."],["What must be learned","The new decoder and head must learn the target masks."],["What to verify","A faster start and better final validation performance are separate questions."]],"Pretraining is a starting point. Evaluate it under the same preprocessing and validation protocol.")}</div>`},
    {id:'initialization',chapter:'02 / Controlled comparison',title:'Comparing random and pretrained weights',lede:"Compare the encoder starts while keeping the decoder start identical.",note:'transfer.html',body:`<div class="body wide-left">${panel(`<div data-lab="initialization"><div class="controls"><label><input type="checkbox" name="matched" checked> Copy the same initial decoder/head state into both arms</label></div><div class="viz"></div></div>`)}${lesson("A controlled starting point","One saved decoder.<br>Two encoder starts.",[["Save","Create and save one initial decoder/head state."],["Reuse","Load that same state into both training arms."],["Record","Log the encoder weight source; keep both arms fully trainable for this comparison."]],`Construction order can consume random numbers differently. Compare freezing policies separately.<div class="notes-links">${jump("Optional: MRI channels →","mri-inputs")}</div>`)}</div>`},
    {id:'mri-inputs',chapter:'Optional / MRI application',title:'Three MRI channels, three different sequences',lede:'The three channels are sequences, not RGB colours.',note:'inputs.html',optional:true,body:`<div class="body two">${panel(`<div data-lab="channels"><div class="controls"><label>Input <select name="mode"><option value="sequences">Three supplied sequences</option><option value="repeat">FLAIR repeated three times</option></select></label>${reset()}</div><div class="viz"></div></div>`)}${lesson("Respect channel meaning","Sequences are not<br>red, green and blue.",[["Scale","Convert uint8 values with <code>raw / 255</code>."],["Normalize","Apply the pretrained channel-specific mean and standard deviation."],["Interpret","Repeated raw values need not stay equal after normalization. Avoid RGB hue jitter."]],`Missing sequences in this dataset may be replaced by FLAIR.<div class="notes-links">${jump("Continue: fine-tuning →","freeze")}</div>`)}</div>`},
    {id:'freeze',chapter:'03 / Freezing & fine-tuning',title:'What does “freeze the encoder” mean?',lede:"Change the switches. Predict the parameter and BatchNorm buffer separately.",note:'finetuning.html',body:`<div class="body two">${panel(`<div data-lab="freeze"><div class="switches"><label>Compute parameter gradient<input name="grad" type="checkbox" checked></label><label>Parameter is in optimizer<input name="optimizer" type="checkbox" checked></label><label>Encoder in train mode<input name="train" type="checkbox" checked></label></div><div class="controls" style="margin-top:20px">${btn('Run one step','step','primary')}${reset()}</div><div class="viz"></div></div>`)}${lesson("Predict before the step","Parameters and buffers<br>are different state.",[["Parameter start","<code>w = 1.00 · g = 0.40 · η = 0.10</code>"],["Buffer start","BN running mean = 0. Batch mean = 2; momentum = 0.1."],["Step rule","Here, the weight updates only with a newly computed gradient and optimizer membership."]],"The example starts with cleared gradients. Clear old gradients when freezing parameters in real code.")}</div>`},
    {id:'batchnorm',chapter:'03 / BatchNorm',title:'BatchNorm can still change',lede:"Frozen parameters do not automatically freeze running statistics.",note:'finetuning.html',body:`<div class="body two">${panel(`<div data-lab="bn"><div class="controls"><label>Encoder mode <select name="mode"><option value="train">train()</option><option value="eval">eval()</option></select></label>${btn('Observe one batch','step','primary')}${reset()}</div><div class="viz"></div></div>`)}${lesson("Order matters","Set the encoder mode<br>after model.train().",[["Each training phase","<pre class=\"notes-code\">model.train()\n<strong>model.encoder.eval()</strong>\n# then the training batches</pre>"],["Why this order","<code>model.train()</code> recursively changes child modes. Reapply encoder eval mode when keeping its BN statistics fixed."]],"<code>eval()</code> changes BatchNorm and dropout behavior. It <strong>does not disable gradient tracking.</strong>")}</div>`},
    {id:'unfreeze',chapter:'03 / Optimizer membership',title:'Why isn’t the encoder updating?',lede:"Inspect what this optimizer actually owns.",note:'finetuning.html',body:`<div class="body two">${panel(`<div data-lab="unfreeze"><pre class="code">encoder.requires_grad_(True)
optimizer = AdamW(
    <span class="accent">decoder.parameters()</span>, lr=1e-3
)
loss.backward()
optimizer.step()</pre><div class="controls" style="margin-top:20px">${btn('Try the step','try','primary')}${btn('Rebuild with both groups','fix')}${reset()}</div><div class="viz"></div></div>`)}${lesson("Two checks before updating","A gradient is necessary.<br>Membership matters too.",[["01 / Gradient","Does the encoder receive a fresh gradient?"],["02 / Optimizer","Does the optimizer contain the encoder’s parameters?"],["When unfreezing","Rebuild or update the parameter groups deliberately."]],"Rebuilding an optimizer resets its internal state unless that state is restored.")}</div>`},
    {id:'rates',chapter:'03 / Learning rates',title:'Different learning rates for encoder and decoder',lede:"Hold the gradient fixed. Compare the size of each update.",note:'finetuning.html',body:`<div class="body two">${panel(`<div data-lab="rates"><div class="stack"><label class="small">Encoder learning rate <select name="encoder"><option value="0.0001">0.0001</option><option value="0.001">0.001</option><option value="0.01">0.01</option></select></label><label class="small">Decoder learning rate <select name="decoder"><option value="0.001">0.001</option><option value="0.0001">0.0001</option><option value="0.01">0.01</option></select></label></div><div class="controls" style="margin-top:22px">${btn('Compare one update','step','primary')}${reset()}</div><div class="viz"></div></div>`)}${lesson("Read the update","<span class=\"notes-equation\">w′ = w − ηg</span>",[["Step size","For the same gradient, a larger learning rate gives a larger update."],["This example","Both scalar gradients are <code>g = 0.4</code>; updates use plain SGD."],["Real AdamW","Updates also depend on optimizer state and weight decay."]],"If the experiment tests initialization alone, keep the learning-rate policy the same in both arms.")}</div>`},
    {id:'logits',chapter:'04 / Losses & evidence',title:'Logits, probabilities and pixel decisions',lede:"Move the logit; trace the probability, loss and hard decision.",note:'experiments.html',body:`<div class="body two">${panel(`<div data-lab="logits"><div class="controls"><label>Logit z <input type="range" name="z" min="-5" max="5" step="0.1" value="0"></label><label>Target <select name="target"><option value="1">1 · foreground</option><option value="0">0 · background</option></select></label></div><div class="viz"></div>${reset()}</div>`)}${lesson("Two uses of the output","Train on logits.<br>Decide on probabilities.",[["Training","Pass raw logits to <code>BCEWithLogitsLoss</code>. Its stable calculation includes sigmoid internally."],["Evaluation","Convert with <code>p = sigmoid(z)</code>, then apply a threshold to make a hard mask."]],"Do not apply sigmoid before BCEWithLogitsLoss. Hard thresholding is not part of the differentiable training loss.")}</div>`},
    {id:'soft-dice',chapter:'04 / Training objective',title:'BCE and Dice loss',lede:"Move foreground and background confidence independently.",note:'experiments.html',body:`<div class="body wide-left">${panel(`<div data-lab="soft"><div class="controls"><label>p on target cells <input name="positive" type="range" min="0.05" max="0.95" step="0.05" value="0.7"></label><label>p on background <input name="negative" type="range" min="0.05" max="0.95" step="0.05" value="0.2"></label></div><div class="viz"></div>${reset()}</div>`)}${lesson("Complementary feedback","Pixel confidence.<br>Probability overlap.",[["BCE","Penalizes each wrong probability."],["Soft Dice","<span class=\"notes-math\">(2Σpy + ε) / (Σp + Σy + ε)</span>Measures overlap using probabilities."],["Combined objective","<strong>mean BCE + (1 − soft Dice)</strong>"]],"Here ε = 10⁻⁶, one sample. Average per-sample Dice in the notes. For empty targets, soft Dice can have a weak gradient; BCE still penalizes false positives.")}</div>`},
    {id:'threshold',chapter:'04 / Hard-mask evaluation',title:'Choosing a probability threshold',lede:"Keep probabilities fixed. Change only the decision threshold.",note:'experiments.html',body:`<div class="body wide-left">${panel(`<div data-lab="threshold"><div class="controls"><label>Threshold <input name="threshold" type="range" min="0" max="1" step="0.05" value="0.5"></label><label>Target <select name="target"><option value="square">Central square</option><option value="empty">Empty mask</option></select></label>${reset()}</div><div class="viz"></div></div>`)}${lesson("Read the errors","Count the mistakes.<br>Measure the overlap.",[["Hard Dice","<span class=\"notes-math\">2TP / (2TP + FP + FN)</span>"],["Error map","<div class=\"notes-legend\"><span><i class=\"tp\"></i>TP · true positive</span><span><i class=\"fp\"></i>FP · false positive</span><span><i class=\"fn\"></i>FN · false negative</span><span><i class=\"tn\"></i>TN · true negative</span></div>"]],"Both empty → 1; exactly one empty → 0. Use <strong>0.5</strong> for the initialization comparison. Tune thresholds only under a declared validation protocol.")}</div>`},
    {id:'audit',chapter:'04 / Experiment design',title:'Is this a fair comparison?',lede:"Match the controls until initialization is the only intended difference.",note:'experiments.html',body:`<div class="body wide-left">${panel(`<div data-lab="audit"><div class="viz"></div><div class="controls" style="margin-top:14px">${btn('Match all settings','match','primary')}${reset()}</div></div>`)}${lesson("Design the comparison","One intended difference:<br>encoder initialization.",[["Match","Split, training budget, loss, encoder policy and threshold."],["Also hold fixed","Input pipeline, decoder start, optimizer, sample-order policy, seed policy and evaluation examples."],["Then repeat","Use other seeds to assess how stable the difference is."]],"Match the settings before attributing a score change to pretraining.")}</div>`},
    {id:'curves',chapter:'04 / Interpreting evidence',title:'Does a faster start give a better result?',lede:"Illustrative curves, not measured results. Read training and validation separately.",note:'experiments.html',body:`<div class="body wide-left">${panel(`<div data-lab="curves"><div class="controls">${btn('Training loss','train','primary')}${btn('Validation Dice','validation')}${btn('Reveal interpretation','reveal')}</div><div class="viz"></div><div class="answer" role="status"></div></div>`)}${lesson("Two different questions","Faster fitting ≠<br>better generalization.",[["Training curve","At the same budget, which arm fits the training examples sooner?"],["Validation curve","On held-out examples, which arm performs better?"]],"Inspect fixed predictions, misses, false positives and subgroup performance. Lower training loss or one clean-looking mask does not prove generalization.")}</div>`},
    {id:'review',chapter:'04 / Class discussion',title:'What would you test next?',lede:"Choose a comparison whose outcome you could explain.",note:'experiments.html',body:`<div class="body two">${panel(quiz('Which follow-up comparison is easiest to interpret?', ['Two ImageNet runs: freeze the encoder in one; train it in the other. Match everything else.','Frozen ImageNet for 10 epochs vs random initialization for 2 epochs.','Change backbone, loss, input size and threshold together.'],0,'Yes. Keep ImageNet initialization in both runs; change only the freezing policy. Specify BatchNorm mode and optimizer groups.'))}${lesson("Choose what to revisit","Follow the uncertainty.",[["A failed mask",`Check the errors and threshold.<div class="notes-links">${jump("Revisit Dice →","threshold")}</div>`],["A frozen encoder",`Check gradients, membership and mode.<div class="notes-links">${jump("Revisit switches →","freeze")}</div>`],["A real-data extension",`Use the optional MRI notebook.<div class="notes-links">${jump("Optional notebook →","mri-notebook")}</div>`]],"")}</div>`},
    {id:'mri-notebook',chapter:'Optional / After-class extension',title:'Try the MRI notebook after class',lede:'When you have time, take the same questions to real image–mask pairs.',note:'notebook.html',optional:true,body:`<div class="body two"><ol class="notebook-path"><li><span>Validate the pairs</span><p>Check TIFF image–mask pairs and patient identities.</p></li><li><span>Split patients first</span><p>Select slices only after assigning patients to partitions.</p></li><li><span>Match the training arms</span><p>Keep controls fixed for random and ImageNet starts.</p></li><li><span>Inspect the evidence</span><p>Compare losses, predictions and held-out metrics.</p></li></ol>${lesson("Choose your next step","Start with shapes.<br>Then try real pairs.",[["Classroom notebook","Dummy tensors for feature shapes, decoder joins and gradient flags."],["Full MRI notebook","Download real data separately and train matched random/ImageNet arms."]],`The small default MRI subset is a teaching run, not a cohort benchmark.<div class="notes-links"><a href="${notesBase}notebook.html" target="_blank" rel="noopener">Open notebook guide ↗</a>${jump("Return to the takeaway →","finish")}</div><span class="notes-source">Dataset: <a href="https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation" target="_blank" rel="noopener">Mateusz Buda · LGG MRI</a></span>`)}</div>`},
    {id:'finish',chapter:'26 October / Take it back to your U-Net',title:'Before you train your next model',lede:'Four questions to carry into your next experiment.',note:'',body:`<div class="closing-layout"><ol class="closing-checklist"><li><span>The shapes</span><p>Can you track spatial size and channel count through the model?</p></li><li><span>The starting point</span><p>Have you separated architecture, initialization and training policy?</p></li><li><span>The update</span><p>Have you checked gradients, optimizer membership and module mode?</p></li><li><span>The evidence</span><p>Is the change supported by matched, held-out comparisons?</p></li></ol><aside class="closing-question"><div class="notes-kicker">Exit question</div><blockquote>“My pretrained model<br>got a better score.”</blockquote><p>What would you check first to decide whether the comparison was fair?</p><div class="notes-links">${jump('Audit the experiment →','audit')}<a href="${notesBase}" target="_blank" rel="noopener">Detailed revision guides ↗</a></div><small>Use the notes for full explanations and runnable code.</small></aside></div>`},
  ];

  $('#slides').innerHTML=slides.map((s,i)=>`<section id="${s.id}" class="slide ${s.hero?'hero':''}" aria-labelledby="title-${i}" ${i?'hidden inert':''}>${s.hero?'':`<div class="eyebrow">${escape(s.chapter)}</div><h2 id="title-${i}" tabindex="-1">${escape(s.title)}</h2>${s.lede?`<p class="lede">${s.lede}</p>`:''}`} ${s.body}</section>`).join('');
  $('#start h1').id='title-0'; $('#start h1').tabIndex=-1;
  const chapters=[['01','Model','task'],['02','Weights','pretraining'],['03','Fine-tune','freeze'],['04','Evidence','logits']];
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
  let current=0, includeMRI=false, motionOff=matchMedia('(prefers-reduced-motion: reduce)').matches;
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
  const route=()=>slides.map((s,i)=>!s.optional||includeMRI?i:null).filter(i=>i!==null);
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
    $('#counter').textContent=position<0?`Optional · ${slides.filter(s=>s.optional).findIndex(s=>s.id===slides[n].id)+1}/3`:`${String(position+1).padStart(2,'0')} / ${path.length}`;
    $('#previous').disabled=n===0;$('#next').disabled=n===slides.length-1;
    const next=path.find(i=>i>n),prev=path.filter(i=>i<n).at(-1);
    $('#next').title=next===undefined?'End of session':`Next: ${slides[next].title}`;
    $('#previous').title=prev===undefined?'Start of session':`Previous: ${slides[prev].title}`;
    $('#notes').href=notesBase+slides[n].note;
    $('#progress').style.width=`${(position<0?(n+1)/slides.length:(position+1)/path.length)*100}%`;
    try{history.replaceState(null,'',`#${slides[n].id}`);}catch{/* Embedded/offline viewers may restrict URL changes. */}
    if(focus)$(`#title-${n}`).focus({preventScroll:true});
    if(innerWidth<=700)scrollTo({top:0,behavior:'instant'});
    $$('#index-list button').forEach(b=>b.setAttribute('aria-current',b.dataset.jump===slides[n].id?'page':'false'));
  }
  function move(direction){let n=current+direction;while(n>=0&&n<slides.length&&!includeMRI&&slides[n].optional)n+=direction;go(n);}
  $('#previous').onclick=()=>move(-1);$('#next').onclick=()=>move(1);
  const dialog=$('#index-dialog');
  $('#contents').onclick=()=>{settle();$('#slide-search').value='';filterIndex();dialog.showModal();$('#slide-search').focus();};$('#close-index').onclick=()=>dialog.close();
  $('#include-mri').onchange=e=>{includeMRI=e.target.checked;go(current,false);};
  const indexGroups=[['Opening',[slides[0]]],...chapters.map(([n,label])=>[`${n} / ${label}`,slides.filter(s=>s.chapter.startsWith(n))]),['Optional / MRI',slides.filter(s=>s.optional)]];
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

  const grid=(values,cols,classes=()=>'',label='')=>`<div class="pixel-grid" role="img" aria-label="${escape(label||values.join(', '))}" style="grid-template-columns:repeat(${cols},1fr)">${values.map((v,i)=>`<span class="pixel ${classes(v,i)}" aria-hidden="true">${escape(v)}</span>`).join('')}</div>`;
  const setPressed=(root,active)=>$$('[data-action]',root).forEach(b=>{if(b.dataset.action!=='reset'){const selected=b.dataset.action===active;b.setAttribute('aria-pressed',String(selected));b.classList.toggle('primary',selected);}});
  const labs={};
  function mount(name,setup){$$(`[data-lab="${name}"]`).forEach(root=>{
    const api=setup(root),viz=$('.viz',root);
    $$('input[type=range]',root).forEach((input,i)=>{
      input.id=`range-${name}-${i}`;
      input.insertAdjacentHTML('afterend',`<output class="range-value" for="${input.id}" aria-live="off"></output>`);
    });
    function update(){api.render(viz);$$('input[type=range]',root).forEach(input=>{const value=Number(input.value).toFixed(input.name==='z'?1:2);input.nextElementSibling.value=value;input.setAttribute('aria-valuetext',value);});}

    root.addEventListener('click',e=>{const b=e.target.closest('[data-action]');if(!b)return;settle();api.action?.(b.dataset.action,update);if(!api.ownsRender){const focused=document.activeElement===b;update();if(focused&&!b.isConnected){const replacement=$$('[data-action]',root).find(x=>x.dataset.action===b.dataset.action);replacement?.focus({preventScroll:true});}}});
    root.addEventListener('input',e=>{if(e.target.matches('input,select')){settle();api.input?.(e.target);update();}});
    root.addEventListener('change',e=>{if(e.target.matches('select')){settle();api.input?.(e.target);update();}});
    update();labs[name]={root,update,...api};
  });}
  mount('task',root=>{let segmentation=true;return{action:a=>{segmentation=a==='segment';},render:v=>{setPressed(root,segmentation?'segment':'classify');v.innerHTML=`<div class="flow"><div><div class="grid-label">Toy input pattern</div>${grid(Array.from({length:36},(_,i)=>[8,9,14,15,20,21,26,27].includes(i)?'■':'',),6,v=>v?'on':'','A toy 6 by 6 image with a foreground region')}</div><span class="arrow">→</span><div><div class="grid-label">${segmentation?'36 pixel decisions':'One image decision'}</div>${segmentation?grid(Array.from({length:36},(_,i)=>[8,9,14,15,20,21,26,27].includes(i)?1:0),6,v=>v?'on':'','Segmentation output: one binary label at each of 36 positions'):'<div class="node active" style="width:200px;height:190px;display:grid;place-items:center"><strong>Foreground<br>present</strong></div>'}</div></div><p class="note-line">${segmentation?'The output preserves where the foreground is.':'An image label says what is present, without locating every pixel.'}</p>`;}};});
  mount('shapes',root=>{let selected=4;const names=['Stem','Layer 1','Layer 2','Layer 3','Layer 4'],channels=[64,64,128,256,512],divisors=[2,4,8,16,32];return{action:a=>{if(a==='reset'){$('[name=size]',root).value='256,256';selected=4;}else if(a.startsWith('stage-'))selected=+a.split('-')[1];},render:v=>{const [h,w]=$('[name=size]',root).value.split(',').map(Number);v.innerHTML=`<div class="shape-steps">${names.map((n,i)=>`<button class="shape-step" data-action="stage-${i}" aria-pressed="${selected===i}"><div class="shape-box" style="height:${148-i*23}px">${channels[i]} ch</div><strong>${n}</strong><span>${h/divisors[i]} × ${w/divisors[i]}</span></button>`).join('')}</div><div class="rule"></div><div class="row spread"><span class="small">${names[selected]} output</span><strong class="mono" style="font-size:27px">[B, ${channels[selected]}, ${h/divisors[selected]}, ${w/divisors[selected]}]</strong></div><p class="note-line">Stem shown before max-pool. Layer 1 follows that pool. Values assume the listed inputs and the standard ResNet18 stride layout.</p>`;}};});
  const image=[0,0,1,1,1, 0,0,1,1,1, 0,0,1,1,1, 0,0,1,1,1, 0,0,1,1,1],kernel=[-1,0,1,-1,0,1,-1,0,1];
  const convolutionAt=n=>{const row=Math.floor(n/3),col=n%3;return kernel.reduce((sum,k,i)=>sum+k*image[(row+Math.floor(i/3))*5+col+i%3],0);};
  mount('conv',root=>{let position=0;const outputs=Array.from({length:9},(_,i)=>convolutionAt(i));return{ownsRender:true,action:(a,update)=>{if(a==='reset')position=0;else if(a==='step')position=(position+1)%9;else if(a==='play'){position=0;update();animate(3600,u=>{const next=Math.min(8,Math.floor(u*9));if(next!==position||u===1){position=next;update();}},()=>{$('#announcer').textContent='Convolution scan complete: output rows are 3, 3, 0.';});return;}update();},render:v=>{const row=Math.floor(position/3),col=position%3;const terms=kernel.map((k,i)=>`${image[(row+Math.floor(i/3))*5+col+i%3]}×${k<0?'(−1)':k}`);v.innerHTML=`<div class="conv-layout"><div><div class="grid-label">Input · 5 × 5</div>${grid(image,5,(_,i)=>Math.floor(i/5)>=row&&Math.floor(i/5)<row+3&&i%5>=col&&i%5<col+3?'hot':'','Input image: each row is 0,0,1,1,1; coral outline marks patch row '+(row+1)+', column '+(col+1))}</div><div><div class="grid-label">Filter · 3 × 3</div>${grid(kernel,3,()=> 'filter','Filter: each row is negative one, zero, one')}</div><div><div class="grid-label">Output · 3 × 3</div>${grid(outputs.map((x,i)=>i<=position?x:'·'),3,(_,i)=>i===position?'hot':'','Cross-correlation responses: each complete row is 3,3,0')}</div></div><p class="small mono" style="margin-top:15px">${terms.slice(0,3).join(' + ')}<br>+ ${terms.slice(3,6).join(' + ')}<br>+ ${terms.slice(6).join(' + ')} = <strong>${outputs[position]}</strong></p><p class="tiny muted" style="margin-top:8px">Patch row ${row+1}, column ${col+1}. Sum nine element-wise products.</p>`;}};});
  mount('join',root=>{let step=0;return{action:a=>{step=a==='reset'?0:Math.min(3,step+1);},render:v=>{const titles=['1 · Inspect the mismatch','2 · Upsample to the skip size','3 · Concatenate channels','4 · Mix features with a convolution'];v.innerHTML=`<div class="step-indicator" aria-label="Step ${step+1} of 4">${titles.map((_,i)=>`<span class="${i<=step?'done':''}"></span>`).join('')}</div><h3>${titles[step]}</h3><div class="join-visual">${step<2?`<div><div class="join-block" style="height:${step===0?68:135}px">512</div><div class="join-label">${step===0?'8 × 8':'16 × 16'}</div></div><span class="join-operation">${step===0?'≠':'concat'}</span><div><div class="join-block skip">256</div><div class="join-label">16 × 16 · skip</div></div>`:`<div><div class="join-block joined">768</div><div class="join-label">16 × 16</div></div>${step===3?'<span class="arrow">→</span><div><div class="join-block">256</div><div class="join-label">16 × 16</div></div>':''}`}</div><pre class="code" style="font-size:17px;margin-top:8px">${['deep: [B,512,8,8]   skip: [B,256,16,16]','up = interpolate(deep, size=skip.shape[-2:])','joined = cat([up, skip], dim=1)','# Conv2d(768, 256, kernel_size=3, padding=1)\nout = block(joined)'][step]}</pre><p class="note-line">${['Cannot concatenate while H and W differ.','Upsampling changes H and W, not the number of channels.','Concatenation preserves both channel sets; it does not add feature values.','New decoder weights learn to combine coarse and detailed information.'][step]}</p>`;$('[data-action=step]',root).disabled=step===3;}};});
  mount('pretraining',root=>{let pretrained=true;return{action:a=>{pretrained=a==='imagenet';},render:v=>{setPressed(root,pretrained?'imagenet':'random');v.innerHTML=`<div class="flow" style="margin:16px 0 28px"><div class="node active"><strong>ResNet18<br>encoder</strong><span>${pretrained?'ImageNet-trained weights':'Random initial weights'}</span></div><span class="arrow">→</span><div class="node"><strong>New<br>decoder</strong><span>Random initial weights</span></div><span class="arrow">→</span><div class="node"><strong>New pixel<br>head</strong><span>Binary logits</span></div></div><div class="callout">${pretrained?'Keep the feature extractor. Discard the ImageNet classifier.':'Use the same feature-extractor architecture with no ImageNet weights.'}</div><p class="small muted" style="margin-top:20px">Only the encoder starts with ImageNet weights. The decoder and pixel head start from scratch.</p>`;}};});
  mount('features',root=>{let tab='early';const content={early:['Local patterns','Edges, orientation and texture can be useful across many image domains.','These are common intuitions, not a guarantee that a specific feature is present or useful.'],late:['Task-dependent combinations','Deeper features combine information over larger regions and may be more specialized to the pretraining task.','A model trained on natural-image classes has not automatically learned abnormality masks.'],shift:['The input distribution changes','MRI sequences, acquisition settings, contrast and target shapes differ from natural photographs.','Test transfer under the same preprocessing and validation protocol; inspect where it fails.']};return{action:a=>tab=a,render:v=>{setPressed(root,tab);const c=content[tab];v.innerHTML=`<div class="tag">Schematic intuition</div><h3 style="font-size:32px;margin:15px 0">${c[0]}</h3><p>${c[1]}</p><div class="rule"></div><p class="small muted">${c[2]}</p>`;}};});
  mount('initialization',root=>({render:v=>{const same=$('[name=matched]',root).checked;v.innerHTML=`<table class="check-table"><thead><tr><th>Component</th><th>Random arm</th><th>Pretrained arm</th></tr></thead><tbody><tr><td>Encoder</td><td>Random</td><td>ImageNet</td></tr><tr><td>Decoder/head</td><td>Saved state A</td><td class="${same?'ok':'warning'}">${same?'Same state A':'Different state B'}</td></tr><tr><td>Training policy</td><td>All trainable</td><td>All trainable</td></tr></tbody></table><div class="status ${same?'ok':'warning'}" role="status">${same?'One intended difference: encoder starting weights. Keep the other controls matched too.':'Two initialization differences: encoder AND decoder. A score change cannot be assigned to the encoder alone.'}</div>`;}}));
  mount('channels',root=>({action:a=>{if(a==='reset')$('[name=mode]',root).value='sequences';},render:v=>{const repeat=$('[name=mode]',root).value==='repeat',raw=repeat?[120,120,120]:[80,120,160],means=[.485,.456,.406],stds=[.229,.224,.225];v.innerHTML=`<div class="tag">Synthetic single-pixel values, not measured MRI intensities</div><table class="check-table"><thead><tr><th>Input channel</th><th>Raw</th><th>After normalization</th></tr></thead><tbody>${raw.map((x,i)=>`<tr><td>${repeat?'Repeated FLAIR': ['Pre-contrast','FLAIR','Post-contrast'][i]}</td><td>${x}</td><td class="mono">${((x/255-means[i])/stds[i]).toFixed(3)}</td></tr>`).join('')}</tbody></table><p class="small mono" style="margin-top:18px">(raw / 255 − mean[c]) / std[c]</p><p class="note-line">Mean = [0.485, 0.456, 0.406]<br>Std = [0.229, 0.224, 0.225]</p>`;}}));
  mount('freeze',root=>{let result=null;return{action:a=>{if(a==='reset'){$$('input',root).forEach(x=>x.checked=true);result=null;}else result={weight:$('[name=grad]',root).checked&&$('[name=optimizer]',root).checked ? .96 : 1,mean:$('[name=train]',root).checked ? .2 : 0};},input:()=>result=null,render:v=>{v.innerHTML=`<div class="metric-grid"><div class="metric"><strong>${result?result.weight.toFixed(2):'1.00'}</strong><span>Parameter w</span></div><div class="metric"><strong>${result?result.mean.toFixed(2):'0.00'}</strong><span>BN running mean</span></div><div class="metric"><strong>${result?'1':'0'}</strong><span>Steps taken</span></div></div><div class="status" role="status">${result?`Weight ${result.weight===1?'stays fixed':'updates: 1 − 0.1 × 0.4 = 0.96'}. BN mean ${result.mean===0?'stays fixed in eval mode':'updates: 0.9 × 0 + 0.1 × 2 = 0.20'}.`:'Choose switches, then run one step. Each click recomputes from the stated starting values.'}</div>`;}};});
  mount('bn',root=>{let mean=0,count=0;return{ownsRender:true,action:(a,update)=>{if(a==='reset'){mean=0;count=0;$('[name=mode]',root).value='train';update();return;}const from=mean,to=$('[name=mode]',root).value==='train'?.9*mean+.2:mean;count++;animate(650,u=>{mean=from+(to-from)*(1-Math.pow(1-u,3));update();});},render:v=>{if(!$('.bn-bar',v))v.innerHTML='<div class="row spread"><span class="small">Running mean</span><strong class="big number"></strong></div><div class="bn-bar" aria-hidden="true"><div></div></div><div class="track-labels"><span>0</span><span>Batch mean: 2</span></div><div class="rule"></div><div class="formula mono" style="font-size:23px">new = 0.9 × old + 0.1 × 2</div><p class="status" role="status"></p>';
      $('.number',v).textContent=mean.toFixed(3);$('.bn-bar div',v).style.width=`${mean/2*100}%`;
      const status=count?`${count} batch${count===1?'':'es'} observed. ${$('[name=mode]',root).value==='train'?'Train mode moves the buffer toward the batch mean.':'Eval mode keeps the running buffer fixed.'}`:'Start with frozen parameters. Does the buffer still move?';
      if($('.status',v).textContent!==status)$('.status',v).textContent=status;}};});
  mount('unfreeze',root=>{let fixed=false,tried=false;return{action:a=>{if(a==='reset'){fixed=false;tried=false;}else if(a==='fix'){fixed=true;tried=true;}else tried=true;},render:v=>{const code=$('pre',root);code.innerHTML=fixed?'encoder.requires_grad_(True)\noptimizer = AdamW([\n  {"params": encoder.parameters(), "lr": 1e-4},\n  {"params": decoder.parameters(), "lr": 1e-3}\n])\nloss.backward()\noptimizer.step()':'encoder.requires_grad_(True)\noptimizer = AdamW(\n    <span class="accent">decoder.parameters()</span>, lr=1e-3\n)\nloss.backward()\noptimizer.step()';code.style.fontSize=fixed?'16px':'20px';v.innerHTML=`<div class="status ${fixed?'ok':tried?'warning':''}" role="status">${fixed?'Encoder and decoder now belong to the optimizer. With fresh nonzero gradients, both can update. The new optimizer begins with new state.':tried?'The encoder can have gradients, but this optimizer updates only the decoder. Unfreezing alone did not change its parameter list.':'Predict: would the encoder update in this code?'}</div>`;}};});
  mount('rates',root=>{let applied=false;return{action:a=>{applied=a!=='reset';if(a==='reset'){$('[name=encoder]',root).value='0.0001';$('[name=decoder]',root).value='0.001';}},input:()=>applied=false,render:v=>{const e=+$('[name=encoder]',root).value,d=+$('[name=decoder]',root).value;v.innerHTML=`<table class="check-table"><thead><tr><th>Scalar</th><th>Step size η × 0.4</th><th>New w (start 1)</th></tr></thead><tbody><tr><td>Encoder</td><td>${applied?(e*.4).toFixed(5):'—'}</td><td>${applied?(1-e*.4).toFixed(5):'1.00000'}</td></tr><tr><td>Decoder</td><td>${applied?(d*.4).toFixed(5):'—'}</td><td>${applied?(1-d*.4).toFixed(5):'1.00000'}</td></tr></tbody></table><p class="status" role="status">${applied?`The decoder step is ${(d/e).toFixed(2)}× the encoder step for these identical toy gradients.`:'Compare the two step sizes before clicking.'}</p>`;}};});
  const sigmoid=z=>1/(1+Math.exp(-z));
  const bceLogit=(z,y)=>Math.max(z,0)-z*y+Math.log1p(Math.exp(-Math.abs(z)));
  mount('logits',root=>({action:a=>{if(a==='reset'){$('[name=z]',root).value='0';$('[name=target]',root).value='1';}},render:v=>{const z=+$('[name=z]',root).value,y=+$('[name=target]',root).value,p=sigmoid(z);v.innerHTML=`<div class="row spread"><div><div class="tag">Logit z</div><strong class="big number">${z.toFixed(1)}</strong></div><span class="arrow">→</span><div><div class="tag">Probability p</div><strong class="big number">${p.toFixed(3)}</strong></div></div><div class="bn-bar" aria-hidden="true"><div style="width:${p*100}%"></div></div><div class="metric-grid"><div class="metric"><strong>${bceLogit(z,y).toFixed(3)}</strong><span>One-pixel BCE</span></div><div class="metric"><strong>${p>=.5?1:0}</strong><span>Hard mask at ≥ 0.5</span></div><div class="metric"><strong>${y}</strong><span>Target</span></div></div><p class="note-line">${y===1?'Make the logit strongly negative: this becomes a confident foreground miss.':'Make the logit strongly positive: this becomes a confident false positive.'}</p>`;}}));
  const toyTarget=Array.from({length:16},(_,i)=>[5,6,9,10].includes(i)?1:0);
  function softMetrics(positive,negative){const probabilities=toyTarget.map(y=>y?positive:negative),sum=probabilities.reduce((a,b)=>a+b,0),intersection=probabilities.reduce((a,p,i)=>a+p*toyTarget[i],0),dice=(2*intersection+1e-6)/(sum+4+1e-6),bce=probabilities.reduce((a,p,i)=>a-(toyTarget[i]?Math.log(p):Math.log1p(-p)),0)/16;return{probabilities,sum,intersection,dice,bce,total:bce+1-dice};}
  mount('soft',root=>({action:a=>{if(a==='reset'){$('[name=positive]',root).value='.7';$('[name=negative]',root).value='.2';}},render:v=>{const a=+$('[name=positive]',root).value,b=+$('[name=negative]',root).value,m=softMetrics(a,b);v.innerHTML=`<div class="soft-layout"><div><div class="grid-label">Target · 4 positives</div>${grid(toyTarget,4,y=>y?'on':'','4 by 4 target: central 2 by 2 foreground')}</div><div><div class="grid-label">Probabilities</div>${grid(m.probabilities.map(p=>p.toFixed(2)),4,(_,i)=>toyTarget[i]?'filter':'','Probabilities: target cells '+a+', background cells '+b)}</div><div class="stack" style="max-width:200px"><p class="small mono">Σpy = ${m.intersection.toFixed(2)}<br>Σp = ${m.sum.toFixed(2)}<br>Σy = 4</p><p class="small muted">Raise background confidence. Which losses get worse?</p></div></div><div class="metric-grid" style="margin-top:20px"><div class="metric"><strong>${m.bce.toFixed(3)}</strong><span>Mean BCE</span></div><div class="metric"><strong>${m.dice.toFixed(3)}</strong><span>Soft Dice</span></div><div class="metric"><strong>${m.total.toFixed(3)}</strong><span>Combined loss</span></div></div>`;}}));
  const probabilities=Array.from({length:36},(_,i)=>{const r=Math.floor(i/6),c=i%6;return Math.max(.05,Math.round((.9-.18*Math.hypot(r-2.5,c-2.5))*100)/100);});
  function hardMetrics(threshold,empty){let tp=0,fp=0,fn=0;const targets=probabilities.map((_,i)=>empty?0:(Math.floor(i/6)>=2&&Math.floor(i/6)<=3&&i%6>=2&&i%6<=3?1:0));const predicted=probabilities.map(p=>p>=threshold?1:0);predicted.forEach((p,i)=>{if(p&&targets[i])tp++;else if(p)fp++;else if(targets[i])fn++;});const denominator=2*tp+fp+fn;return{tp,fp,fn,dice:denominator?2*tp/denominator:1,targets,predicted};}
  mount('threshold',root=>({action:a=>{if(a==='reset'){$('[name=threshold]',root).value='.5';$('[name=target]',root).value='square';}},render:v=>{const t=+$('[name=threshold]',root).value,empty=$('[name=target]',root).value==='empty',m=hardMetrics(t,empty);v.innerHTML=`<div class="threshold-layout"><div><div class="grid-label">Fixed toy probabilities</div>${grid(probabilities.map(p=>p.toFixed(2)),6,()=>'', 'Fixed radial 6 by 6 toy probability map, values shown in every cell')}</div><div><div class="grid-label">Decision: p ≥ ${t.toFixed(2)}</div>${grid(m.predicted,6,(p,i)=>p?(m.targets[i]?'tp':'fp'):(m.targets[i]?'fn':'tn'),'Hard-mask error map; count summary follows')}</div><div style="width:137px"><div class="tag">Hard Dice</div><div class="big number">${m.dice.toFixed(3)}</div><p class="small mono" style="margin-top:15px">TP ${m.tp}<br>FP ${m.fp}<br>FN ${m.fn}</p></div></div><p class="note-line">${empty?'Empty target: any predicted foreground makes Dice 0; no predicted foreground makes it 1.':'The target is the central 2 × 2 square. A lower threshold usually adds foreground and may add false positives.'}</p>`;}}));
  const auditRows=[['split','Split','Patient A','Patient B',['Patient A','Patient B']],['budget','Epochs','2','10',['2','10']],['loss','Loss','BCE + Dice','BCE + Dice',['BCE + Dice','BCE only']],['policy','Encoder policy','Full tune','Frozen',['Full tune','Frozen']],['threshold','Threshold','0.5','0.5',['0.5','0.3']]];
  mount('audit',root=>{let settings=Object.fromEntries(auditRows.map(r=>[r[0],r[3]]));return{input:e=>{if(e.name in settings)settings[e.name]=e.value;},action:a=>{settings=Object.fromEntries(auditRows.map(r=>[r[0],a==='match'?r[2]:r[3]]));},render:v=>{
    // Build controls once so a select keeps focus while its status updates.
    if(!$('table',v))v.innerHTML=`<table class="check-table"><thead><tr><th>Setting</th><th>Random</th><th>ImageNet</th></tr></thead><tbody>${auditRows.map(([id,label,left,,options])=>`<tr><td>${label}</td><td>${left}</td><td><select name="${id}" aria-label="ImageNet arm ${label}">${options.map(o=>`<option>${o}</option>`).join('')}</select></td></tr>`).join('')}</tbody></table><div class="status" role="status"></div>`;
    auditRows.forEach(([id,,left])=>{const select=$(`[name=${id}]`,v);select.value=settings[id];select.className=left===settings[id]?'ok':'warning';});
    const bad=auditRows.filter(r=>settings[r[0]]!==r[2]);$('.status',v).className=`status ${bad.length?'warning':'ok'}`;$('.status',v).textContent=bad.length?`${bad.length} confounder${bad.length===1?'':'s'}: ${bad.map(r=>r[1].toLowerCase()).join(', ')}. Match these before attributing a difference to pretraining.`:'These displayed settings match. Verify the shared decoder state and remaining controls too; then run the comparison.';
  }};});
  mount('curves',root=>{let mode='train',revealed=false;return{action:a=>{if(a==='reveal')revealed=!revealed;else{mode=a;revealed=false;}},render:v=>{setPressed(root,mode);const random=mode==='train'?[.92,.76,.64,.55,.49,.43]:[.25,.36,.47,.55,.60,.63],pretrained=mode==='train'?[.74,.55,.43,.36,.31,.27]:[.35,.46,.55,.60,.62,.63];const path=values=>values.map((x,i)=>`${i?'L':'M'}${55+i*95},${220-x*175}`).join(' ');v.innerHTML=`<svg class="curve-svg" viewBox="0 0 575 275" role="img" aria-label="Illustrative ${mode==='train'?'training loss falls faster for the pretrained arm':'validation Dice ends at the same value for both arms'}. Not measured results."><line x1="55" y1="20" x2="55" y2="220" stroke="#a9aeb6"/><line x1="55" y1="220" x2="540" y2="220" stroke="#a9aeb6"/>${[0,.5,1].map(x=>`<text x="18" y="${225-x*175}">${x}</text>`).join('')}<path d="${path(random)}" fill="none" stroke="var(--coral)" stroke-width="3" stroke-dasharray="8 6"/><path d="${path(pretrained)}" fill="none" stroke="var(--blue)" stroke-width="4"/>${[1,2,3,4,5,6].map((x,i)=>`<text x="${50+i*95}" y="242">${x}</text>`).join('')}<text x="245" y="258">Equal-budget epochs</text></svg><div class="row small"><span style="color:var(--blue)">━ Pretrained start</span><span style="color:var(--coral)">┄ Random start</span><span class="chip">Illustrative, not measured</span></div>`;$("[data-action=reveal]",root).setAttribute('aria-pressed',String(revealed));$('.answer',root).textContent=revealed?(mode==='train'?'The pretrained arm fits training data faster in this example. That alone says nothing about held-out quality.':'Both end at the same validation Dice here, despite different early learning. A faster start does not guarantee a better final score.'):'Predict what this curve lets us conclude. Then reveal.';}};});
  $$('[data-lab=quiz]').forEach(root=>{
    root.addEventListener('click',e=>{const b=e.target.closest('[data-action]');if(!b)return;const answer=$('.answer',root);if(b.dataset.action==='reset'){answer.className='answer';answer.textContent='Choose an answer, then explain your reasoning.';$$('button',root).forEach(x=>{x.removeAttribute('aria-pressed');delete x.dataset.result;});return;}const correct=+b.dataset.action.split('-')[1]===+root.dataset.correct;answer.className=`answer ${correct?'':'wrong'}`;answer.textContent=correct?'Correct. '+root.dataset.explanation:'Look again at what changes between the two options. Try another answer.';$$('[data-action^=answer]',root).forEach(x=>{x.setAttribute('aria-pressed',String(x===b));if(x===b)x.dataset.result=correct?'correct':'incorrect';else delete x.dataset.result;});});
  });
  // A small model surface supports numerical verification in the browser.
  window.Presentation=Object.freeze({go,slides:slides.map(({id,title,optional})=>({id,title,optional:!!optional})),get current(){return slides[current].id;},get animating(){return !!pending;},math:Object.freeze({convolutionAt,softMetrics,hardMetrics,bceLogit})});
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
