const code=s=>`<pre class="code">${escape(s)}</pre>`;
const split=(visual,label,title,rows,foot='')=>`<div class="body wide-left">${panel(visual)}${lesson(label,title,rows,foot)}</div>`;
const spread=(lt,l,rt,r,foot='')=>`<div class="comparison-spread"><article><div class="comparison-label">Concept</div><h3>${lt}</h3>${l}</article><article><div class="comparison-label">In practice</div><h3>${rt}</h3>${r}</article></div>${foot?`<p class="slide-takeaway">${foot}</p>`:''}`;
const activity=name=>`<div data-distill="${name}" data-ai-help="off"></div>`;
const slides=[];
function S(id,chapter,title,lede,note,body,optional=false){slides.push({id,chapter,title,lede,note,body,optional});}

slides.push({
  id:'start',
  chapter:'23 November 2026 · Session 6',
  title:'Distillation, pseudo-labeling & data tricks',
  hero:true,
  note:'index.html',
  body:`<div class="body two"><div class="hero-left"><div class="eyebrow">23 November 2026 · 40-Minute Live Session</div><h1>Teach smaller.<br>Label <em>smarter.</em></h1><p>Ensemble distillation, pseudo-labeling, and the data-centric habits that actually move segmentation scores.</p><div class="row" style="margin-top:22px">${jump('Start the 40-min session ↗','why')}</div><p class="small muted">40 minutes · 4 interactive animated labs<br>Zero-jargon intuition &amp; safety checks</p></div><figure class="cover-workbench"><figcaption class="tag">Four levers beyond architecture</figcaption><div class="inf-cover">Distill.<br>Pseudo.<br>Clean.<br>Calibrate.</div><p class="small muted">Shrink heavy ensembles, learn from unlabeled strips, and fix the data bottlenecks first.</p></figure></div><nav class="session-route" aria-label="Session roadmap">${[['01','Distillation','Ensemble → 1 fast student','dark-knowledge'],['02','Pseudo-Labels','Learn from unlabeled data','pseudo-intro'],['03','Data Tricks','Shift, scale, paste & jitter','data-shift'],['04','Safety','Calibration & 4-step check','calibration']].map(([n,t,d,id])=>`<button data-jump="${id}"><span>${n}</span><strong>${t}</strong><small>${d}</small><i aria-hidden="true">↗</i></button>`).join('')}</nav>`
});

S('why','01 / Motivation','The ensemble dilemma: great to win, heavy to ship','Ensembles cancel out errors—but multiply latency, memory, and GPU cost.','01_distillation.html',
  spread('Why ensembles win',
    '<p>Averaging 3–5 diverse architectures (e.g., CNN + Transformer) cancels random mistakes and smooths noisy mask borders.</p><p>On segmentation tasks, ensembles routinely beat every single member.</p>',
    'Why they hurt to deploy',
    '<div class="inf-equation">5 models × 4 TTA views<br>= 20 forward passes / image</div><p>Too slow for live cameras, too heavy for small GPUs, too expensive at scale.</p>',
    'Question for the next 10 minutes: how do we keep most of the ensemble’s accuracy at single-model speed?'));

S('dark-knowledge','01 / Distillation','Soft targets carry “dark knowledge” that 0/1 masks hide','A binary mask says yes or no. An ensemble probability map explains how sure it is.','01_distillation.html',
  spread('Hard human mask (0 or 1)',
    code('defect core:   1.00\nfuzzy edge:    1.00  (looks identical!)\nfaint smudge:  0.00\nclean metal:   0.00  (looks identical!)'),
    'Soft ensemble teacher map',
    code('defect core:   0.94\nfuzzy edge:    0.62  (uncertain border)\nfaint smudge:  0.18  (hard negative!)\nclean metal:   0.01  (clearly clean)'),
    'Soft targets tell the student where the boundary is fuzzy and which background textures are tricky near-misses.'));

S('temp-lab','01 / Distillation','Temperature scaling unlocks hidden teacher signal','Drag T from 0.5 to 5.0 and watch the probability bars soften in real time.','01_distillation.html',
  split(activity('temp'),
    'Interactive lab 1',
    'Higher T reveals<br>relative confidence.',
    [['T = 1.0','Standard sigmoid probabilities. Extreme logits saturate near 0 and 1.'],
     ['T = 2.0–3.0','Softens sharp peaks so the student sees gradient signal on edges and smudges.'],
     ['At test time','Always reset the student back to T = 1.0.']],
    'Try T = 0.5 vs T = 3.0. Notice how the gap between the faint smudge and clean metal becomes visible.'));

S('distill-loss','01 / Distillation','Blend ground truth with the teacher’s soft map','Never trust the teacher blindly—keep real labels in the loop.','01_distillation.html',
  split(code('with torch.no_grad():\n    p_teacher = torch.sigmoid(z_teacher / T)\n\np_student_T = torch.sigmoid(z_student / T)\nloss_hard = bce_dice(z_student, y_true)\nloss_soft = bce(p_student_T, p_teacher) * (T ** 2)\nloss = alpha * loss_hard + (1 - alpha) * loss_soft'),
    'Two loss terms',
    'Real labels anchor.<br>Soft maps guide.',
    [['α ≈ 0.3–0.5','Balances hard ground-truth loss against soft teacher imitation.'],
     ['T² factor','Compensates for the 1/T² gradient shrinkage from dividing logits by T.'],
     ['OOF rule','Generate teacher maps strictly out-of-fold so teachers never leak memorized training images.']],
    'If a teacher trained on image X, its soft map on image X is artificially overconfident. Always use out-of-fold teachers.'));

S('distill-flow','01 / Distillation','Watch an ensemble compress into one student','Click Animate to trace a batch from 3 teachers into 1 deployment-ready student.','01_distillation.html',
  split(activity('distill-flow'),
    'Interactive lab 2',
    '3× training effort.<br>1× inference cost.',
    [['Training','Run teachers once offline (OOF) and cache their soft maps—or run them frozen.'],
     ['Deployment','Throw the heavy ensemble away. Ship only the single student checkpoint.'],
     ['Bonus','A student trained on diverse teachers often beats the same architecture trained from scratch.']],
    'Caching OOF teacher maps on disk once means student training runs at normal single-model speed.'));

S('pseudo-intro','02 / Pseudo-Labeling','Pseudo-labeling turns unlabeled images into training data','Unlabeled factory strips are cheap. Pixel masks are expensive.','02_pseudo_labeling.html',
  spread('What you have',
    '<p>A labeled training set with human masks, plus an <strong>unlabeled pool</strong> (extra strips or target-domain test images).</p><p>Your best ensemble already makes strong predictions on those unlabeled images.</p>',
    'What you gain',
    '<ol class="inf-steps"><li>More surface textures and defect shapes</li><li>Domain adaptation to the target camera/lighting</li><li>Stronger single-model student at zero extra inference cost</li></ol>',
    'Think of pseudo-labeling as distillation extended to images that have no human mask at all.'));

S('pseudo-loop','02 / Pseudo-Labeling','The Noisy Student loop: predict clean, train noisy','The student improves over the teacher because it has to solve a harder version of the same image.','02_pseudo_labeling.html',
  spread('Step-by-step loop',
    '<ol class="inf-steps"><li>Train a strong teacher (or ensemble) on labeled data</li><li>Predict on unlabeled images using clean inputs + TTA</li><li>Filter low-confidence guesses (or keep soft targets)</li><li>Train a fresh student on labeled + pseudo data with <strong>stronger augmentation</strong></li></ol>',
    'Why “noisy” student?',
    '<p>Teacher sees the <em>easy</em> view (clean + TTA).</p><p>Student sees the <em>hard</em> view (flips, brightness jitter, crops, dropout) and must still match the teacher’s mask.</p><p>That forces generalization instead of copying.</p>',
    'Hard 0/1 pseudo-labels lock in edge mistakes. Continuous soft pseudo-labels preserve uncertainty.'));

S('pseudo-lab','02 / Pseudo-Labeling','Confidence gating vs. confirmation bias','Move the confidence gate τ and watch which unlabeled strips enter training.','02_pseudo_labeling.html',
  split(activity('pseudo'),
    'Interactive lab 3',
    'Too low = noise.<br>Too high = no data.',
    [['Low gate (0.50)','High coverage, but teacher false positives sneak into training and get reinforced.'],
     ['High gate (0.95)','Very clean, but you throw away most informative borderline and rare-class examples.'],
     ['Sweet spot','Filter the ambiguous middle band, or use soft targets with a lower loss weight (e.g. 0.5×).']],
    'Watch the Confirmation Bias meter as you lower the gate—reinforcing a model’s own false positives is the #1 failure mode.'));

S('pseudo-rules','02 / Pseudo-Labeling','Three rules that prevent silent leakage','A pseudo-labeling gain only counts if your validation ruler never moved.','02_pseudo_labeling.html',
  spread('Protect the ruler',
    '<p><strong>Rule 1 · Frozen validation:</strong> Never put validation images into the unlabeled pool.</p><p><strong>Rule 2 · No fold leakage:</strong> Fold k’s student must only see pseudo-labels from teachers that never trained on fold k’s validation set.</p>',
    'Control the mix',
    '<p><strong>Rule 3 · Cap the ratio:</strong> Keep real human labels as the anchor (start with 20–30% pseudo-labeled samples per batch).</p><p>Track both positive-mask Dice and empty-image false positives.</p>',
    'If a pseudo-labeled student does not beat the baseline on the exact same untouched validation fold, discard it.'));

S('data-shift','03 / Data Tricks','Domain shift: why your CV and test set can disagree','If training mixes two camera sources but test uses only one, overall CV lies.','03_data_tricks.html',
  spread('The hidden mix trap',
    '<p>Suppose Train = 70% Source A + 30% Source B, while Test = 100% Source B.</p><p>A tweak that gives <strong>+0.02</strong> on Source A and <strong>−0.01</strong> on Source B looks like a win on overall CV—and drops on test!</p>',
    'The data-centric fix',
    '<p>Run a quick train-vs-test check (brightness, texture, or a simple source classifier).</p><p><strong>Validate and tune on the target-like slice</strong> so every decision reflects the real test distribution.</p>',
    'Before blaming your model architecture, check whether your validation set actually looks like your test set.'));

S('resolution','03 / Data Tricks','Resolution & aspect ratio: small defects vanish fast','Industrial strips are 800 × 128 (6.25:1). Downsampling erases tiny anomalies.','03_data_tricks.html',
  spread('What stride 32 does',
    '<p>A thin 4-pixel defect shrinks to a fraction of one feature cell in a deep encoder.</p><p>Squashing 800 × 128 into a square distorts horizontal scratches and crushes vertical detail.</p>',
    'What actually helps',
    '<p>Preserve the native 6.25:1 aspect ratio.</p><p>If small defects dominate your error budget, test <strong>2× upscaled inputs</strong> or native-resolution crops—often a bigger jump than changing backbones.</p>',
    'Always check your error budget by defect size before spending hours tuning optimizers.'));

S('copypaste','03 / Data Tricks','Copy-paste augmentation for rare classes (and its trap)','Pasting real rare defects onto clean strips boosts recall—unless the seam gives it away.','03_data_tricks.html',
  spread('How copy-paste works',
    '<p>Take a real rare-class defect mask and its pixels from one image, paste it onto a clean background strip, and update the target mask.</p><p>Instantly creates hundreds of new rare-class training examples.</p>',
    'The edge-halo trap',
    '<p>If the pasted patch has a hard straight box or mismatched background brightness, the network cheats by detecting the <strong>paste seam</strong>.</p><p><strong>Fix:</strong> match local background brightness, tighten/feather the mask, and validate only on real unpasted images.</p>',
    'Synthetic augmentation is only useful if it improves performance on untouched real validation images.'));

S('jitter','03 / Data Tricks','Annotation jitter: why ±2px label noise is normal','Human labelers rarely agree to the exact pixel on a blurry defect edge.','03_data_tricks.html',
  spread('The reality of masks',
    '<p>On faint scratches, two careful annotators often differ by <strong>±1 to ±2 pixels</strong> around the perimeter.</p><p>Forcing a model to predict 0.999 vs 0.001 on a fuzzy border makes it memorize annotator quirks.</p>',
    'Why soft targets win here',
    '<p>Averaging diverse models smooths out random ±2px jitter.</p><p>Distilling those soft probabilities teaches the student where the true uncertainty lives.</p>',
    'Stop chasing 0.0001 training loss on noisy borders—smooth targets generalize better.'));

S('calibration','04 / Safety','Calibration: make probabilities mean what they say','Modern neural networks are often overconfident—especially with Dice loss.','04_calibration_and_safety.html',
  spread('Why it matters',
    '<p>An overconfident weak model can drown out a calibrated strong model in an ensemble.</p><p>An overconfident teacher turns soft distillation targets back into hard 0/1 masks.</p>',
    'Simple OOF fix',
    '<div class="inf-equation">Platt: p<sub>cal</sub> = σ(a·z + b)</div><p>Fit 1 scalar (temperature T) or 2 scalars (a, b) per class on <strong>out-of-fold logits</strong>. Monotonic, fast, and cheap.</p>',
    'Calibrate on held-out OOF logits before blending ensembles, distilling, or setting pseudo-label gates.'));

S('calibration-lab','04 / Safety','Compare an overconfident vs. calibrated teacher','Toggle calibration and watch how the student’s boundary & empty-strip decisions change.','04_calibration_and_safety.html',
  split(activity('calib'),
    'Interactive lab 4',
    'Honest probabilities<br>protect the student.',
    [['Overconfident','Extreme 0.97 / 0.02 scores hide the uncertain edge and trigger false positives on smudges.'],
     ['Calibrated','Platt / T scaling restores realistic 0.72 / 0.14 scores so thresholds and soft losses behave cleanly.'],
     ['Takeaway','Check calibration before blaming the student architecture.']],
    'Calibration does not change pixel ranking inside one class—it fixes how scores combine across models and thresholds.'));

S('checklist','04 / Safety','The 4-step safety checklist for your next experiment','Keep this on your desk before trusting any distillation or pseudo-labeling run.','04_calibration_and_safety.html',
  `<div class="body"><table class="inf-table"><tr><th>Step</th><th>What to check</th><th>Red flag</th></tr><tr><td>1 · Ruler</td><td>Locked held-out folds (on target-like slice if shifted)</td><td>Validation images touched by teacher or pseudo pool</td></tr><tr><td>2 · Calibration</td><td>Inspect teacher probability sharpness on OOF</td><td>Teacher outputs 0.99 on wrong predictions</td></tr><tr><td>3 · Isolation</td><td>Change one lever at a time (distill OR pseudo OR paste)</td><td>Bundling 4 changes and guessing what worked</td></tr><tr><td>4 · Both sides</td><td>Track positive Dice AND empty-image false positives</td><td>Higher positive Dice hiding a flood of false positives</td></tr></table></div><p class="slide-takeaway">If a fancier loop does not beat the simpler baseline on your locked validation set, keep the simpler recipe.</p>`);

S('exit','04 / Exit ticket','One check before we wrap up','38–40 min · Pick an answer and explain why.','05_reference.html',
  split(quiz('Why must distillation teachers be generated out-of-fold (OOF)?',
    ['To make the GPU run faster during inference',
     'Because in-fold teachers memorized the training images and produce overconfident, leaky targets',
     'Because soft targets only work with binary 0/1 masks'],
    1,
    'An in-fold teacher has already seen those training images, so its predictions are artificially sharp and leak validation information.'),
    'Wrap-up discussion',
    'Your next<br>defensible step.',
    [['Distillation','When is shrinking an ensemble worth the extra training step?'],
     ['Pseudo-labeling','How will you guard against confirmation bias?'],
     ['Data-centric','Which data check (shift, scale, rare class, jitter) will you run first?']],
    'Remember: better data understanding + honest validation beats blind complexity every time.'));

S('advanced','Optional / Beyond 40 min','Multi-generation self-training (Gen 1 → Gen 2 → Gen 3)','When you have compute and a locked validation ruler, you can iterate—carefully.','05_reference.html',
  spread('Iterative generations',
    '<p><strong>Gen 1:</strong> Diverse ensemble trained on real labels.</p><p><strong>Gen 2:</strong> Stronger/larger students distilled from Gen 1 OOF + soft pseudo-labels.</p><p><strong>Gen 3:</strong> New diverse family trained with Gen 2 teachers + higher resolution.</p>',
    'When to stop',
    '<p>Stop the moment held-out validation stops improving or empty-image false positives start creeping up.</p><p>Diminishing returns arrive fast after 2–3 generations.</p>',
    'Optional depth only. Never iterate generations without a frozen, leak-free validation split.'),true);
