const code=s=>`<pre class="code">${escape(s)}</pre>`;
const split=(visual,label,title,rows,foot='')=>`<div class="body wide-left">${panel(visual)}${lesson(label,title,rows,foot)}</div>`;
const spread=(lt,l,rt,r,foot='')=>`<div class="comparison-spread"><article><div class="comparison-label">Concept</div><h3>${lt}</h3>${l}</article><article><div class="comparison-label">In practice</div><h3>${rt}</h3>${r}</article></div>${foot?`<p class="slide-takeaway">${foot}</p>`:''}`;
const activity=name=>`<div data-distill="${name}" data-ai-help="off"></div>`;
const slides=[];
function S(id,chapter,title,lede,note,body,optional=false){slides.push({id,chapter,title,lede,note,body,optional});}

slides.push({
  id:'start',
  chapter:'23 November 2026 · Session 6',
  title:'Ensemble distillation and pseudo-labeling',
  hero:true,
  note:'index.html',
  body:`<div class="body two"><div class="hero-left"><div class="eyebrow">23 November 2026 · Session 6</div><h1>Ensemble distillation<br>and <em>pseudo-labeling</em></h1><p>Train a single segmentation model with ensemble predictions and unlabeled images.</p><div class="row" style="margin-top:22px">${jump('Open slides ↗','why')}</div></div><figure class="cover-workbench"><figcaption class="tag">Segmentation</figcaption><div class="inf-cover">Teacher<br>Soft masks<br>Student</div><p class="small muted">The ensemble generates training targets. A single student runs at inference.</p></figure></div><nav class="session-route" aria-label="Topics">${[['01','Distillation','Learn from an ensemble','dark-knowledge'],['02','Pseudo-labeling','Learn from unlabeled data','pseudo-intro'],['03','Data quality','Shift, scale, and masks','data-shift'],['04','Calibration','Check confidence scores','calibration']].map(([n,t,d,id])=>`<button data-jump="${id}"><span>${n}</span><strong>${t}</strong><small>${d}</small><i aria-hidden="true">↗</i></button>`).join('')}</nav>`
});

S('why','01 / Distillation','What does an ensemble cost at inference?','An ensemble averages predictions from several models. Test-time augmentation (TTA) adds more passes.','01_distillation.html',
  spread('Why average predictions?',
    '<p>Models make different errors. Averaging their probability maps can reduce isolated false positives and smooth noisy mask boundaries.</p>',
    'Forward passes per image',
    '<div class="inf-equation">5 models × 4 TTA views = 20 passes</div><p>A single model without TTA needs one pass.</p>'));

S('dark-knowledge','01 / Distillation','Hard masks and soft targets','A hard mask assigns 0 or 1. A soft target keeps the teacher’s predicted probability.','01_distillation.html',
  spread('Hard human mask (0 or 1)',
    code('defect core:   1.00\nfuzzy edge:    1.00\nfaint smudge:  0.00\nclean metal:   0.00'),
    'Soft ensemble teacher map',
    code('defect core:   0.94\nfuzzy edge:    0.62  (uncertain border)\nfaint smudge:  0.18\nclean metal:   0.01  (clearly clean)'),
    'Soft targets tell the student where the boundary is fuzzy and which background textures are tricky near-misses.'));

S('temp-lab','01 / Distillation','Temperature changes the soft targets','Drag T from 0.5 to 5.0 and watch the probability bars soften in real time.','01_distillation.html',
  split(activity('temp'),
    'Interactive lab 1',
    'Higher T moves scores<br>toward 0.5.',
    [['T = 1.0','Standard sigmoid probabilities. Extreme logits saturate near 0 and 1.'],
     ['T = 2.0–3.0','Makes extreme probabilities less saturated.'],
     ['At test time','Use T = 1 for the student’s raw predictions.']],
    'Try T = 0.5 vs T = 3.0. Notice how the gap between the faint smudge and clean metal becomes visible.'));

S('distill-loss','01 / Distillation','Blend ground truth with the teacher’s soft map','The loss combines a human mask and a teacher probability map.','01_distillation.html',
  split(code('with torch.no_grad():\n    p_teacher = torch.sigmoid(z_teacher / T)\n\np_student_T = torch.sigmoid(z_student / T)\nloss_hard = bce_dice(z_student, y_true)\nloss_soft = bce(p_student_T, p_teacher) * (T ** 2)\nloss = alpha * loss_hard + (1 - alpha) * loss_soft'),
    'Two loss terms',
    'Hard loss + soft loss',
    [['α ≈ 0.3–0.5','Balances hard ground-truth loss against soft teacher imitation.'],
     ['T² factor','Helps keep the soft loss gradients comparable across temperatures.'],
     ['OOF rule','Generate out-of-fold targets with teachers that held the image out of training.']],
    'For validation, also exclude the student’s validation fold from the teachers’ training data.'));

S('distill-flow','01 / Distillation','Training and deploying the student','Play the animation to see where the ensemble is used.','01_distillation.html',
  split(activity('distill-flow'),
    'Interactive lab 2',
    'Three teachers.<br>One deployed model.',
    [['Training','Run frozen teachers to generate soft targets. Cache the maps to reuse them.'],
     ['Deployment','Deploy the student checkpoint on its own.'],
     ['Comparison','Evaluate against the same student architecture trained on human masks alone.']],
    'Cached targets avoid running the teachers again in every training epoch.'));

S('pseudo-intro','02 / Pseudo-labeling','Pseudo-labeling turns unlabeled images into training data','Unlabeled factory strips are cheap. Pixel masks are expensive.','02_pseudo_labeling.html',
  spread('What you have',
    '<p>A labeled training set with human masks, plus an <strong>unlabeled pool</strong> (such as extra strips from the deployment camera).</p><p>Your best ensemble already makes strong predictions on those unlabeled images.</p>',
    'What you gain',
    '<ol class="inf-steps"><li>More surface textures and defect shapes</li><li>Domain adaptation to the target camera/lighting</li><li>The student architecture and inference cost stay the same</li></ol>',
    'The teacher’s predictions supply the missing training targets.'));

S('pseudo-loop','02 / Pseudo-labeling','The Noisy Student training loop','The teacher labels an image. The student trains on a more heavily augmented version.','02_pseudo_labeling.html',
  spread('Training steps',
    '<ol class="inf-steps"><li>Train a strong teacher (or ensemble) on labeled data</li><li>Predict on unlabeled images using clean inputs + TTA</li><li>Filter low-confidence guesses (or keep soft targets)</li><li>Train a fresh student on labeled + pseudo data with <strong>stronger augmentation</strong></li></ol>',
    'Why “noisy” student?',
    '<p>Teacher sees the <em>easy</em> view (clean + TTA).</p><p>Student sees the <em>hard</em> view (flips, brightness jitter, crops, dropout) and must still match the teacher’s mask.</p><p>Apply the same geometric changes to the image and its target mask.</p>',
    'Soft pseudo-labels preserve uncertainty at defect boundaries.'));

S('pseudo-lab','02 / Pseudo-labeling','How confidence thresholds filter pseudo-labels','Move the confidence gate τ and watch which unlabeled strips enter training.','02_pseudo_labeling.html',
  split(activity('pseudo'),
    'Interactive lab 3',
    'Coverage and<br>label errors',
    [['Low gate (0.50)','High coverage, but teacher false positives sneak into training and get reinforced.'],
     ['High gate (0.95)','Keeps fewer examples and can exclude rare defects.'],
     ['Threshold choice','Check how many examples each class keeps and how often they are correct.']],
    'A false positive that enters training can teach the student to repeat the same error.'));

S('pseudo-rules','02 / Pseudo-labeling','Validation and pseudo-labels','Compare both students on the same held-out images.','02_pseudo_labeling.html',
  spread('Keep validation separate',
    '<p><strong>Rule 1 · Frozen validation:</strong> Never put validation images into the unlabeled pool.</p><p><strong>Rule 2 · No fold leakage:</strong> Fold k’s student must only see pseudo-labels from teachers that never trained on fold k’s validation set.</p>',
    'Control the mix',
    '<p><strong>Rule 3 · Cap the ratio:</strong> Keep real human labels as the anchor (start with 20–30% pseudo-labeled samples per batch).</p><p>Track both positive-mask Dice and empty-image false positives.</p>',
    'If a pseudo-labeled student does not beat the baseline on the exact same untouched validation fold, discard it.'));

S('data-shift','03 / Data quality','Domain shift across camera sources','A pooled cross-validation (CV) score can hide a drop on the deployment camera.','03_data_tricks.html',
  spread('A mixed validation set',
    '<p>Suppose Train = 70% Source A + 30% Source B, while Test = 100% Source B.</p><p>A tweak that gives <strong>+0.02</strong> on Source A and <strong>−0.01</strong> on Source B looks like a win on overall CV—and drops on test!</p>',
    'Scores by source',
    '<p>Run a quick train-vs-test check (brightness, texture, or a simple source classifier).</p><p><strong>Validate and tune on the target-like slice</strong> so every decision reflects the real test distribution.</p>',
    'With a 70/30 validation mix, the overall gain is 0.011 even though Source B drops by 0.01.'));

S('resolution','03 / Data quality','Resolution and small defects','Industrial strips are 800 × 128 (6.25:1). Downsampling erases tiny anomalies.','03_data_tricks.html',
  spread('What stride 32 does',
    '<p>A thin 4-pixel defect shrinks to a fraction of one feature cell in a deep encoder.</p><p>Squashing 800 × 128 into a square distorts horizontal scratches and crushes vertical detail.</p>',
    'Preserve image detail',
    '<p>Preserve the native 6.25:1 aspect ratio.</p><p>If small defects dominate your error budget, test <strong>2× upscaled inputs</strong> or native-resolution crops.</p>',
    'Compare recall by defect size on the same validation images.'));

S('copypaste','03 / Data quality','Copy-paste augmentation for rare defects','Copy pixels and their mask together to create new training examples.','03_data_tricks.html',
  spread('How copy-paste works',
    '<p>Take a real rare-class defect mask and its pixels from one image, paste it onto a clean background strip, and update the target mask.</p><p>Vary the background and placement to create new examples of the rare class.</p>',
    'Visible paste seams',
    '<p>If the pasted patch has a hard straight box or mismatched background brightness, the network cheats by detecting the <strong>paste seam</strong>.</p><p><strong>Fix:</strong> match local background brightness, tighten/feather the mask, and validate only on real unpasted images.</p>',
    'Synthetic augmentation is only useful if it improves performance on untouched real validation images.'));

S('jitter','03 / Data quality','Annotation variation at defect edges','Human labelers rarely agree to the exact pixel on a blurry defect edge.','03_data_tricks.html',
  spread('Variation between annotators',
    '<p>On faint scratches, two careful annotators often differ by <strong>±1 to ±2 pixels</strong> around the perimeter.</p><p>Forcing a model to predict 0.999 vs 0.001 on a fuzzy border makes it memorize annotator quirks.</p>',
    'What a soft target retains',
    '<p>Averaging diverse models smooths out random ±2px jitter.</p><p>The student can learn intermediate probabilities around unclear boundaries.</p>'));

S('calibration','04 / Calibration','What does a score of 0.70 mean?','If scores are calibrated, about 70% of pixels scored near 0.70 are true positives.','04_calibration_and_safety.html',
  spread('Why it matters',
    '<p>An overconfident weak model can drown out a calibrated strong model in an ensemble.</p><p>An overconfident teacher turns soft distillation targets back into hard 0/1 masks.</p>',
    'Fit on out-of-fold predictions',
    '<div class="inf-equation">Platt: p<sub>cal</sub> = σ(a·z + b)</div><p>Fit 1 scalar (temperature T) or 2 scalars (a, b) per class on <strong>out-of-fold logits</strong>. Monotonic, fast, and cheap.</p>',
    'Calibrate on held-out OOF logits before blending ensembles, distilling, or setting pseudo-label gates.'));

S('calibration-lab','04 / Calibration','Calibration and a fixed threshold','Apply calibration to see which teacher scores cross the 0.50 threshold.','04_calibration_and_safety.html',
  split(activity('calib'),
    'Interactive lab 4',
    'The same pixels.<br>Different scores.',
    [['Overconfident','The smudge scores 0.62 and passes the 0.50 threshold.'],
     ['Calibrated','The smudge drops to 0.24. The scratch still passes the threshold.'],
     ['Ranking','The four pixels stay in the same order.']],
    'A monotonic calibration map can change which pixels pass a fixed threshold.'));

S('checklist','04 / Calibration','How to compare experiments','Keep the evaluation fixed and isolate the effect of each change.','04_calibration_and_safety.html',
  `<div class="body"><table class="inf-table"><tr><th>Step</th><th>What to check</th><th>Red flag</th></tr><tr><td>1 · Validation</td><td>Locked held-out folds (on target-like slice if shifted)</td><td>Validation images touched by teacher or pseudo pool</td></tr><tr><td>2 · Calibration</td><td>Inspect teacher probability sharpness on OOF</td><td>Teacher outputs 0.99 on wrong predictions</td></tr><tr><td>3 · Isolation</td><td>Change one lever at a time (distill OR pseudo OR paste)</td><td>Bundling 4 changes and guessing what worked</td></tr><tr><td>4 · Error types</td><td>Track positive Dice AND empty-image false positives</td><td>Higher positive Dice hiding a flood of false positives</td></tr></table></div><p class="slide-takeaway">If a fancier loop does not beat the simpler baseline on your locked validation set, keep the simpler recipe.</p>`);

S('exit','04 / Review','Check your understanding','','05_reference.html',
  split(quiz('Which teacher can generate an out-of-fold target for an image in fold 2?',
    ['A teacher trained on all folds',
     'A teacher trained on the other folds, with fold 2 held out',
     'Any teacher with a low training loss'],
    1,
    'An out-of-fold prediction comes from a model that held the image out of training.'),
    'Out-of-fold predictions',
    'Hold out the<br>target image.',
    [['Fold 2','Generate targets with a teacher trained on the other folds.'],
     ['Evaluation','Keep the student’s validation split out of teacher training and the pseudo-label pool.']]));

S('advanced','Further reading','Multi-generation self-training (Gen 1 → Gen 2 → Gen 3)','A trained student can become a teacher in the next round.','05_reference.html',
  spread('Iterative generations',
    '<p><strong>Gen 1:</strong> Diverse ensemble trained on real labels.</p><p><strong>Gen 2:</strong> Stronger/larger students distilled from Gen 1 OOF + soft pseudo-labels.</p><p><strong>Gen 3:</strong> New diverse family trained with Gen 2 teachers + higher resolution.</p>',
    'When to stop',
    '<p>Stop the moment held-out validation stops improving or empty-image false positives start creeping up.</p><p>Compare each generation with the previous one on the same held-out split.</p>'),true);
