/* Interactive animated teaching labs for Session 6. Deterministic toy numbers, zero network calls. */
(() => {
'use strict';
const root = document;
const sigmoid = z => 1 / (1 + Math.exp(-z));

/* Lab 1: Temperature Scaling Explorer */
function renderTemp(el) {
  const items = [
    { name: 'Defect core', z: 3.2 },
    { name: 'Fuzzy edge', z: 0.6 },
    { name: 'Faint smudge', z: -1.6 },
    { name: 'Clean metal', z: -4.2 }
  ];
  el.innerHTML = `
    <div class="inf-controls">
      <label>Temperature T <input type="range" min="0.5" max="5.0" step="0.1" value="1.0"><output>1.0</output></label>
      <button type="button" data-preset="1.0">T = 1</button>
      <button type="button" data-preset="2.5">T = 2.5</button>
      <button type="button" data-preset="4.0">T = 4</button>
    </div>
    <div class="d6-bars" aria-live="polite"></div>
    <p class="inf-readout" role="status"></p>`;
  const slider = el.querySelector('input');
  const out = el.querySelector('output');
  const bars = el.querySelector('.d6-bars');
  const status = el.querySelector('[role=status]');
  bars.innerHTML = items.map((it, i) => `
    <div class="d6-bar-row">
      <span class="d6-bar-label">${it.name} <small>(z=${it.z})</small></span>
      <div class="d6-bar-track"><div class="d6-bar-fill" data-idx="${i}"></div></div>
      <span class="d6-bar-val" data-val="${i}">0.00</span>
    </div>`).join('');
  const draw = () => {
    const T = +slider.value;
    out.textContent = T.toFixed(1);
    const ps = items.map(it => sigmoid(it.z / T));
    ps.forEach((p, i) => {
      bars.querySelector(`[data-idx="${i}"]`).style.width = `${(p * 100).toFixed(1)}%`;
      bars.querySelector(`[data-val="${i}"]`).textContent = p.toFixed(3);
    });
    const gap = (ps[2] - ps[3]).toFixed(3);
    status.textContent = T < 1
      ? `Sharp (T=${T.toFixed(1)}): smudge vs clean gap is ${gap}.`
      : T <= 1.2
        ? `Standard (T=${T.toFixed(1)}): smudge=${ps[2].toFixed(2)}, clean=${ps[3].toFixed(2)} (gap ${gap}).`
        : `Softened (T=${T.toFixed(1)}): smudge=${ps[2].toFixed(2)} vs clean=${ps[3].toFixed(2)} (gap ${gap}).`;
  };
  slider.oninput = draw;
  el.querySelectorAll('[data-preset]').forEach(b => b.onclick = () => { slider.value = b.dataset.preset; draw(); });
  draw();
}

/* Lab 2: Animated Teacher -> Student Pipeline */
function renderDistillFlow(el) {
  let step = 0, timer = null;
  const stages = [
    { label: 'Ready', desc: 'Play the animation or advance one step at a time.' },
    { label: 'Step 1 · Ensemble teachers', desc: 'Three teachers each predict a probability map for the same held-out image.' },
    { label: 'Step 2 · Soft average', desc: 'Average the three probability maps to form a soft target.' },
    { label: 'Step 3 · Single student', desc: 'Train the student on the soft target and the human mask. Deploy the student on its own.' }
  ];
  el.innerHTML = `
    <div class="inf-controls">
      <button type="button" data-play>Animate flow ▶</button>
      <button type="button" data-step>Next step</button>
      <button type="button" data-reset>Reset</button>
    </div>
    <div class="d6-flow">
      <div class="d6-node" data-stage="1"><strong>3 Teachers</strong><small>3 forward passes</small></div>
      <div class="d6-arrow" data-stage="2">→</div>
      <div class="d6-node" data-stage="2"><strong>Soft Map</strong><small>0.94 / 0.62 / 0.18</small></div>
      <div class="d6-arrow" data-stage="3">→</div>
      <div class="d6-node d6-student" data-stage="3"><strong>1 Student</strong><small>1 forward pass</small></div>
    </div>
    <div class="d6-metrics">
      <div><span>Inference passes</span><strong data-m="passes">3 → 1</strong></div>
      <div><span>Deployed models</span><strong>1</strong></div>
      <div><span>Boundary signal</span><strong data-m="sig">Soft OOF</strong></div>
    </div>
    <p class="inf-readout" role="status"></p>`;
  const draw = () => {
    el.querySelectorAll('[data-stage]').forEach(n => {
      const s = +n.dataset.stage;
      n.classList.toggle('is-active', s === step);
      n.classList.toggle('is-done', s < step);
    });
    el.querySelector('[role=status]').textContent = `${stages[step].label}: ${stages[step].desc}`;
  };
  const stop = () => { if (timer) { clearInterval(timer); timer = null; } };
  el.querySelector('[data-play]').onclick = () => {
    stop();
    step = 1;
    draw();
    timer = setInterval(() => {
      if (step < 3) { step++; draw(); } else { stop(); }
    }, 1100);
  };
  el.querySelector('[data-step]').onclick = () => { stop(); step = step < 3 ? step + 1 : 0; draw(); };
  el.querySelector('[data-reset]').onclick = () => { stop(); step = 0; draw(); };
  draw();
}

/* Lab 3: Confidence Gate & Confirmation Bias Simulator */
function renderPseudo(el) {
  const strips = [
    { id: 'S1', p: 0.96, ok: true },
    { id: 'S2', p: 0.91, ok: true },
    { id: 'S3', p: 0.84, ok: true },
    { id: 'S4', p: 0.76, ok: false },
    { id: 'S5', p: 0.68, ok: true },
    { id: 'S6', p: 0.59, ok: false },
    { id: 'S7', p: 0.52, ok: false },
    { id: 'S8', p: 0.88, ok: true }
  ];
  el.innerHTML = `
    <div class="inf-controls">
      <label>Confidence gate τ <input type="range" min="0.50" max="0.95" step="0.05" value="0.80"><output>0.80</output></label>
      <button type="button" data-reset>Reset</button>
    </div>
    <div class="d6-strips"></div>
    <p class="inf-readout" role="status"></p>`;
  const slider = el.querySelector('input');
  const out = el.querySelector('output');
  const box = el.querySelector('.d6-strips');
  const status = el.querySelector('[role=status]');
  const draw = () => {
    const tau = +slider.value;
    out.textContent = tau.toFixed(2);
    let kept = 0, wrong = 0;
    box.innerHTML = strips.map(s => {
      const pass = s.p >= tau;
      if (pass) { kept++; if (!s.ok) wrong++; }
      const cls = pass ? (s.ok ? 'is-good' : 'is-bad') : 'is-skip';
      return `<div class="d6-strip ${cls}"><span>${s.id}</span><strong>${s.p.toFixed(2)}</strong><small>${pass ? (s.ok ? '✓ kept' : 'false positive') : 'skip'}</small></div>`;
    }).join('');
    const risk = kept === 0 ? 'None (0 kept)' : wrong === 0 ? 'Low (clean)' : wrong === 1 ? 'Moderate (1 FP)' : `High (${wrong} false positives)`;
    status.textContent = `Kept ${kept}/${strips.length} unlabeled strips · Wrong pseudo-labels: ${wrong} · Confirmation bias risk: ${risk}.`;
  };
  slider.oninput = draw;
  el.querySelector('[data-reset]').onclick = () => { slider.value = '0.80'; draw(); };
  draw();
}

/* Lab 4: Overconfident vs Calibrated Teacher */
function renderCalib(el) {
  let calibrated = false;
  const rows = [
    { name: 'True scratch', raw: 0.99, cal: 0.88 },
    { name: 'Uncertain edge', raw: 0.91, cal: 0.61 },
    { name: 'Tricky smudge (FP)', raw: 0.62, cal: 0.24 },
    { name: 'Clean steel', raw: 0.04, cal: 0.02 }
  ];
  el.innerHTML = `
    <div class="inf-controls">
      <label><input type="checkbox"> Apply example calibration</label>
      <button type="button" data-reset>Reset</button>
    </div>
    <div class="d6-bars"></div>
    <p class="inf-readout" role="status"></p>`;
  const cb = el.querySelector('input');
  const bars = el.querySelector('.d6-bars');
  const status = el.querySelector('[role=status]');
  bars.innerHTML = rows.map((r, i) => `
    <div class="d6-bar-row">
      <span class="d6-bar-label">${r.name}</span>
      <div class="d6-bar-track"><div class="d6-bar-fill" data-idx="${i}"></div></div>
      <span class="d6-bar-val" data-val="${i}">0.00</span>
    </div>`).join('');
  const draw = () => {
    rows.forEach((r, i) => {
      const p = calibrated ? r.cal : r.raw;
      const fill = bars.querySelector(`[data-idx="${i}"]`);
      fill.style.width = `${(p * 100).toFixed(1)}%`;
      fill.classList.toggle('is-warn', !calibrated && i === 2);
      bars.querySelector(`[data-val="${i}"]`).textContent = p.toFixed(2);
    });
    status.textContent = calibrated
      ? 'Calibrated: smudge drops to 0.24 (below 0.50 threshold). The edge scores 0.61.'
      : 'Overconfident: smudge scores 0.62 and passes the 0.50 threshold. The edge scores 0.91.';
  };
  cb.onchange = e => { calibrated = e.target.checked; draw(); };
  el.querySelector('[data-reset]').onclick = () => { calibrated = false; cb.checked = false; draw(); };
  draw();
}

for (const el of root.querySelectorAll('[data-distill]')) {
  const fn = { temp: renderTemp, 'distill-flow': renderDistillFlow, pseudo: renderPseudo, calib: renderCalib }[el.dataset.distill];
  if (fn) fn(el);
}

/* Accessible quiz handler */
root.addEventListener('click', e => {
  const b = e.target.closest('[data-lab="quiz"] button');
  if (!b) return;
  const q = b.closest('[data-lab="quiz"]'), answer = q.querySelector('.answer');
  if (b.dataset.action === 'reset') {
    answer.textContent = 'Choose an answer.';
    answer.classList.remove('wrong');
    q.querySelectorAll('.answer-button').forEach(x => x.removeAttribute('aria-pressed'));
    return;
  }
  const selected = Number(b.dataset.action.replace('answer-', ''));
  const correct = selected === Number(q.dataset.correct);
  answer.textContent = (correct ? 'Correct. ' : 'Try again. ') + q.dataset.explanation;
  answer.classList.toggle('wrong', !correct);
  q.querySelectorAll('.answer-button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
});
})();
