/* Focused teaching labs. Event-driven rendering: no animation loop or focus capture. */
"use strict";
const seedRng = (seed) => () => {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const range = (key, title, min, max, value, step = 1) =>
  `<label>${title}<input data-control="${key}" aria-label="${title}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"></label>`;
function startLab(el, title, markup) {
  el.innerHTML = `<h3>${title}</h3>${markup}<output aria-live="polite"></output>`;
  return {
    q: (s) => el.querySelector(s),
    all: (s) => [...el.querySelectorAll(s)],
    out: el.querySelector("output"),
  };
}
function canvas(el, key, w, h, draw) {
  const cv = el.querySelector(`[data-canvas="${key}"]`);
  const ctx = cv.getContext("2d");
  let dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  cv.style.aspectRatio = `${w}/${h}`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw(ctx, w, h);
}
function encode(bits) {
  let pairs = [];
  for (let i = 0; i < bits.length; ) {
    if (!bits[i]) {
      i++;
      continue;
    }
    let s = i;
    while (i < bits.length && bits[i]) i++;
    pairs.push([s + 1, i - s]);
  }
  return pairs;
}
const labs = {
  cardinality(el) {
    const { q, out } = startLab(
      el,
      "Switch labels on; count three different units",
      `<p>Each switch represents one image–class pair, not one pixel or connected component.</p><div class="labels"></div>`,
    );
    let data = [
      [0, 0, 0, 0],
      [1, 0, 0, 0],
      [1, 0, 1, 0],
      [0, 1, 1, 1],
    ];
    const names = ["A", "B", "C", "D"];
    const area = q(".labels");
    data.forEach((row, i) => {
      let group = document.createElement("div");
      group.className = "controls";
      group.append(`Strip ${i + 1} `);
      row.forEach((v, j) => {
        let b = document.createElement("button");
        b.textContent = names[j];
        b.setAttribute("aria-label", `Toggle strip ${i + 1} class ${names[j]}`);
        b.setAttribute("aria-pressed", !!v);
        b.onclick = () => {
          data[i][j] ^= 1;
          b.setAttribute("aria-pressed", !!data[i][j]);
          render();
        };
        group.append(b);
      });
      area.append(group);
    });
    function render() {
      let sums = data.map((r) => r.reduce((a, b) => a + b, 0));
      out.textContent = `4 images × 4 classes = 16 CSV rows. Positive pairs: ${sums.reduce((a, b) => a + b, 0)}. Clean images: ${sums.filter((v) => !v).length}. Cardinalities: [${sums.join(", ")}]. Class counts: [${names.map((_, j) => data.reduce((a, r) => a + r[j], 0)).join(", ")}].`;
    }
    render();
  },
  overlap(el) {
    const { q, out } = startLab(
      el,
      "Image co-occurrence vs. pixel overlap",
      range("shift", "Move C region left → right", 0, 6, 5) +
        `<canvas data-canvas="map" role="img" aria-label="A and C masks on a toy strip; overlap is highlighted"></canvas>`,
    );
    function render() {
      let s = +q("input").value,
        overlap = Math.max(0, Math.min(4, s + 3) - Math.max(1, s)) * 3;
      canvas(el, "map", 560, 200, (c, w, h) => {
        c.fillStyle = "#f8f5ed";
        c.fillRect(0, 0, w, h);
        for (let y = 0; y < 4; y++)
          for (let x = 0; x < 10; x++) {
            let a = x >= 1 && x < 4 && y < 3,
              b = x >= s && x < s + 3 && y < 3;
            c.fillStyle =
              a && b ? "#73419d" : a ? "#087f8c" : b ? "#df653c" : "#e2ddd1";
            c.fillRect(12 + x * 53, 18 + y * 40, 51, 38);
            c.fillStyle = a || b ? "#fff" : "#172d3a";
            c.font = "13px monospace";
            c.fillText(
              a && b ? "A+C" : a ? "A" : b ? "C" : "0",
              20 + x * 53,
              42 + y * 40,
            );
          }
      });
      out.textContent = `Image presence stays [1,0,1,0]. Overlapping pixels = ${overlap} / 40. ${overlap ? "Integer targets cannot retain both labels at those pixels." : "Each pixel has at most one class; a background-inclusive softmax target is possible."}`;
    }
    q("input").oninput = render;
    render();
  },
  heads(el) {
    const { q, all, out } = startLab(
      el,
      "Change a logit; watch probability competition",
      `<p>Five logits: background, A, B, C, D. Four sigmoid outputs ignore the background logit.</p>` +
        ["BG", "A", "B", "C", "D"]
          .map((s, i) =>
            range(i, s + " logit", -4, 4, [0, 2, 1, -1, -2][i], 0.1),
          )
          .join("") +
        '<div class="twocol"><div class="sig"></div><div class="soft"></div></div>',
    );
    function render() {
      let z = all("input").map((i) => +i.value),
        m = Math.max(...z),
        e = z.map((v) => Math.exp(v - m)),
        sum = e.reduce((a, b) => a + b, 0),
        soft = e.map((v) => v / sum),
        sig = z.slice(1).map((v) => 1 / (1 + Math.exp(-v)));
      function rows(values, names) {
        return values
          .map(
            (v, i) =>
              `<div class="probrow"><span>${names[i]}</span><div class="bar" style="width:${v * 100}%"></div><b>${v.toFixed(3)}</b></div>`,
          )
          .join("");
      }
      q(".sig").innerHTML =
        "<b>Independent sigmoids</b>" + rows(sig, ["A", "B", "C", "D"]);
      q(".soft").innerHTML =
        "<b>Five-way softmax</b>" + rows(soft, ["BG", "A", "B", "C", "D"]);
      const selected = ["A", "B", "C", "D"].filter((_, i) => sig[i] >= 0.5);
      const winner = ["BG", "A", "B", "C", "D"][soft.indexOf(Math.max(...soft))];
      out.classList.add("probability-summary");
      out.innerHTML = `<div class="probability-summary-scroll"><table><caption>Current values</caption><tbody>
        <tr><th scope="row">Logits (BG, A, B, C, D)</th><td>[${z.map((v) => v.toFixed(1)).join(", ")}]</td></tr>
        <tr><th scope="row">Sigmoid total</th><td>${sig.reduce((a, b) => a + b, 0).toFixed(3)}</td></tr>
        <tr><th scope="row">Sigmoid classes ≥ 0.5</th><td>${selected.join(", ") || "None"}</td></tr>
        <tr><th scope="row">Softmax total</th><td>${soft.reduce((a, b) => a + b, 0).toFixed(3)}</td></tr>
        <tr><th scope="row">Softmax winner (argmax)</th><td>${winner}</td></tr>
      </tbody></table></div>`;
    }
    all("input").forEach((i) => (i.oninput = render));
    render();
  },
  rle(el) {
    const { q, all, out } = startLab(
      el,
      "Mask → numbered pixels → runs",
      `<p>Tap a cell to change foreground. Cell labels are <b>1-based indices</b>; coordinates use zero-based (x,y). One toy cell represents one mask pixel.</p><div class="controls"><label>Traversal<select aria-label="Traversal order"><option value="F">F: down columns</option><option value="C">C: across rows</option></select></label><button data-preset="worked">Worked mask</button><button data-preset="empty">Empty</button><button data-preset="patch">Patch</button><button data-preset="checker">Checkerboard</button><button data-preset="boundary">Boundary crossing</button></div><div class="grid"></div><div class="buffer" aria-label="Flattened mask"></div>${range("step", "Trace flattened index", 1, 24, 1)}<div class="readout"></div>`,
    );
    const H = 4,
      W = 6;
    let mask = Array(24).fill(0);
    let buttons = [];
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let b = document.createElement("button");
        b.onclick = () => {
          mask[y * W + x] ^= 1;
          render();
        };
        q(".grid").append(b);
        buttons.push(b);
      }
    const index = (x, y) => (q("select").value === "F" ? x * H + y : y * W + x);
    function render() {
      let bits = Array(24),
        step = +q("input").value;
      buttons.forEach((b, i) => {
        let x = i % W,
          y = Math.floor(i / W),
          k = index(x, y);
        bits[k] = mask[i];
        b.innerHTML = `${k + 1}<br><small>(${x},${y})</small>`;
        b.setAttribute(
          "aria-label",
          `Pixel x ${x}, y ${y}, index ${k + 1}, ${mask[i] ? "foreground" : "background"}`,
        );
        b.setAttribute("aria-pressed", !!mask[i]);
        b.classList.toggle("current", k === step - 1);
      });
      q(".buffer").innerHTML = bits
        .map(
          (b, i) =>
            `<span class="${b ? "on" : ""}" title="Index ${i + 1}">${b}</span>`,
        )
        .join("");
      let pairs = encode(bits),
        encoded = pairs.flat().join(" ");
      let decoded = Array(24).fill(0);
      pairs.forEach(([s, n]) => decoded.fill(1, s - 1, s - 1 + n));
      let coords =
        q("select").value === "F"
          ? `x=floor((${step}−1)/4)=${Math.floor((step - 1) / 4)}, y=(${step}−1)%4=${(step - 1) % 4}`
          : `x=(${step}−1)%6=${(step - 1) % 6}, y=floor((${step}−1)/6)=${Math.floor((step - 1) / 6)}`;
      q(".readout").textContent =
        `Index ${step}: ${coords}. Value=${bits[step - 1]}.`;
      out.textContent = `RLE: ${encoded || "(empty string)"}. ${pairs.length} run pairs; ${bits.reduce((a, b) => a + b, 0)} foreground pixels. Round trip=${bits.every((v, i) => v === decoded[i]) ? "PASS" : "FAIL"}. ${pairs.map(([s, n]) => `${s} ${n} → slice [${s - 1}:${s - 1 + n}]`).join("; ")}`;
    }
    all("[data-preset]").forEach(
      (b) =>
        (b.onclick = () => {
          mask.fill(0);
          let type = b.dataset.preset;
          mask = mask.map((_, i) => {
            let x = i % W,
              y = Math.floor(i / W);
            return +(type === "worked"
              ? [3, 4, 5, 6, 10, 11].includes(x * H + y + 1)
              : type === "patch"
                ? x >= 1 && x <= 3 && y >= 1 && y <= 2
                : type === "checker"
                  ? (x + y) % 2 === 0
                  : type === "boundary"
                    ? (x === 0 && y === 3) || (x === 1 && y === 0)
                    : false);
          });
          render();
        }),
    );
    q("select").onchange = render;
    q("input").oninput = render;
    q("[data-preset=worked]").click();
  },
  geometry(el) {
    const { q, all, out } = startLab(
      el,
      "Resample a synthetic binary buffer",
      range("width", "Target width", 120, 600, 240, 20) +
        range("height", "Target height", 25, 250, 240, 5) +
        range("position", "Crack left edge x", 0, 590, 101) +
        range("crack", "Native crack width", 1, 10, 2) +
        `<div class="controls"><button data-size="600,100">Native</button><button data-size="240,240">Square</button><button data-size="300,50">Isotropic half-size</button></div><p>Top: native mask. Bottom: resized target in its actual aspect ratio. Nearest-neighbor pixel-center sampling is explicit.</p><canvas data-canvas="native" role="img" aria-label="Native 600 by 100 binary crack mask"></canvas><canvas data-canvas="target" role="img" aria-label="Resampled binary crack mask"></canvas>`,
    );
    function get(k) {
      return +q(`[data-control=${k}]`).value;
    }
    function render() {
      let w = get("width"),
        h = get("height"),
        x = get("position"),
        n = get("crack");
      const source = (xx, yy) => xx >= x && xx < x + n && yy >= 12 && yy < 88;
      let target = new Uint8Array(w * h);
      for (let yy = 0; yy < h; yy++)
        for (let xx = 0; xx < w; xx++)
          target[yy * w + xx] = source(
            Math.min(599, Math.floor(((xx + 0.5) * 600) / w)),
            Math.min(99, Math.floor(((yy + 0.5) * 100) / h)),
          );
      let gt = 0,
        pred = 0,
        both = 0;
      for (let yy = 0; yy < 100; yy++)
        for (let xx = 0; xx < 600; xx++) {
          let a = source(xx, yy),
            b =
              target[
                Math.min(h - 1, Math.floor(((yy + 0.5) * h) / 100)) * w +
                  Math.min(w - 1, Math.floor(((xx + 0.5) * w) / 600))
              ];
          gt += a;
          pred += b;
          both += a && b;
        }
      function draw(key, ww, hh, value) {
        canvas(el, key, ww, hh, (c) => {
          c.fillStyle = "#18262d";
          c.fillRect(0, 0, ww, hh);
          c.fillStyle = "#ffcb4d";
          for (let yy = 0; yy < hh; yy++)
            for (let xx = 0; xx < ww; xx++)
              if (value(xx, yy)) c.fillRect(xx, yy, 1, 1);
        });
      }
      draw("native", 600, 100, source);
      draw("target", w, h, (xx, yy) => target[yy * w + xx]);
      out.textContent = `Target ${w} × ${h}. sx=${(w / 600).toFixed(3)}, sy=${(h / 100).toFixed(3)}, R=${(h / 100 / (w / 600)).toFixed(3)}. Geometric crack width=${((n * w) / 600).toFixed(3)}px. Target foreground=${target.reduce((a, b) => a + b, 0)}px. Restored foreground=${pred}px vs native ${gt}px. Round-trip Dice=${((2 * both) / (gt + pred)).toFixed(3)}. Change x by one pixel to test phase.`;
    }
    all("input").forEach((i) => (i.oninput = render));
    all("[data-size]").forEach(
      (b) =>
        (b.onclick = () => {
          let [w, h] = b.dataset.size.split(",");
          q("[data-control=width]").value = w;
          q("[data-control=height]").value = h;
          render();
        }),
    );
    render();
  },
  contrast(el) {
    const { q, all, out } = startLab(
      el,
      "Signed contrast in standard-deviation units",
      range("mi", "Inside mean", 0, 255, 110) +
        range("mo", "Outside mean", 0, 255, 100) +
        range("si", "Inside SD", 0, 60, 20) +
        range("so", "Outside SD", 0, 60, 20) +
        range("ni", "Inside sample count", 2, 1000, 100) +
        range("no", "Outside sample count", 2, 1000, 100) +
        `<canvas data-canvas="curves" role="img" aria-label="Illustrative Gaussian density curves for the selected summary statistics"></canvas><p>Curves are illustrative Gaussians, not measured image distributions; intensity axis extends to show their tails.</p>`,
    );
    function render() {
      let v = Object.fromEntries(
        all("input").map((i) => [i.dataset.control, +i.value]),
      );
      let pool = Math.sqrt(
        ((v.ni - 1) * v.si ** 2 + (v.no - 1) * v.so ** 2) / (v.ni + v.no - 2),
      );
      canvas(el, "curves", 600, 240, (c, w, h) => {
        c.fillStyle = "#fffefa";
        c.fillRect(0, 0, w, h);
        c.strokeStyle = "#82908a";
        c.beginPath();
        c.moveTo(35, 200);
        c.lineTo(575, 200);
        c.stroke();
        let max = Math.max(
          v.si ? 1 / (v.si * 2.5066) : 0,
          v.so ? 1 / (v.so * 2.5066) : 0,
          0.01,
        );
        [
          [v.mi, v.si, "#df653c"],
          [v.mo, v.so, "#087f8c"],
        ].forEach(([m, s, col]) => {
          c.strokeStyle = col;
          c.lineWidth = 3;
          c.beginPath();
          for (let i = 0; i <= 540; i++) {
            let x = -80 + (i * 415) / 540;
            let p = s ? Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * 2.5066) : 0;
            c.lineTo(i + 35, 200 - (160 * p) / max);
          }
          c.stroke();
          if (!s) {
            let x = 35 + ((m + 80) * 540) / 415;
            c.beginPath();
            c.moveTo(x, 200);
            c.lineTo(x, 40);
            c.stroke();
          }
        });
        c.fillStyle = "#172d3a";
        c.font = "12px monospace";
        [-50, 0, 100, 200, 300].forEach((t) =>
          c.fillText(t, 35 + ((t + 80) * 540) / 415, 223),
        );
        c.fillStyle = "#df653c";
        c.fillText("Inside", 40, 22);
        c.fillStyle = "#087f8c";
        c.fillText("Outside", 135, 22);
      });
      out.textContent = pool
        ? `Pooled SD=${pool.toFixed(3)}. Mean difference=${v.mi - v.mo}. d=${((v.mi - v.mo) / pool).toFixed(3)}. Positive means brighter, negative darker. Conventional effect-size labels describe mean separation, not segmentation success.`
        : "Pooled SD=0: Cohen’s d is undefined. No arbitrary epsilon is substituted.";
    }
    all("input").forEach((i) => (i.oninput = render));
    render();
  },
  texture(el) {
    const { q, out } = startLab(
      el,
      "Identical histogram; different spatial evidence",
      range("shuffle", "Amount of deterministic pixel swapping", 0, 100, 100) +
        `<div class="twocol"><div><p>Ordered stripes</p><canvas data-canvas="ordered" role="img" aria-label="Ordered two-intensity striped texture"></canvas></div><div><p>Swapped arrangement</p><canvas data-canvas="shuffled" role="img" aria-label="Deterministically shuffled texture with the same intensities"></canvas></div></div>`,
    );
    const W = 64,
      H = 32,
      orig = Array.from({ length: W * H }, (_, i) =>
        Math.floor(i / W) % 4 < 2 ? 75 : 175,
      );
    function render() {
      let vals = orig.slice(),
        rng = seedRng(42),
        steps = Math.round(+q("input").value * 30);
      for (let i = 0; i < steps; i++) {
        let a = Math.floor(rng() * vals.length),
          b = Math.floor(rng() * vals.length);
        [vals[a], vals[b]] = [vals[b], vals[a]];
      }
      function edges(v) {
        let s = 0;
        for (let y = 0; y < H; y++)
          for (let x = 0; x < W - 1; x++)
            s += Math.abs(v[y * W + x] - v[y * W + x + 1]);
        return s / (H * (W - 1));
      }
      for (let [key, v] of [
        ["ordered", orig],
        ["shuffled", vals],
      ])
        canvas(el, key, 512, 256, (c) => {
          for (let y = 0; y < H; y++)
            for (let x = 0; x < W; x++) {
              let z = v[y * W + x];
              c.fillStyle = `rgb(${z},${z},${z})`;
              c.fillRect(x * 8, y * 8, 8, 8);
            }
        });
      out.textContent = `Both contain 1,024 pixels at 75 and 1,024 at 175. Mean=125; population SD=50; mean difference d=0. Mean horizontal neighbor difference: ordered=${edges(orig).toFixed(2)}, swapped=${edges(vals).toFixed(2)}. A spatial feature responds while a histogram cannot.`;
    }
    q("input").oninput = render;
    render();
  },
};
document.querySelectorAll("[data-lab]").forEach((el) => {
  if (labs[el.dataset.lab]) labs[el.dataset.lab](el);
});
