# Ensemble diversity · results and comparisons

Models are useful to an ensemble when they make different errors. Different architectures, seeds, or input scales can all change which errors a model makes. These anonymized examples show how to check whether that variation improves the combined predictions.

<style>
.ensemble-chart { margin: 0 0 1.5rem; }
.ensemble-chart > p { color: var(--muted); font-size: .875rem; margin-bottom: .5rem; }
.ensemble-chart svg { display: block; width: 100%; max-width: 600px; height: auto; overflow: visible; }
.ensemble-chart svg text { fill: var(--ink); font: 13.5px system-ui, sans-serif; font-variant-numeric: tabular-nums; }
.ensemble-chart svg .value { font-weight: 700; }
.ensemble-chart .grid { stroke: var(--line); fill: none; }
.ensemble-chart .first { --series-color: var(--accent); }
.ensemble-chart .second { --series-color: var(--flower); }
.ensemble-chart polyline { fill: none; stroke: var(--series-color); stroke-width: 2.7; stroke-linecap: round; stroke-linejoin: round; }
.ensemble-chart circle { fill: var(--series-color); }
.ensemble-chart .second polyline { stroke-dasharray: 7 4; }
.ensemble-chart .legend { display: flex; flex-wrap: wrap; gap: .25rem 1rem; list-style: none; padding: 0; margin: 0 0 .4rem; font-size: .8rem; }
.ensemble-chart .legend li::before { content: ""; display: inline-block; width: 1.15rem; margin: 0 .4rem .2rem 0; border-top: 2px solid var(--series-color); }
.ensemble-chart .legend .second::before { border-top-style: dashed; }
.ensemble-chart figcaption { color: var(--muted); font-size: .875rem; }
</style>

## A direct check on held-out images

All four blends used predictions from the same 791 held-out images and the same mask decoder. The two-model blends used equal weights.

| Blend | Local validation score |
| --- | ---: |
| Family A medium + Family A large | **0.76819** |
| Family A medium + Family B | 0.75133 |
| Family A large + Family B | 0.75423 |
| Family A medium + Family A large + Family B (40/40/20) | 0.76741 |

The two Family A variants worked best together on this fold. Adding Family B to both gave a near tie. Architecture names alone tell you little about whether the models correct each other’s errors.

In a separate Kaggle experiment, blending predictions from two different families into a fixed ensemble raised the score from **0.76402 to 0.76830**, with the set of predicted defect presences held fixed. The gain came from changing the submitted masks on those detections.

## How the mixed ensemble improved

<div class="ensemble-chart">
<p>Kaggle public score · Selected milestones, 14–23 Nov</p>
<figure><svg viewBox="0 0 510 190" role="img" aria-label="Mixed ensemble: 14 Nov, 0.74591; 15 Nov, 0.75604; 16 Nov, 0.75641; 17 Nov, 0.76250; 20 Nov, 0.76402; 21 Nov, 0.77199; 22 Nov, 0.77303; 23 Nov, 0.77353">
<path class="grid" d="M64 155H492"/>
<text x="57.0" y="159.0" text-anchor="end">0.740</text>
<path class="grid" d="M64 122H492"/>
<text x="57.0" y="126.0" text-anchor="end">0.750</text>
<path class="grid" d="M64 89H492"/>
<text x="57.0" y="93.0" text-anchor="end">0.760</text>
<path class="grid" d="M64 56H492"/>
<text x="57.0" y="60.0" text-anchor="end">0.770</text>
<path class="grid" d="M64 23H492"/>
<text x="57.0" y="27.0" text-anchor="end">0.780</text>
<text x="82.1" y="181.0" text-anchor="middle">14 Nov</text>
<text x="203.3" y="181.0" text-anchor="middle">17 Nov</text>
<text x="324.5" y="181.0" text-anchor="middle">20 Nov</text>
<text x="445.6" y="181.0" text-anchor="middle">23 Nov</text>
<g class="first">
<polyline points="82.11764705882356,135.49700000000013 122.50980392156866,102.06799999999987 162.90196078431376,100.84699999999992 203.29411764705884,80.75000000000017 324.47058823529414,75.73399999999994 364.8627450980393,49.433000000000206 405.25490196078437,46.00100000000009 445.64705882352945,44.3509999999999"/>
<circle cx="82.11764705882356" cy="135.49700000000013" r="4.5"><title>14 Nov · 0.74591</title></circle>
<circle cx="122.50980392156866" cy="102.06799999999987" r="4.5"><title>15 Nov · 0.75604</title></circle>
<circle cx="162.90196078431376" cy="100.84699999999992" r="4.5"><title>16 Nov · 0.75641</title></circle>
<circle cx="203.29411764705884" cy="80.75000000000017" r="4.5"><title>17 Nov · 0.76250</title></circle>
<circle cx="324.47058823529414" cy="75.73399999999994" r="4.5"><title>20 Nov · 0.76402</title></circle>
<circle cx="364.8627450980393" cy="49.433000000000206" r="4.5"><title>21 Nov · 0.77199</title></circle>
<circle cx="405.25490196078437" cy="46.00100000000009" r="4.5"><title>22 Nov · 0.77303</title></circle>
<circle cx="445.64705882352945" cy="44.3509999999999" r="4.5"><title>23 Nov · 0.77353</title></circle>
</g>
<text x="91.1" y="151.5" class="value" text-anchor="start">0.74591</text>
<text x="450.6" y="32.4" class="value" text-anchor="middle">0.77353</text>
</svg>
<figcaption>The selected milestones gained 0.02762. Model mix, training and mask decoding changed over this period.</figcaption>
</figure>
</div>

## Standalone checkpoints versus an ensemble

<div class="ensemble-chart">
<p>Kaggle public score · Best recorded in each group on the two dates</p>
<ul class="legend">
<li class="first">Mixed ensemble</li>
<li class="second">Standalone checkpoint</li>
</ul><figure><svg viewBox="0 0 510 190" role="img" aria-label="Mixed ensemble: 20 Nov, 0.76402; 21 Nov, 0.77199. Standalone checkpoint: 20 Nov, 0.73307; 21 Nov, 0.76178">
<path class="grid" d="M64 155H492"/>
<text x="57.0" y="159.0" text-anchor="end">0.730</text>
<path class="grid" d="M64 111H492"/>
<text x="57.0" y="115.0" text-anchor="end">0.745</text>
<path class="grid" d="M64 67H492"/>
<text x="57.0" y="71.0" text-anchor="end">0.760</text>
<path class="grid" d="M64 23H492"/>
<text x="57.0" y="27.0" text-anchor="end">0.775</text>
<text x="126.7" y="181.0" text-anchor="middle">20 Nov</text>
<text x="425.3" y="181.0" text-anchor="middle">21 Nov</text>
<g class="first">
<polyline points="126.72463768115969,55.20799999999994 425.2753623188403,31.829333333333523"/>
<circle cx="126.72463768115969" cy="55.20799999999994" r="4.5"><title>20 Nov · 0.76402</title></circle>
<circle cx="425.2753623188403" cy="31.829333333333523" r="4.5"><title>21 Nov · 0.77199</title></circle>
</g>
<g class="second">
<polyline points="126.72463768115969,145.99466666666663 425.2753623188403,61.77866666666665"/>
<circle cx="126.72463768115969" cy="145.99466666666663" r="4.5"><title>20 Nov · 0.73307</title></circle>
<circle cx="425.2753623188403" cy="61.77866666666665" r="4.5"><title>21 Nov · 0.76178</title></circle>
</g>
<text x="126.7" y="45.2" class="value" text-anchor="middle">0.76402</text>
<text x="425.3" y="21.8" class="value" text-anchor="middle">0.77199</text>
<text x="126.7" y="136.0" class="value" text-anchor="middle">0.73307</text>
<text x="425.3" y="51.8" class="value" text-anchor="middle">0.76178</text>
</svg>
<figcaption>The standalone checkpoints differed between dates. These results compare recorded submissions, with different recipes in each group.</figcaption>
</figure>
</div>

## What happened when we added folds

<div class="ensemble-chart">
<p>Kaggle public score · Same family mix across two fold-coverage recipes</p>
<figure><svg viewBox="0 0 510 190" role="img" aria-label="Mixed ensemble: 4 folds, 0.75491; 5 folds, 0.75491">
<path class="grid" d="M64 155H492"/>
<text x="57.0" y="159.0" text-anchor="end">0.750</text>
<path class="grid" d="M64 111H492"/>
<text x="57.0" y="115.0" text-anchor="end">0.753</text>
<path class="grid" d="M64 67H492"/>
<text x="57.0" y="71.0" text-anchor="end">0.756</text>
<path class="grid" d="M64 23H492"/>
<text x="57.0" y="27.0" text-anchor="end">0.759</text>
<text x="132.9" y="181.0" text-anchor="middle">4-fold recipe</text>
<text x="419.1" y="181.0" text-anchor="middle">5-fold recipe</text>
<g class="first">
<polyline points="132.9444444444445,82.98666666666718 419.05555555555566,82.98666666666718"/>
<circle cx="132.9444444444445" cy="82.98666666666718" r="4.5"><title>4 folds · 0.75491</title></circle>
<circle cx="419.05555555555566" cy="82.98666666666718" r="4.5"><title>5 folds · 0.75491</title></circle>
</g>
<text x="132.9" y="72.0" class="value" text-anchor="middle">0.75491</text>
<text x="419.1" y="72.0" class="value" text-anchor="middle">0.75491</text>
</svg>
<figcaption>Both recipes scored 0.75491 at Kaggle’s reported precision. The five-fold version also expanded coverage unevenly across families.</figcaption>
</figure>
</div>

## How the blend weights affected the score

<div class="ensemble-chart">
<p>Kaggle public score · Weight assigned to the newer models when deciding whether a defect is present</p>
<figure><svg viewBox="0 0 510 190" role="img" aria-label="Mixed ensemble blend: 62%, 0.77267; 68%, 0.77353; 72%, 0.77322; 75%, 0.77303; 85%, 0.77258; 100%, 0.77040">
<path class="grid" d="M64 155H492"/>
<text x="57.0" y="159.0" text-anchor="end">0.770</text>
<path class="grid" d="M64 122H492"/>
<text x="57.0" y="126.0" text-anchor="end">0.771</text>
<path class="grid" d="M64 89H492"/>
<text x="57.0" y="93.0" text-anchor="end">0.772</text>
<path class="grid" d="M64 56H492"/>
<text x="57.0" y="60.0" text-anchor="end">0.773</text>
<path class="grid" d="M64 23H492"/>
<text x="57.0" y="27.0" text-anchor="end">0.774</text>
<text x="89.2" y="181.0" text-anchor="middle">62%</text>
<text x="146.7" y="181.0" text-anchor="middle">68%</text>
<text x="213.7" y="181.0" text-anchor="middle">75%</text>
<text x="309.5" y="181.0" text-anchor="middle">85%</text>
<text x="453.3" y="181.0" text-anchor="middle">100%</text>
<g class="first">
<polyline points="89.16279069767441,66.89000000000172 146.65116279069767,38.50999999999901 184.97674418604652,48.740000000000066 213.72093023255815,55.010000000000844 309.5348837209302,69.85999999999919 453.25581395348837,141.80000000000146"/>
<circle cx="89.16279069767441" cy="66.89000000000172" r="4.5"><title>62% · 0.77267</title></circle>
<circle cx="146.65116279069767" cy="38.50999999999901" r="4.5"><title>68% · 0.77353</title></circle>
<circle cx="184.97674418604652" cy="48.740000000000066" r="4.5"><title>72% · 0.77322</title></circle>
<circle cx="213.72093023255815" cy="55.010000000000844" r="4.5"><title>75% · 0.77303</title></circle>
<circle cx="309.5348837209302" cy="69.85999999999919" r="4.5"><title>85% · 0.77258</title></circle>
<circle cx="453.25581395348837" cy="141.80000000000146" r="4.5"><title>100% · 0.77040</title></circle>
</g>
<text x="164.7" y="29.5" class="value" text-anchor="middle">0.77353</text>
<text x="453.3" y="156.8" class="value" text-anchor="end">0.77040</text>
</svg>
<figcaption>The best tested recipe scored 0.77353 at 68%. Mask weights also changed, so the curve compares complete recipes.</figcaption>
</figure>
</div>

## What happened when we increased resolution

<div class="ensemble-chart">
<p>Local validation score proxy · The same 791 held-out images</p>
<ul class="legend">
<li class="first">Family A · Medium</li>
<li class="second">Family A · Large</li>
</ul><figure><svg viewBox="0 0 510 190" role="img" aria-label="Family A · Medium: 2×, 0.74462; 3×, 0.72468. Family A · Large: 2×, 0.74356; 3×, 0.71168">
<path class="grid" d="M64 155H492"/>
<text x="57.0" y="159.0" text-anchor="end">0.710</text>
<path class="grid" d="M64 122H492"/>
<text x="57.0" y="126.0" text-anchor="end">0.720</text>
<path class="grid" d="M64 89H492"/>
<text x="57.0" y="93.0" text-anchor="end">0.730</text>
<path class="grid" d="M64 56H492"/>
<text x="57.0" y="60.0" text-anchor="end">0.740</text>
<path class="grid" d="M64 23H492"/>
<text x="57.0" y="27.0" text-anchor="end">0.750</text>
<text x="132.9" y="181.0" text-anchor="middle">2× input</text>
<text x="419.1" y="181.0" text-anchor="middle">3× input</text>
<g class="first">
<polyline points="132.94444444444443,40.74656187029669 419.0555555555555,106.55901202909929"/>
<circle cx="132.94444444444443" cy="40.74656187029669" r="4.5"><title>2× · 0.74462</title></circle>
<circle cx="419.0555555555555" cy="106.55901202909929" r="4.5"><title>3× · 0.72468</title></circle>
</g>
<g class="second">
<polyline points="132.94444444444443,44.2586329427138 419.0555555555555,149.44755108549964"/>
<circle cx="132.94444444444443" cy="44.2586329427138" r="4.5"><title>2× · 0.74356</title></circle>
<circle cx="419.0555555555555" cy="149.44755108549964" r="4.5"><title>3× · 0.71168</title></circle>
</g>
<text x="419.1" y="96.6" class="value" text-anchor="middle">0.72468</text>
<text x="419.1" y="139.4" class="value" text-anchor="middle">0.71168</text>
</svg>
<figcaption>Both variants scored lower at 3× than at 2×. The two sizes were evaluated with the same decoder on fold 0.</figcaption>
</figure>
</div>

## How to test whether a new member helps

1. Compare two blends with the same number of checkpoints. Start with two seeds of one model, then replace one checkpoint with a model from another family.
2. Use the same held-out images, prediction averaging, and mask decoder. Compare the score and review where misses and false positives overlap.
3. Repeat on another fold before choosing a recipe. The four-fold versus five-fold chart compares training splits. A separate seed-only comparison is needed to measure the effect of changing seeds.

[Back to the Session 6 overview](README.md)
