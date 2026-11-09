# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6"]
# ///
# %% [markdown]
# # Inference lab: TTA, EMA and mask decisions
# Deterministic synthetic mechanics, CPU only. No trained model or score claims.
# Run cells in order. Optional setup in a fresh notebook: `%pip install numpy==2.2.6`.
# %%
import numpy as np

np.set_printoptions(precision=4, suppress=True)
H, W = 8, 50  # Same 6.25:1 aspect ratio as H128 x W800.
x = np.zeros((1, 4, H, W), dtype=np.float32)
x[0, 0, 1:4, 6:10] = 1
x[0, 1, 5, 37] = 1
x[0, 2, 2:6, 28:30] = 1
# Channel 3 intentionally empty. These are synthetic markers, not real images.
sigmoid = lambda z: 1 / (1 + np.exp(-z))
print('Synthetic tensor [B,C,H,W]:', x.shape)

# %% [markdown]
# ## A · TTA alignment
# Predict: where does column 6 move after a width-50 flip?
# The toy predictor is pixelwise and perfectly equivariant.
# Aligned TTA therefore equals the original prediction exactly.
# %%
views = ((), (-1,), (-2,), (-2, -1))

def flip(a, axes):
    return np.flip(a, axis=axes) if axes else a

def toy_predict(a):
    return sigmoid(6 * a - 3)

def tta(a, predictor, invert=True):
    total = np.zeros_like(a, dtype=np.float32)
    for axes in views:
        p = predictor(flip(a, axes))
        total += flip(p, axes) if invert else p
    return total / len(views)

for axes in views:
    assert np.array_equal(flip(flip(x, axes), axes), x)
assert np.array_equal(flip(x, (-1,))[0, 0, :, 40:44], x[0, 0, :, 6:10][:, ::-1])
base = toy_predict(x)
aligned = tta(x, toy_predict)
broken = tta(x, toy_predict, invert=False)
assert np.allclose(aligned, base)
assert not np.allclose(broken, base)
print('Column 6 becomes:', W - 1 - 6)
print('Max aligned error:', float(abs(aligned - base).max()))
print('True-marker confidence, aligned / broken:', aligned[0, 0, 1, 6], broken[0, 0, 1, 6])
print('Mirror-location confidence, aligned / broken:', aligned[0, 0, 1, 43], broken[0, 0, 1, 43])

# %%
# A deterministic positional bias breaks exact equivariance.
# TTA changes this output; better quality would require held-out evidence.
def biased_predict(a):
    position_bias = np.linspace(-1, 1, W, dtype=np.float32)[None, None, None, :]
    return sigmoid(6 * a - 3 + position_bias)

biased = biased_predict(x)
corrected = tta(x, biased_predict)
print('Toy positional-bias output change:', float(abs(corrected - biased).max()))
logits = np.array([-4., -4., 6.])
print('Mean probability:', sigmoid(logits).mean())
print('Sigmoid(mean logit):', sigmoid(logits.mean()))

# %% [markdown]
# ## B · EMA and ensembles
# Predict the three updates for beta=0.5 before execution.
# This scalar arithmetic represents parameter updates, not a training benchmark.
# %%
def ema_trace(values, beta=0.5, initial=0.):
    if not 0 <= beta < 1:
        raise ValueError('EMA decay must be in [0,1)')
    state = initial
    result = []
    for value in values:
        state = beta * state + (1 - beta) * value
        result.append(state)
    return np.array(result)

trace = ema_trace([2., 4., 6.])
assert np.allclose(trace, [1., 2.5, 4.25])
print('EMA beta .5:', trace)
print('EMA beta .9:', ema_trace([2., 4., 6.], beta=.9))
# A nonlinear toy network demonstrates non-equivalence of the averages.
weights = np.array([-2., 2.])
parameter_average_prediction = np.maximum(0, weights.mean())
prediction_average = np.maximum(0, weights).mean()
assert parameter_average_prediction != prediction_average
print('ReLU(mean parameter) / mean ReLU predictions:', parameter_average_prediction, prediction_average)
member_probabilities = np.array([.2, .8])
blend_weights = np.array([.25, .75])
blend = np.dot(blend_weights, member_probabilities)
assert np.isclose(blend, .65)
print('Probability blend:', blend, 'Decision at .5:', bool(blend >= .5))

# %% [markdown]
# ## C · Threshold and area decisions
# Four pixels provide a fully inspectable example.
# The best threshold here is not a competition recommendation.
# %%
def positive_dice(pred, target):
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator if denominator else None), tp, fp, fn

p = np.array([.8, .6, .4, .2])
y = np.array([1, 1, 0, 0], dtype=bool)
for threshold in [.3, .5, .7]:
    pred = p >= threshold
    print('Threshold', threshold, 'mask', pred.astype(int), '(Dice,TP,FP,FN)', positive_dice(pred, y))
assert np.isclose(positive_dice(p >= .5, y)[0], 1.)

# %%
def filter_components(mask, minimum_area, connectivity=8):
    """Small flood-fill reference, not a fast production implementation."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or connectivity not in (4, 8) or minimum_area < 0:
        raise ValueError('Expected a 2D mask, connectivity 4/8 and nonnegative area')
    seen = np.zeros_like(mask)
    result = np.zeros_like(mask)
    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
               if (dr or dc) and (connectivity == 8 or abs(dr) + abs(dc) == 1)]
    for row, col in zip(*np.where(mask & ~seen)):
        if seen[row, col]:
            continue
        pending, component = [(int(row), int(col))], []
        seen[row, col] = True
        while pending:
            r, c = pending.pop()
            component.append((r, c))
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    pending.append((nr, nc))
        if len(component) >= minimum_area:
            for r, c in component:
                result[r, c] = True
    return result

mask = np.zeros((H, W), dtype=bool)
mask[1, 1:3] = True  # A genuine small target, erased by area 3.
mask[5, 30:35] = True
filtered = filter_components(mask, 3)
assert mask.sum() == 7 and filtered.sum() == 5
print('Before / after area 3:', int(mask.sum()), int(filtered.sum()))
diagonal = np.eye(3, dtype=bool)
assert filter_components(diagonal, 2, 4).sum() == 0
assert filter_components(diagonal, 2, 8).sum() == 3
print('Diagonal pixels kept, 4 / 8 connectivity:', filter_components(diagonal, 2, 4).sum(), filter_components(diagonal, 2, 8).sum())

# %% [markdown]
# ## D · Metrics and RLE
# Diagnostics have explicit denominators and are not the official scorer.
# Production submissions must follow the actual sample submission rows.
# %%
def diagnostics(pred, target):
    pred, target = np.asarray(pred, dtype=bool), np.asarray(target, dtype=bool)
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError('Expected matching [B,C,H,W] masks')
    positives, empty_correct, empty_count = [], 0, 0
    for pm, tm in zip(pred.reshape(-1, *pred.shape[-2:]), target.reshape(-1, *target.shape[-2:])):
        if tm.any():
            positives.append(positive_dice(pm, tm)[0])
        else:
            empty_count += 1
            empty_correct += int(not pm.any())
    return dict(positive_pair_dice=float(np.mean(positives)) if positives else None,
                positive_pairs=len(positives), empty_pairs=empty_count,
                empty_pair_specificity=empty_correct / empty_count if empty_count else None)

print('Synthetic diagnostics:', diagnostics(aligned >= .5, x > 0))
assert diagnostics(aligned >= .5, x > 0)['positive_pairs'] == 3
assert diagnostics(np.zeros_like(x), np.zeros_like(x))['positive_pair_dice'] is None
assert diagnostics(np.ones_like(x), np.ones_like(x))['empty_pair_specificity'] is None

# %%
def encode_rle(mask):
    flat = np.asarray(mask, dtype=np.uint8).flatten(order='F')
    changes = np.flatnonzero(np.diff(np.r_[0, flat, 0])) + 1
    changes[1::2] -= changes[::2]
    return ' '.join(map(str, changes)) if changes.size else ' '

def decode_rle(encoded, shape):
    flat = np.zeros(int(np.prod(shape)), dtype=np.uint8)
    if not encoded.strip():
        return flat.reshape(shape, order='F')
    tokens = np.array([int(v) for v in encoded.split()])
    if len(tokens) % 2:
        raise ValueError('Odd RLE token count')
    previous_end = 0
    for start, length in tokens.reshape(-1, 2):
        if start < 1 or length < 1 or start - 1 < previous_end or start - 1 + length > len(flat):
            raise ValueError('Invalid RLE bounds or overlap')
        flat[start - 1:start - 1 + length] = 1
        previous_end = start - 1 + length
    return flat.reshape(shape, order='F')

toy_mask = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)
assert encode_rle(toy_mask) == '2 3'
for sample in [toy_mask, np.zeros((H, W), dtype=np.uint8), np.ones((H, W), dtype=np.uint8), (x[0, 0] > 0).astype(np.uint8)]:
    assert np.array_equal(sample, decode_rle(encode_rle(sample), sample.shape))
assert encode_rle(np.zeros((H, W))) == ' '
print('Toy RLE:', repr(encode_rle(toy_mask)), 'Empty sentinel:', repr(encode_rle(np.zeros((H, W)))))
print('All mechanics checks passed. No real-model quality claim follows from these checks.')

# %% [markdown]
# ## Exit ticket
# 1. Explain the inverse-transform location in TTA.
# 2. Explain why EMA uses one evaluation per view.
# 3. Explain how an area cutoff can remove a genuine defect.
# 4. Describe a leakage-free selection and confirmation protocol.
