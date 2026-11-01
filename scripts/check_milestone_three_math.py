# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.8.0"]
# ///
"""CPU checks of the assessment loss and corrected teaching claims.

Run: uv run --no-project scripts/check_milestone_three_math.py
No competition data, pretrained downloads, training benchmarks or answer keys.
"""
from pathlib import Path
import json
import re

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def check():
    # Execute the actual assessment loss reference, not a duplicate implementation.
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "MILESTONE_3.md").read_text(encoding="utf-8"), re.S)
    scope = {"torch": torch}
    exec(next(code for code in blocks if "class SoftDiceLoss" in code), scope)
    criterion = scope["SoftDiceLoss"]()
    compound = scope["CompoundLoss"]()

    z = torch.tensor([-1.7, 0.0, 2.4], dtype=torch.float64, requires_grad=True)
    y = torch.zeros_like(z)
    loss = criterion(z, y)
    loss.backward()
    p = z.detach().sigmoid()
    expected = p * (1 - p) / (p.sum() + 1)**2
    torch.testing.assert_close(z.grad, expected)
    assert torch.autograd.gradcheck(lambda a: criterion(a, y), (z.detach().requires_grad_(),))

    # Large total mass and sigmoid saturation can suppress the logit signal.
    for value in (0.0, 12.0):
        a = torch.full((20,), value, dtype=torch.float64, requires_grad=True)
        criterion(a, torch.zeros_like(a)).backward()
        if value == 0.0:
            reference = a.grad.clone()
        else:
            assert torch.all(a.grad < reference)

    # A positive elsewhere in the reduction invalidates the all-empty derivative.
    mixed_z = torch.zeros((2, 4), dtype=torch.float64, requires_grad=True)
    mixed_y = torch.zeros_like(mixed_z)
    mixed_y[1, 0] = 1
    criterion(mixed_z, mixed_y).backward()
    q = mixed_z.detach().sigmoid()
    all_empty_formula = q * (1 - q) / (q.sum() + 1)**2
    assert not torch.allclose(mixed_z.grad[0], all_empty_formula[0])

    # Non-contiguous target tensors should obey the same declared flattening rule.
    a = torch.randn(2, 4, 3, 5, dtype=torch.float64).transpose(-1, -2)
    b = torch.zeros_like(a)
    torch.testing.assert_close(criterion(a, b), criterion(a.contiguous(), b.contiguous()))
    torch.testing.assert_close(compound(a, b), 0.5 * nn.functional.binary_cross_entropy_with_logits(a, b) + 0.5 * criterion(a, b))

    # Reproduce Guide 5's distinct practice patch, not Q5's assessment population.
    probs = torch.full((20,), 0.2, dtype=torch.float64)
    target = torch.zeros_like(probs)
    target[:4] = 1
    torch.testing.assert_close(criterion(probs.logit(), target), torch.tensor(1 - 2.6/9, dtype=torch.float64))

    # Guide 1's three-stage practice trace and operator distinction.
    image = torch.zeros(1, 1, 96, 643)
    widths = [image.shape[-1]]
    for _ in range(3):
        image = nn.functional.max_pool2d(image, 2)
        widths.append(image.shape[-1])
    assert widths == [643, 321, 160, 80]
    assert widths[-1] * 2 == widths[-2]
    assert widths[-2] * 2 != widths[-3]
    assert nn.functional.max_pool2d(torch.zeros(1, 1, 8, 101), 2).shape[-1] == 50
    assert nn.Conv2d(1, 1, 3, stride=2, padding=1)(torch.zeros(1, 1, 8, 101)).shape[-1] == 51

    # A nonzero copied bias is not divided when channel weights are averaged.
    torch.manual_seed(42)
    rgb = nn.Conv2d(3, 4, 3, bias=True).double()
    gray = nn.Conv2d(1, 4, 3, bias=True).double()
    x = torch.randn(2, 1, 10, 14, dtype=torch.float64)
    with torch.no_grad():
        rgb.bias.fill_(0.7)
        gray.bias.copy_(rgb.bias)
        gray.weight.copy_(rgb.weight.sum(1, keepdim=True))
        repeated = rgb(x.repeat(1, 3, 1, 1))
        torch.testing.assert_close(gray(x), repeated)
        gray.weight.copy_(rgb.weight.mean(1, keepdim=True))
        bias = rgb.bias.reshape(1, -1, 1, 1)
        torch.testing.assert_close(gray(x), (repeated - bias)/3 + bias)
        assert not torch.allclose(gray(x), repeated/3)

    bn = nn.BatchNorm2d(1)
    bn.requires_grad_(False)
    before = bn.running_mean.clone()
    bn(torch.ones(2, 1, 4, 4))
    assert not torch.equal(before, bn.running_mean)
    bn.eval()
    before = bn.running_mean.clone()
    bn(torch.ones(2, 1, 4, 4) * 2)
    assert torch.equal(before, bn.running_mean)

    # Keep every runnable notebook cell paired with the percent-cell Python file.
    folder = ROOT / "docs/milestone_3_student_guide/downloads"
    script = (folder / "synthetic_transfer_learning_walkthrough.py").read_text(encoding="utf-8")
    paired = []
    for metadata, cell in re.findall(r"^# %%([^\n]*)\n(.*?)(?=^# %%|\Z)", script, flags=re.M | re.S):
        if "[markdown]" not in metadata:
            paired.append(cell.strip())
    notebook = json.loads((folder / "synthetic_transfer_learning_walkthrough.ipynb").read_text(encoding="utf-8"))
    actual = ["".join(c["source"]).strip() for c in notebook["cells"] if c["cell_type"] == "code"]
    assert paired == actual, "Notebook code and paired script diverged"
    assert all(not c.get("outputs") for c in notebook["cells"] if c["cell_type"] == "code")
    print("PASS: loss/autograd, reduction, practice arithmetic, pooling, bias, BatchNorm and notebook parity")


if __name__ == "__main__":
    check()
