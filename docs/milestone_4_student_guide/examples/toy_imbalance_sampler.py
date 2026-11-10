"""Exact combination-group sampler arithmetic on fictional image labels."""
from __future__ import annotations

from collections import Counter
import math

import torch
from torch.utils.data import WeightedRandomSampler


def per_image_weights(groups: list[str], cap: float = 5.0) -> list[float]:
    counts = Counter(groups)
    if not counts.get("clean"):
        raise ValueError("the fictional pool needs a clean reference group")
    n_clean = counts["clean"]
    return [min(math.sqrt(n_clean / counts[group]), cap) for group in groups]


def main() -> None:
    # A fictional combination table. Labels can overlap in real data; each
    # unique combination is one group for this sampler calculation.
    groups = ["clean"] * 25 + ["defect_a"] * 16 + ["defect_b"] * 4 + ["defect_a+defect_b"] * 5
    weights = per_image_weights(groups)
    counts = Counter(groups)
    group_weights = {group: weights[groups.index(group)] for group in counts}
    total_mass = sum(counts[group] * group_weights[group] for group in counts)

    print("Fictional image counts and expected draws in a 500-draw epoch")
    print(f"{'group':<20} {'n':>5} {'per-image w':>13} {'draw share':>12} {'expected':>10}")
    for group, n in counts.items():
        w = group_weights[group]
        share = n * w / total_mass
        print(f"{group:<20} {n:>5} {w:>13.3f} {share:>11.1%} {500 * share:>10.2f}")

    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=500,
        replacement=True,
        generator=torch.Generator().manual_seed(23),
    )
    sampled = Counter(groups[index] for index in sampler)
    b_positive_expected = sum(
        500 * counts[g] * group_weights[g] / total_mass
        for g in counts
        if "defect_b" in g
    )
    print(f"\nOne seeded draw: {dict(sampled)}")
    print(f"Expected defect_b-positive draws (both B-containing groups): {b_positive_expected:.2f}/500")
    print("These are toy expectations; sampling with replacement varies from one epoch to the next.")


if __name__ == "__main__":
    main()
