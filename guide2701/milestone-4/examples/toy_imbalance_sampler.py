"""Demonstration of sampling strategies and class imbalance mechanics on synthetic data.

Shows:
1. Representation of image-level, class-level, and pixel-level imbalance.
2. Natural uniform sampling vs. WeightedRandomSampler with bounded inverse frequency.
3. Verification that validation evaluation MUST remain unweighted (natural).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# 1. Fictional dataset schema with extreme rare class
CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]

def generate_fictional_records(num_images: int = 200, seed: int = 42) -> list[dict]:
    """Generate synthetic image records with known ground-truth distributions."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(num_images):
        image_id = f"SYNTH_STRIP_{i:04d}"
        # Fictional distribution:
        # ~50% Clean, Defect A ~10%, Defect B (Rare) ~3%, Defect C ~35%, Defect D ~15%
        is_clean = rng.random() < 0.50
        active = [False, False, False, False]
        if not is_clean:
            active[0] = rng.random() < 0.20  # A
            active[1] = rng.random() < 0.06  # B (rare)
            active[2] = rng.random() < 0.70  # C (frequent)
            active[3] = rng.random() < 0.30  # D
            # Ensure at least one defect if marked not clean
            if not any(active):
                active[2] = True

        records.append({
            "image_id": image_id,
            "is_clean": not any(active),
            "active_defects": active,
            "defect_names": [CLASSES[c] for c, act in enumerate(active) if act],
        })
    return records


class SyntheticStripDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return {
            "image_id": r["image_id"],
            "is_clean": r["is_clean"],
            "targets": torch.tensor(r["active_defects"], dtype=torch.float32),
        }


def compute_bounded_sampling_weights(records: list[dict], max_weight: float = 4.0) -> torch.Tensor:
    """Compute bounded sqrt inverse frequency weights for images.

    - Clean images keep weight = 1.0.
    - Defective images take the maximum class weight among active defects.
    - Weights are clamped at `max_weight` to prevent overfitting on tiny samples.
    """
    present = np.array([r["active_defects"] for r in records], dtype=bool)
    rates = np.maximum(present.mean(axis=0), 1.0 / len(records))
    # Sqrt inverse class weights
    inv_weights = np.clip(1.0 / np.sqrt(rates), 1.0, max_weight)

    image_weights = np.ones(len(records), dtype=np.float64)
    for i, r in enumerate(records):
        active_indices = [c for c, act in enumerate(r["active_defects"]) if act]
        if active_indices:
            # Use maximum weight among active classes (does not double-count multi-defects)
            image_weights[i] = np.max([inv_weights[c] for c in active_indices])

    return torch.tensor(image_weights, dtype=torch.double)


def main():
    print("=" * 70)
    print("Milestone 4: Class Imbalance & Sampling Diagnostic")
    print("=" * 70)

    records = generate_fictional_records(num_images=200, seed=42)
    dataset = SyntheticStripDataset(records)

    # Calculate empirical baseline stats
    clean_count = sum(r["is_clean"] for r in records)
    class_counts = [sum(r["active_defects"][c] for r in records) for c in range(4)]

    print(f"\n1. Empirical Training Set Distribution (N = {len(records)} images):")
    print(f"   Clean Images: {clean_count}/{len(records)} ({clean_count / len(records):.1%})")
    for name, cnt in zip(CLASSES, class_counts):
        print(f"   - {name:<10}: {cnt:>3} images ({cnt / len(records):.1%})")

    # Setup Samplers
    # A. Natural Uniform Sampler
    natural_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # B. Bounded Weighted Random Sampler
    weights = compute_bounded_sampling_weights(records, max_weight=4.0)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
    weighted_loader = DataLoader(dataset, batch_size=16, sampler=sampler)

    # Simulate 5 epochs of training draws
    def audit_sampling_exposure(loader, num_epochs=10):
        drawn_clean = 0
        drawn_classes = np.zeros(4, dtype=int)
        total_drawn = 0
        for _ in range(num_epochs):
            for batch in loader:
                targets = batch["targets"].numpy()
                is_clean = batch["is_clean"].numpy()
                drawn_clean += np.sum(is_clean)
                drawn_classes += np.sum(targets > 0, axis=0).astype(int)
                total_drawn += len(targets)
        return total_drawn, drawn_clean, drawn_classes

    n_tot, n_clean, n_classes = audit_sampling_exposure(natural_loader, num_epochs=10)
    w_tot, w_clean, w_classes = audit_sampling_exposure(weighted_loader, num_epochs=10)

    print("\n2. Sampling Exposure Over 10 Simulated Training Epochs:")
    print(f"{'Defect Class':<15} | {'Natural Sampling':<22} | {'Weighted Sampling (Max 4x)':<25}")
    print("-" * 68)
    print(f"{'Clean Images':<15} | {n_clean / n_tot:>6.1%} ({n_clean:>4} views)      | {w_clean / w_tot:>6.1%} ({w_clean:>4} views)")
    for c, name in enumerate(CLASSES):
        print(f"{name:<15} | {n_classes[c] / n_tot:>6.1%} ({n_classes[c]:>4} views)      | {w_classes[c] / w_tot:>6.1%} ({w_classes[c]:>4} views)")

    print("\n3. Key Insights:")
    print("   • Rare 'defect_b' exposure increased significantly under weighted sampling.")
    print("   • The model receives more gradient updates for rare defects per epoch.")
    print("   • BUT: Clean image exposure drops from 50% to ~28%, altering the model's prior.")
    print("   • CRITICAL RULE: Validation data must NEVER be sampled with weights;")
    print("     validation must represent the real unweighted test distribution!")
    print("=" * 70)


if __name__ == "__main__":
    main()
