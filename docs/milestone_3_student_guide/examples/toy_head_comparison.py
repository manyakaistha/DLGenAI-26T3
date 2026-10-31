"""Educational Toy Example: Output Head Semantics (4-Sigmoid vs 5-Softmax).

Demonstrates:
1. Multi-label independent sigmoid probabilities vs mutually exclusive softmax.
2. The synthetic 2-defect pixel experiment where two defect logits are high.
3. Decision rules (thresholding vs argmax).
"""
import torch
import torch.nn.functional as F


def compare_heads():
    print("=== Educational Head Semantics Comparison ===\n")

    # Synthetic pixel with 5 class logits: [Background, Defect_A, Defect_B, Defect_C, Defect_D]
    # Suppose Defect A (scratch) and Defect C (inclusion) both have high positive logits:
    logits_5class = torch.tensor([[-1.0, 2.5, -4.0, 2.1, -3.0]])
    classes_5 = ["Background", "defect_a", "defect_b", "defect_c", "defect_d"]
    classes_4 = ["defect_a", "defect_b", "defect_c", "defect_d"]

    # 1. 4-Channel Independent Sigmoid (ignores background channel)
    logits_4class = logits_5class[:, 1:]  # [2.5, -4.0, 2.1, -3.0]
    probs_sigmoid = torch.sigmoid(logits_4class)

    print("--- 1. Multi-Label Independent Sigmoid Head (4 channels) ---")
    print("Raw Logits:    ", [round(x.item(), 2) for x in logits_4class[0]])
    print("Probabilities: ", [round(x.item(), 4) for x in probs_sigmoid[0]])
    print(f"Sum of defect probabilities: {probs_sigmoid.sum().item():.4f} (NOT constrained to 1.0!)")

    tau = 0.5
    binary_preds = (probs_sigmoid >= tau).int()
    print(f"Decisions (tau={tau}):")
    for cls_name, p, d in zip(classes_4, probs_sigmoid[0], binary_preds[0]):
        status = "ACTIVE DEFECT" if d.item() == 1 else "clean"
        print(f"  • {cls_name:10s}: P = {p.item():.4f} -> {status}")
    print("  -> Pixel claims BOTH defect_a and defect_c simultaneously!\n")

    # 2. 5-Channel Mutually Exclusive Softmax Head (includes background channel)
    probs_softmax = F.softmax(logits_5class, dim=1)

    print("--- 2. Multi-Class Mutually Exclusive Softmax Head (5 channels) ---")
    print("Raw Logits:    ", [round(x.item(), 2) for x in logits_5class[0]])
    print("Probabilities: ", [round(x.item(), 4) for x in probs_softmax[0]])
    print(f"Sum of all 5 probabilities: {probs_softmax.sum().item():.4f} (EXACTLY 1.0!)")

    argmax_idx = torch.argmax(probs_softmax, dim=1).item()
    winning_class = classes_5[argmax_idx]
    print(f"Decisions (Argmax):")
    for cls_name, p in zip(classes_5, probs_softmax[0]):
        is_winner = "WINNER" if cls_name == winning_class else "suppressed"
        print(f"  • {cls_name:10s}: P = {p.item():.4f} -> {is_winner}")
    print(f"  -> Class {argmax_idx} ({winning_class}) wins with {probs_softmax[0, argmax_idx].item()*100:.1f}%.")
    print("  -> Defect C probability was suppressed from 89.1% (sigmoid) down to 39.3% (softmax) by competition.")


if __name__ == "__main__":
    compare_heads()
