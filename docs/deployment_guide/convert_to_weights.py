"""Turn almost any saved PyTorch file (.pth / .pt) into a clean weights.pt.

Usage:   python convert_to_weights.py model.pth            -> writes weights.pt
         python convert_to_weights.py model.pth out.pt     -> writes out.pt

Handles: a plain state_dict, a training checkpoint ({"model_state_dict": ...}),
weights saved from DataParallel ("module." prefix), a TorchScript file, and a
whole pickled model (needs the class importable: keep the .py file that defines
it in the same folder as this script).
"""
import sys
import warnings

import torch

warnings.filterwarnings("ignore")      # hide harmless PyTorch deprecation notices

KEYS = ("model_state_dict", "state_dict", "model", "net", "network")


def load_any(path):
    try:                                   # 1. safe load: plain tensors / dicts only
        return torch.load(path, map_location="cpu")
    except Exception as safe_error:
        first = safe_error
    try:                                   # 2. TorchScript archive
        return torch.jit.load(path, map_location="cpu")
    except Exception:
        pass
    print("Safe loading failed, so this file holds more than plain weights.")
    print("Trying full unpickling. Only do this for files YOU created.\n")
    sys.path.insert(0, ".")
    try:                                   # 3. whole pickled model
        return torch.load(path, map_location="cpu", weights_only=False)
    except (AttributeError, ModuleNotFoundError) as error:
        sys.exit(
            f"\nCould not rebuild the saved class: {error}\n"
            "The file stores the model's class NAME, not its code.\n"
            "Fix: run these two lines in the notebook/script where the model was trained,\n"
            "where the class already exists, then use the resulting weights.pt:\n\n"
            "    import torch\n"
            "    torch.save(model.state_dict(), 'weights.pt')\n"
        )


def to_state_dict(obj):
    if isinstance(obj, torch.nn.Module):   # whole model or TorchScript
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in KEYS:                   # checkpoint wrapper
            if key in obj:
                return to_state_dict(obj[key])
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj                     # already a state_dict
    sys.exit(f"Do not know how to read a {type(obj).__name__}. "
             f"Top-level keys: {list(obj)[:10] if isinstance(obj, dict) else 'n/a'}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path, out = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "weights.pt")
    state = to_state_dict(load_any(path))
    state = {k.removeprefix("module."): v.detach().cpu().contiguous() for k, v in state.items()}
    torch.save(state, out)
    print(f"Wrote {out}: {len(state)} tensors. First names:")
    for name in list(state)[:5]:
        print("  ", name, tuple(state[name].shape))
    print("\nNext: put these names next to your model.py layers; they must match.")


if __name__ == "__main__":
    main()
