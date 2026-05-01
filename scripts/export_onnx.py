"""Export a trained BC checkpoint to ONNX for inspection / deployment.

Usage
-----
    python scripts/export_onnx.py --checkpoint runs/bc_pilotnet_v1/checkpoints/best.pt

    # custom output path + skip the sanity-check roundtrip
    python scripts/export_onnx.py \\
        --checkpoint runs/bc_pilotnet_v1/checkpoints/best.pt \\
        --output /tmp/bc.onnx --no-verify

The script loads the checkpoint exactly the way ``inference.BCController``
does (same image size, same model arch, same state dict) so the exported
graph matches what runs in ``run_in_carla.py``. The batch dimension is
exported as a dynamic axis, so the ONNX can be used for both visualization
(Netron) and batched inference at any size.

Why a separate script instead of a flag on ``run_in_carla.py``?
  * No CARLA dependency — this runs on a laptop without the simulator.
  * Keeps the inference path honest: the ONNX is built from ``model.forward``
    directly, not from an ad-hoc wrapper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make the project package importable when running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.model import BCModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to a BC checkpoint (.pt) saved by training.")
    p.add_argument("--output", default=None,
                   help="Output ONNX path. Defaults to <checkpoint>.onnx.")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version (default 17 — widely supported).")
    p.add_argument("--device", default="cpu",
                   help="Device used to build the model graph. CPU is fine "
                        "for export; the resulting ONNX is device-agnostic.")
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                   help="If set, run onnxruntime on a random input and "
                        "compare against the PyTorch output.")
    p.add_argument("--atol", type=float, default=1e-4,
                   help="Absolute tolerance for the verification roundtrip.")
    return p.parse_args()


def _load_model(checkpoint_path: Path, device: torch.device):
    """Rebuild BCModel from the checkpoint — mirrors BCController.__init__."""
    state = torch.load(
        str(checkpoint_path), map_location=device, weights_only=False,
    )
    cfg = state["config"]
    image_cfg = cfg["data"]["image"]

    input_hw = (int(image_cfg["resize_height"]), int(image_cfg["resize_width"]))
    model = BCModel(
        arch=state.get("arch", cfg["model"].get("arch", "pilotnet")),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        activation=state.get("activation", cfg["model"].get("activation", "bounded")),
        input_hw=input_hw,
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, input_hw


def _default_output_path(checkpoint: Path) -> Path:
    # Place the .onnx alongside the .pt with the same stem so it's easy to find.
    return checkpoint.with_suffix(".onnx")


def _verify_roundtrip(model: torch.nn.Module,
                      onnx_path: Path,
                      input_hw: tuple[int, int],
                      atol: float) -> None:
    """Run the ONNX through onnxruntime and compare against PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify] onnxruntime not installed — skipping roundtrip check. "
              "Install with `pip install onnxruntime` to enable.", file=sys.stderr)
        return

    h, w = input_hw
    # Use a deterministic input so any mismatch is reproducible.
    rng = np.random.default_rng(0)
    dummy_np = rng.standard_normal((2, 3, h, w), dtype=np.float32)
    dummy_pt = torch.from_numpy(dummy_np)

    with torch.no_grad():
        pt_out = model(dummy_pt).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {in_name: dummy_np})[0]

    diff = float(np.max(np.abs(pt_out - ort_out)))
    if diff <= atol:
        print(f"[verify] PyTorch vs ONNXRuntime max |Δ| = {diff:.2e}  (≤ atol={atol:.1e}) ✓")
    else:
        print(f"[verify] PyTorch vs ONNXRuntime max |Δ| = {diff:.2e}  "
              f"(> atol={atol:.1e}) — outputs disagree!", file=sys.stderr)
        sys.exit(2)


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    out_path = Path(args.output).expanduser().resolve() if args.output \
        else _default_output_path(checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, (h, w) = _load_model(checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {checkpoint.name}  arch={model.arch}  input={h}x{w}  "
          f"params={n_params:,}")

    # Dummy input matches the training resize dims. Dynamic batch axis so
    # the exported graph accepts any batch size at inference / visualization.
    dummy = torch.zeros(1, 3, h, w, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=args.opset,
        input_names=["image"],
        output_names=["controls"],
        dynamic_axes={
            "image":    {0: "batch"},
            "controls": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"ONNX written to {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)")

    if args.verify:
        _verify_roundtrip(model, out_path, (h, w), args.atol)

    print()
    print("Next steps:")
    print(f"  * Open in Netron:  netron {out_path}   (or drop the file at https://netron.app)")
    print(f"  * Inspect graph:   python -c \"import onnx; m = onnx.load('{out_path}'); "
          "print(onnx.helper.printable_graph(m.graph))\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
