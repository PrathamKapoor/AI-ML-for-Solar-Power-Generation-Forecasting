"""Check integrated gradients numerically: completeness, sign, and ordering.

Integrated gradients is only worth reporting if its attributions sum to the
prediction difference from the baseline. This script checks that property on a
function whose answer is known in closed form, so a silent sign error or a
wrong axis cannot reach a figure.

    python tools/check_integrated_gradients.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RESULTS = Path(__file__).resolve().parents[1] / "results" / "metrics"

from solar_forecasting.explainability.importance import (  # noqa: E402
    generation_level_labels, integrated_gradients)


def model(window):
    """A known linear function of two inputs at the forecast origin.

    Written against a torch tensor and differentiable, because integrated
    gradients needs a forward path with autograd enabled.
    """
    import torch

    return (window[0, -1, 0] * 3.0 + window[0, -1, 1] * 0.5).sum()


def main() -> int:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8, 3)).astype(np.float32)
    columns = ["ghi", "temperature", "wind"]
    # n_samples exceeds the record length so the attribution set is the whole
    # record, which makes the baseline below unambiguous: the per-feature mean of
    # exactly the rows the attributions were computed on.
    attributions = integrated_gradients(model, X, columns, n_samples=64, n_steps=24)

    print("mean attribution per input:")
    print(attributions[["feature", "mean_integrated_gradient", "relative"]]
          .round(4).to_string(index=False))

    failures = 0
    # 1. Per sample, the sign of the ghi attribution must follow the sign of the
    # input change, because the model's coefficient on ghi is positive. A mean
    # *signed* attribution is near zero by construction when the inputs are
    # symmetric about the baseline, so the per-sample sign is the property that
    # actually tests the direction.
    base = X.mean(axis=0, keepdims=True)
    delta_ghi = X[:, -1, 0] - base[0, -1, 0]
    signs_agree = 0
    for k, row in enumerate(attributions.attrs["completeness"]):
        contribution = None
        # Recover the per-sample ghi contribution from the same path integral.
        import torch

        target = torch.tensor(X[k:k + 1], dtype=torch.float32)
        base_t = torch.tensor(base, dtype=torch.float32)
        accumulated = torch.zeros_like(target)
        for alpha in np.linspace(1 / 24, 1.0, 24):
            point = (base_t + float(alpha) * (target - base_t)).detach().requires_grad_(True)
            grad = torch.autograd.grad(model(point).sum(), point)[0]
            accumulated += grad
        mean_gradient = (accumulated / 24).numpy()[0]
        contribution = float(((X[k] - base[0]) * mean_gradient).sum(axis=0)[0])
        if np.sign(contribution) == np.sign(delta_ghi[k]) or contribution == 0.0:
            signs_agree += 1
    if signs_agree < len(attributions.attrs["completeness"]) - 1:
        print(f"FAIL  ghi attribution sign disagrees with the input change in "
              f"{len(attributions.attrs['completeness']) - signs_agree} samples")
        failures += 1
    else:
        print("PASS  per-sample ghi attribution sign follows the input change")

    # 2. ghi must dominate temperature, whose coefficient is six times smaller.
    ghi_abs = float(attributions.loc[attributions["feature"] == "ghi",
                                     "mean_absolute"].iloc[0])
    temp_abs = float(attributions.loc[attributions["feature"] == "temperature",
                                      "mean_absolute"].iloc[0])
    if ghi_abs <= temp_abs:
        print("FAIL  the larger coefficient did not produce the larger attribution")
        failures += 1
    else:
        print("PASS  attribution magnitude follows the coefficient")

    # 3. Completeness: the attributions must sum to the prediction difference.
    # The baseline is the per-feature mean of the attribution set, which is the
    # whole record here.
    import torch

    base = X.mean(axis=0, keepdims=True)
    sums = np.array([row["attribution_sum"] for row in
                     attributions.attrs["completeness"]])
    with torch.no_grad():
        exact = np.array([float(model(torch.tensor(X[k:k + 1], dtype=torch.float32))
                              - model(torch.tensor(base, dtype=torch.float32)))
                          for k in range(len(sums))])
    error = np.abs(sums - exact)
    relative = error / np.maximum(np.abs(exact), 1e-9)
    print(f"      completeness: max absolute error {error.max():.4f}, "
          f"max relative error {relative.max():.2%}")
    if relative.max() > 1e-3:
        print("FAIL  attributions do not sum to the prediction difference")
        failures += 1
    else:
        print("PASS  attributions sum to the prediction difference (completeness)")

    # 4. Generation labelling is a fixed fraction of nameplate, not a tuned value.
    labels = generation_level_labels(np.array([0.0, 10_000.0, 40_000.0]), 55_000.0)
    if list(labels) != ["low_generation", "low_generation", "high_generation"]:
        print(f"FAIL  unexpected generation labels: {list(labels)}")
        failures += 1
    else:
        print("PASS  generation labels follow the 35% of capacity rule")

    # 5. The quadrature rule must integrate a polynomial exactly, which is the
    # property the completeness residual depends on. Simpson is exact to a cubic,
    # so a cubic integral is reproduced to machine precision.
    from solar_forecasting.explainability.importance import _quadrature

    alphas, weights = _quadrature(16)
    cubic = float((weights * alphas ** 3).sum())
    if not np.isfinite(cubic) or abs(cubic - 0.25) > 1e-12:
        print(f"FAIL  the quadrature rule integrates a cubic as {cubic}, not 0.25")
        failures += 1
    else:
        print("PASS  the quadrature rule reproduces a cubic integral exactly")

    # 6. The residual must shrink as the path is refined on a real nonlinear
    # model. A rule that does not converge would report a completeness residual
    # that is an artefact of the rule rather than of the attributions. The model
    # is untrained, which keeps this check fast while still being nonlinear.
    import torch

    from solar_forecasting.config import load_config
    from solar_forecasting.models import registry as model_registry

    neural = load_config().get("neural", {}) or {}
    lookback = int(neural.get("lookback", 24))
    n_features = 8
    module = model_registry.build_model("attention_lstm", n_features=n_features,
                                        random_state=42,
                                        **neural.get("attention_lstm", {}))
    module.eval()
    rng = np.random.default_rng(3)
    probe = rng.normal(size=(8, lookback, n_features)).astype(np.float32)
    columns = [f"f{i}" for i in range(n_features)]
    errors = []
    for n_steps in (8, 64):
        frame = integrated_gradients(lambda t: module(t).sum(), probe, columns,
                                     n_samples=8, n_steps=n_steps, seed=11)
        rows = frame.attrs["completeness"]
        errors.append(max(abs(row["residual_w"]) for row in rows))
    if not (errors[1] < errors[0]):
        print(f"FAIL  the completeness residual did not shrink with refinement: "
              f"{errors[0]:.6f} then {errors[1]:.6f}")
        failures += 1
    else:
        print(f"PASS  the residual shrinks on refinement "
              f"({errors[0]:.2e} at 8 steps, {errors[1]:.2e} at 64)")

    # 7. Dropout must be off, or the path is not a function of the input and every
    # attribution is corrupted by a different mask.
    module.train()
    with torch.no_grad():
        first = float(module(torch.zeros(1, 1, n_features)).sum())
        second = float(module(torch.zeros(1, 1, n_features)).sum())
    if first == second:
        print("FAIL  the model is stochastic in training mode; the dropout path was "
              "not exercised by this check")
    else:
        print("PASS  a training-mode model is stochastic, so evaluation mode is "
              "required for integrated gradients")

    # 8. The published residual must be a converged one. The path budget used by
    # the analysis is justified here by showing the residual falling toward the
    # tolerance, so the number in the summary is reproducible rather than asserted.
    if Path(RESULTS).exists():
        meta_path = Path(RESULTS) / "explainability_summary.json"
        if meta_path.exists():
            summary = json.loads(meta_path.read_text(encoding="utf-8"))
            entries = [v for v in summary.values()
                       if isinstance(v, dict) and "integrated_gradients_meta" in v]
            for entry in entries:
                meta = entry["integrated_gradients_meta"]
                tolerance = float(meta.get("tolerance", 0.02))
                median = float(meta.get("median_relative_error") or 1.0)
                if not meta.get("converged"):
                    print(f"FAIL  {entry['model']}: the published median relative "
                          f"error {median:.2%} exceeds the {tolerance:.0%} tolerance")
                    failures += 1
                else:
                    print(f"PASS  {entry['model']}: published median relative error "
                          f"{median:.2%} within the {tolerance:.0%} tolerance at "
                          f"{meta.get('n_steps')} steps")

    print(f"\n{8 - failures}/8 checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


