"""CEGIS: counterexample-guided fine-tuning of a trained network until it verifies as safe.

Loop:  verify (Marabou) -> collect counterexample start states -> fine-tune the network on
           L_total = L_ori + LAMBDA_CE * L_ce
       -> export '<Model>_CEGIS.onnx' (next to the original, which is never touched) -> verify again ...
L_ori is exactly the original training loss (the network keeps imitating the expert data). L_ce rolls the CURRENT network out
from the stored counterexample start states with the true, differentiable dynamics and penalises every violation of the active
safety constraints (constraints.ce_penalties), pushing those trajectories to the safe side.
All counterexamples of all rounds are kept (Datasets/Counterexamples) and reused in every fine-tuning round.
Result of run_cegis: 'SAFE', 'UNRESOLVED' (iteration / time limit) or 'UNKNOWN' (nothing to learn from and no proof).
"""
import os
import sys
import time
import numpy as np
import pandas as pd

import config
import constraints
import utils
from verifier import run_verification


# --------------------------------------------------------------------------- reuse of the training code in "Neural Networks"
def load_training_code():
    """
    Imports 'Neural Networks/utils.py', 'dataset.py' and 'Arch' (same loss, same data windows, same ONNX export as the training
    scripts). Both folders have a config.py / utils.py, so the names are swapped out while importing and restored afterwards.
    """
    # Both 'Neural Networks' and 'Verification' contain modules called config and utils, so the Verification versions are
    # removed from sys.modules while the training code is imported, and put back afterwards.
    clashing = lambda name: name in ("config", "utils", "progress", "dataset", "Arch") or name.startswith("Arch.")
    saved = {k: v for k, v in sys.modules.items() if clashing(k)}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, config.NEURAL_NETWORKS_PATH)
    try:
        import utils as nn_utils
        import dataset as nn_dataset
        from Arch.simple_net import SimpleNet
        from Arch.unicycle_net import UnicycleNet
        import config as nn_config
    finally:
        sys.path.remove(config.NEURAL_NETWORKS_PATH)
        for k in [k for k in sys.modules if clashing(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
    return nn_utils, nn_dataset, {"simple": SimpleNet, "unicycle": UnicycleNet}, nn_config


def model_from_layers(net_class, layers):
    """A trainable PyTorch network holding the weights of the ONNX network (the ONNX file already has the normalisation folded in)."""
    import torch
    import torch.nn as nn
    model = net_class()
    # Copy the weights of the ONNX network (normalisation already folded in) into a fresh PyTorch network of the same architecture.
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    if len(linears) != len(layers):
        raise ValueError("The ONNX network does not have the architecture of the Arch class")
    with torch.no_grad():
        for linear, (W, b, _) in zip(linears, layers):
            linear.weight.copy_(torch.tensor(W, dtype=torch.float32))
            linear.bias.copy_(torch.tensor(b, dtype=torch.float32))
    return model


# --------------------------------------------------------------------------- L_ce
def counterexample_loss(model, x0, env, nn_utils):
    """
    L_ce for a batch of counterexample start states: roll the CURRENT policy out in closed loop with the true dynamics (differentiable)
    and sum, over the active constraints, the weighted violation penalties of constraints.ce_penalties. 0 means: safe with the buffer.
    """
    import torch
    states, controls, s = [x0], [], x0
    for _ in range(env.T):
        # Differentiable closed loop: policy -> true dynamics, T times; then the penalties of all active constraints are summed.
        u = model(s)
        s = nn_utils.compute_next_state(s, u, env.model, env.dt)
        controls.append(u)
        states.append(s)
    penalties = constraints.ce_penalties(env, torch.stack(states, dim=1), torch.stack(controls, dim=1), torch)
    total = 0.0
    for name, value in penalties.items():
        total = total + config.CE_WEIGHTS[name] * value.mean()
    return total, {name: float(value.mean()) for name, value in penalties.items()}


def finetune(model, train_loader, state_std, ce_starts, env, nn_utils, nn_config):
    """FINETUNE_STEPS optimiser steps on L_total = L_ori + LAMBDA_CE * L_ce (L_ori is exactly the training loss)."""
    import torch
    optimizer = torch.optim.Adam(model.parameters(), lr=config.FINETUNE_LR)
    ce = torch.tensor(ce_starts, dtype=torch.float32)

    def endless(loader):          # re-iterates the loader (itertools.cycle would keep every batch in memory)
        while True:
            for batch in loader:
                yield batch

    batches = endless(train_loader)
    model.train()
    for step in range(1, config.FINETUNE_STEPS + 1):
        xb, yb, nb = next(batches)
        pred = nn_utils.rollout_states(model, xb, env.model, env.dt, nb.shape[1])
        l_ori = nn_utils.normalized_loss(nn_utils.state_error(pred, nb, env.model), state_std, env.model)
        # Random mini-batch of stored counterexample start states for this step.
        pick = ce[torch.randint(len(ce), (min(config.CE_BATCH, len(ce)),))]
        l_ce, parts = counterexample_loss(model, pick, env, nn_utils)
        # L_total = L_ori (keep imitating the expert) + LAMBDA_CE * L_ce (stay safe on the counterexamples).
        loss = l_ori + config.LAMBDA_CE * l_ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), nn_config.GRADIENT_CLIP)
        optimizer.step()
        if step % max(1, config.FINETUNE_STEPS // 5) == 0 or step == config.FINETUNE_STEPS:
            print(f"    step {step}/{config.FINETUNE_STEPS}  L_ori={l_ori.item():.5f}  L_ce={float(l_ce):.5f}  {({k: round(v, 4) for k, v in parts.items()})}")
    model.eval()
    return model


# --------------------------------------------------------------------------- the CEGIS loop
def run_cegis(model_name):
    """
    verify -> collect counterexamples -> fine-tune on L_ori + LAMBDA_CE * L_ce -> export "<Model>_CEGIS.onnx" -> verify again ...
    Returns "SAFE" (verified), "UNRESOLVED" (iteration / time limit hit) or "UNKNOWN" (Marabou could neither find a counterexample nor prove safety).
    """
    info = config.MODELS[model_name]
    env = utils.Environment(model_name)
    nn_utils, nn_dataset, net_classes, nn_config = load_training_code()
    cegis_path = utils.cegis_network_file(model_name)
    # Start from the previous CEGIS network when CEGIS_RESUME is True, otherwise from the original trained network.
    start_path = cegis_path if (config.CEGIS_RESUME and os.path.isfile(cegis_path)) else os.path.join(config.NETWORK_PATH, info["onnx"] + ".onnx")

    # data of the original loss (same filtering and windows as the training scripts)
    df = pd.read_csv(os.path.join(config.DATASET_PATH, info["dataset"]))
    df = df[df["STATUS"].isin(info["ok_status"])].reset_index(drop=True)
    df = df[df["LENGTH"].notna()].reset_index(drop=True)
    build = nn_dataset.build_simple_dataset if model_name == "simple" else nn_dataset.build_unicycle_dataset
    X, Y, N = build(df)
    batch_size = nn_config.TRAINING_PRESETS[config.DATASET_SIZE]["BATCH_SIZE"]
    train_loader, _, stats = nn_dataset.create_dataloaders(X, Y, N, batch_size=batch_size)
    import torch
    state_std = torch.tensor(stats["in_std"])

    layers = utils.load_onnx_layers(start_path)
    model = model_from_layers(net_classes[model_name], layers)
    current_path = start_path
    started = time.time()

    # ---- CEGIS loop: each iteration verifies the current network and, if it is not safe, fine-tunes it ----
    for iteration in range(1, config.MAX_CEGIS_ITERATIONS + 1):
        print(f"\n===== [{info['name']} CEGIS] iteration {iteration}/{config.MAX_CEGIS_ITERATIONS} | verifying {current_path} =====")
        # Verify the current network; collect up to MAX_COUNTEREXAMPLES counterexamples (they are appended to the counterexample CSV).
        result = run_verification(model_name, onnx_path=current_path, max_counterexamples=config.MAX_COUNTEREXAMPLES)
        if result["verdict"] == "SAFE":
            print(f"[{info['name']} CEGIS] SAFE after {iteration - 1} fine-tuning round(s). Network: {current_path}")
            return "SAFE"
        # Use every counterexample found so far (all rounds), not only the newest.
        starts = utils.load_counterexample_starts(env)      # every counterexample found so far, from all rounds
        if result["verdict"] == "UNKNOWN" or len(starts) == 0:
            print(f"[{info['name']} CEGIS] UNKNOWN: no counterexample to learn from and no proof of safety.")
            return "UNKNOWN"
        if time.time() - started > config.CEGIS_TIME_LIMIT:
            break
        print(f"[{info['name']} CEGIS] fine-tuning on {len(starts)} stored counterexample start state(s)")
        model = finetune(model, train_loader, state_std, starts, env, nn_utils, nn_config)
        os.makedirs(config.NETWORK_PATH, exist_ok=True)
        # Write '<Model>_CEGIS.onnx'; the next iteration verifies this file.
        nn_utils.export_onnx(model, env.state_dim, cegis_path)
        current_path = cegis_path

    print(f"[{info['name']} CEGIS] UNRESOLVED: limit reached. Last network: {current_path}")
    return "UNRESOLVED"
