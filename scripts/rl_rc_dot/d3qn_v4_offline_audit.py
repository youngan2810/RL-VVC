"""Offline V4 root-cause audit (no VVenC, VVdeC, YOLO, or new RL rollout).

Use only a trusted V4 SAFE ZIP with versioned checkpoints and replay tensors.
The supervised prediction target is the REALIZED behavior-policy return,
not Q* and not a ground-truth advantage label.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import io
import json
import math
import os
from pathlib import Path
import random
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .d3qn_agent_v2 import DuelingQNetwork

EXPECTED_VERSION = "d3qn-native-checkpoint/v4"
EXPECTED_PROTOCOL = "native-masked-d3qn-qp/v4-terminal-aware"
EXPECTED_SOURCE_COMMIT = "ace1db10e05b140ad49e1a676f80563219fb1ede"
EPISODE_LENGTH = 256


def require(condition, msg):
    if not condition:
        raise ValueError(msg)


def read_json(zf, name):
    return json.loads(zf.read(name).decode("utf-8"))


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def safe_stats(x):
    a = np.asarray(x, dtype=np.float64)
    if not a.size:
        return {"count": 0}
    return {
        "count": int(a.size),
        "mean": float(a.mean()), "std": float(a.std()),
        "min": float(a.min()), "median": float(np.median(a)),
        "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def load_safe(path):
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        required = {
            "learning_manifest.json", "train_history.json",
            "checkpoints/latest_checkpoint.pt", "checkpoints/best_checkpoint.pt",
        }
        require(required <= names,
                "SAFE ZIP is missing: " + str(sorted(required - names)))
        manifest = read_json(zf, "learning_manifest.json")
        history = read_json(zf, "train_history.json")
        require(manifest.get("protocol") == EXPECTED_PROTOCOL, "Wrong V4 protocol")
        require(manifest.get("source_commit") == EXPECTED_SOURCE_COMMIT,
                "Unexpected checkpoint source commit")
        require(manifest.get("test_used") is False, "Test split must not be used")
        require(manifest.get("completed_steps") == 5120 and len(history) == 20,
                "This audit requires the completed 5120-step Train16 run")
        require(all(row["total_steps"] == (i + 1) * EPISODE_LENGTH
                    for i, row in enumerate(history)), "Training history order invalid")
        require(manifest.get("status") == "complete", "SAFE run is not marked complete")
        ckpts = {}
        for label in ("latest", "best"):
            name = "checkpoints/" + label + "_checkpoint.pt"
            # weights_only=True intentionally excludes arbitrary pickle execution.
            ck = torch.load(io.BytesIO(zf.read(name)),
                            map_location="cpu", weights_only=True)
            require(ck.get("version") == EXPECTED_VERSION,
                    "Unsupported checkpoint version: " + label)
            require(ck.get("identity", {}).get("protocol") == EXPECTED_PROTOCOL,
                    "Checkpoint identity mismatch: " + label)
            require(ck.get("config", {}).get("n_step") == 3
                    and ck.get("config", {}).get("reward_scale") == 25.0,
                    "Unsupported n-step/reward scale in " + label)
            ckpts[label] = ck
        require(ckpts["latest"]["steps"] == 5120, "Incorrect latest step")
        require(ckpts["best"]["steps"] == manifest["best_steps"],
                "Incorrect best checkpoint step")
        trace_names = sorted(x for x in names
                             if x.startswith("frame_traces/") and x.endswith(".jsonl.gz"))
        trace_contents = [(name, zf.read(name)) for name in trace_names]
        baseline = read_json(zf, "baseline.json") if "baseline.json" in names else []
        monitor = read_json(zf, "monitor_history.json") if "monitor_history.json" in names else []
        prior = read_json(zf, "value_audit.json") if "value_audit.json" in names else None
    return manifest, history, ckpts, trace_contents, baseline, monitor, prior


def check_replay(ck, history=None):
    r = ck["replay"]
    n = int(ck["steps"])
    require(n % EPISODE_LENGTH == 0, "Incomplete checkpoint episodes")
    require(len(r["actions"]) == n, "Replay length does not match checkpoint steps")
    require(tuple(r["obs_global"].shape) == (n, 25), "Bad global tensor")
    require(tuple(r["obs_maps"].shape) == (n, 5, 10, 15), "Bad maps tensor")
    require(tuple(r["obs_legal"].shape) == (n, 25), "Bad legal tensor")
    require(bool(r["obs_legal"].gather(1, r["actions"].long()[:, None]).all()),
            "Replay contains an invalid chosen action")
    ns = int(ck["config"]["n_step"])
    done = r["dones"].cpu().numpy().astype(bool)
    expected = (np.arange(n) % EPISODE_LENGTH) >= EPISODE_LENGTH - ns
    require(np.array_equal(done, expected), "Unexpected n-step terminal flags")
    if history is not None:
        require(len(history) == n // EPISODE_LENGTH, "History/replay episode count differs")
    return r


def recover_single_rewards_and_mc(r, ns, scale, history):
    """Reverse exact gamma=1 n-step sums independently for every episode.

    No overlapping n-step reward double-counting: validate against original
    per-episode objective recorded by the actual native environment.
    """
    n = len(r["rewards"])
    R = r["rewards"].cpu().numpy().astype(np.float64)
    single = np.zeros(n, np.float64)
    mc = np.zeros(n, np.float64)
    for start in range(0, n, EPISODE_LENGTH):
        end = start + EPISODE_LENGTH
        for t in range(end - 1, start - 1, -1):
            single[t] = R[t] - single[t + 1: min(end, t + ns)].sum()
            mc[t] = single[t] + (mc[t + 1] if t + 1 < end else 0.)
        clip_reward = float(history[start // EPISODE_LENGTH]["episode_reward"]) * scale
        require(abs(mc[start] - clip_reward) < 0.002,
                f"Recovered MC does not match true episode reward at {start}: "
                f"{mc[start]} vs {clip_reward}")
    for t in range(n):
        stop = min((t // EPISODE_LENGTH + 1) * EPISODE_LENGTH, t + ns)
        require(abs(single[t:stop].sum() - R[t]) < 0.001,
                f"n-step reconstruction error at index {t}")
    return R, single, mc


def net_from(weights, device):
    m = DuelingQNetwork().to(device)
    m.load_state_dict(weights, strict=True)
    m.eval()
    return m


def observe(r, ix, slot, device):
    # Use packed tensors directly; no VVenC or detector dependency.
    a = ("obs_" if slot == "obs" else "next_")
    return {
        "global": r[a + "global"][ix].to(device=device, dtype=torch.float32),
        "maps": r[a + "maps"][ix].to(device=device, dtype=torch.float32),
        "legal": r[a + "legal"][ix].to(device=device, dtype=torch.bool),
    }


def evaluate_checkpoint(ck, device):
    r = check_replay(ck)
    qnet = net_from(ck["online"], device)
    target_net = net_from(ck["target"], device)
    Q, future, chosen = [], [], []
    with torch.inference_mode():
        for i in range(0, len(r["actions"]), 128):
            ids = slice(i, i + 128)
            obs = observe(r, ids, "obs", device)
            nxt = observe(r, ids, "next", device)
            q = qnet(obs)
            nq = qnet(nxt).masked_fill(~nxt["legal"], -torch.inf)
            sel = nq.argmax(1, keepdim=True)
            boot = target_net(nxt).gather(1, sel).flatten()
            Q.append(q.cpu().numpy())
            future.append(boot.cpu().numpy())
            chosen.append(q.gather(1, r["actions"][ids].to(device)[:, None])
                          .flatten().cpu().numpy())
    Q = np.concatenate(Q)
    chosen = np.concatenate(chosen)
    future = np.concatenate(future)
    R = r["rewards"].cpu().numpy().astype(np.float64)
    done = r["dones"].cpu().numpy().astype(bool)
    target = R + np.where(done, 0.0, future)
    td = target - chosen
    return {
        "Q": Q, "q_chosen": chosen, "nstep_reward": R,
        "bootstrap": future, "td_target": target, "td_error": td,
        "done": done, "actions": r["actions"].cpu().numpy().astype(int),
        "legal": r["obs_legal"].cpu().numpy().astype(bool),
    }


def group_diagnostics(result, raw):
    n = len(result["td_error"])
    pos = np.arange(n) % EPISODE_LENGTH
    masks = {
        "all": np.ones(n, bool),
        "terminal_last3": pos >= 253,
        "nonterminal": pos < 253,
        "coding_step_mod32_0": pos % 32 == 0,
        "coding_step_mod64_0": pos % 64 == 0,
        "other_coding_steps": pos % 32 != 0,
        "frames_0_63": pos < 64,
        "frames_64_127": (pos >= 64) & (pos < 128),
        "frames_128_191": (pos >= 128) & (pos < 192),
        "frames_192_255": pos >= 192,
    }
    out = []
    for name, mask in masks.items():
        td = result["td_error"][mask]
        out.append({
            "group": name, "count": int(mask.sum()),
            "mean_nstep_reward": float(result["nstep_reward"][mask].mean()),
            "mean_single_reward": float(raw[mask].mean()),
            "mean_predicted_q": float(result["q_chosen"][mask].mean()),
            "mean_td_target": float(result["td_target"][mask].mean()),
            "mean_td_error": float(td.mean()),
            "mean_abs_td": float(np.abs(td).mean()),
            "huber_saturated_fraction_abs_td_gt_1": float((np.abs(td) > 1).mean()),
            "mean_abs_huber_gradient": float(np.minimum(np.abs(td), 1).mean()),
        })
    return out


def write_csv(path, data):
    if not data:
        return
    fields = list(data[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)


def trace_reward_components(trace_contents, out):
    """Read optional compact Val4 trajectories in CODING order, not POC order."""
    rows = []
    budget = EPISODE_LENGTH * 120000 / 30
    per_frame = 120000 / 30
    for path, blob in trace_contents:
        raw_lines = gzip.decompress(blob).decode("utf-8").splitlines()
        require(len(raw_lines) == 256, "Incomplete compact trajectory: " + path)
        for t, line in enumerate(raw_lines):
            d = json.loads(line)
            require(d["coding_step"] == t, "Coding order inconsistent: " + path)
            bits = float(d["au_bits"])
            cum = float(d["cumulative_bits"])
            before = cum - bits
            old = max(0.0, (before - 1.02 * t * per_frame) / budget)
            new = max(0.0, (cum - 1.02 * (t + 1) * per_frame) / budget)
            rf = float(d["frame_f1"]) / EPISODE_LENGTH
            rb = -0.15 * bits / budget
            rd = -3.0 * (new - old)
            reconstructed = rf + rb + rd
            require(abs(reconstructed - float(d["reward"])) < 1e-5,
                    "Reward decomposition mismatch: " + path + " at " + str(t))
            rows.append({
                "trace": Path(path).name, "coding_step": t, "poc": int(d["poc"]),
                "action": int(d["action"]), "f1_component": rf,
                "bit_component": rb, "rate_debt_component": rd,
                "reward": reconstructed, "au_bits": bits,
            })
    if rows:
        write_csv(out / "validation_reward_components.csv", rows)
    return {
        "trace_count": len(trace_contents),
        "frame_count": len(rows),
        "component_checks_passed": bool(rows),
        "top_abs_rate_debt": sorted(rows, key=lambda z: abs(z["rate_debt_component"]),
                                    reverse=True)[:12],
    }


def plot_distribution(result, raw, out):
    pos = np.arange(len(raw)) % EPISODE_LENGTH
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    xs = list(range(EPISODE_LENGTH))
    vals = [[raw[pos == t] for t in xs],
            [np.abs(result["td_error"][pos == t]) for t in xs]]
    for ax, groups, label in zip(
            axs, vals, ["Recovered single-step reward x25",
                        "Absolute Double-DQN TD error"]):
        y = [np.median(v) for v in groups]
        ax.plot(xs, y, lw=1.5, label="Median across 20 episodes")
        ax.axvline(253, ls="--", lw=1, color="gray")
        ax.set_xlabel("Coding-step position (NOT display POC)")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(out / "reward_td_by_coding_step.png", dpi=140)
    plt.close(fig)


def split_by_video(history, n, seed):
    groups = np.asarray([history[i // EPISODE_LENGTH]["train_video"]
                         for i in range(n)])
    video_ids = np.asarray(sorted(set(groups)))
    require(len(video_ids) == 16, "Expected 16 distinct training video IDs")
    g = np.random.default_rng(seed)
    g.shuffle(video_ids)
    # Split unique videos, including repeated episodes across rates/windows,
    # so adjacent clips of the same video never cross splits.
    roles = {
        "train": video_ids[:10].tolist(),
        "validation": video_ids[10:13].tolist(),
        "holdout": video_ids[13:].tolist(),
    }
    indices = {k: np.flatnonzero(np.isin(groups, v)) for k, v in roles.items()}
    for k, idx in indices.items():
        require(len(idx) >= EPISODE_LENGTH, "Empty video-disjoint " + k + " split")
    return roles, indices


def metrics(pred, y):
    p = np.asarray(pred, dtype=float)
    t = np.asarray(y, dtype=float)
    mse = float(np.mean((p - t) ** 2))
    variance = float(np.sum((t - t.mean()) ** 2))
    corr = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-10 and t.std() > 1e-10 else None
    return {
        "count": int(len(y)), "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(p - t))),
        "r2": float(1 - np.sum((p - t) ** 2) / max(variance, 1e-12)),
        "pearson": corr,
    }


def time_rate_features(globs):
    p = np.asarray(globs[:, 1], dtype=float)
    v = np.asarray(globs[:, 2], dtype=float)
    return np.column_stack([np.ones(len(p)), p, p*p, v, v*v, p*v])


def offline_supervised(ck, mc, history, device, seed, epochs, out):
    r = ck["replay"]
    N = len(mc)
    roles, ids = split_by_video(history, N, seed)
    y = mc.astype(np.float64)
    ymean = float(y[ids["train"]].mean())
    ystd = float(max(0.001, y[ids["train"]].std()))
    yl = (y - ymean) / ystd
    globs = r["obs_global"].cpu().numpy()
    train, valid, test = ids["train"], ids["validation"], ids["holdout"]
    baseline_constant = np.full(len(test), ymean)
    X = time_rate_features(globs)
    penalty = np.eye(X.shape[1], dtype=float) * 0.001
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(X[train].T @ X[train] + penalty,
                           X[train].T @ y[train])
    baseline_time = X[test] @ beta
    results = {
        "split": roles, "frame_counts": {k: int(len(v)) for k, v in ids.items()},
        "target_definition": "Realized behavior-policy remaining return x25; NOT Q*",
        "target_training_mean": ymean, "target_training_std": ystd,
        "models": {
            "train_mean": metrics(baseline_constant, y[test]),
            "progress_and_rate_ridge": metrics(baseline_time, y[test]),
        },
    }
    pred_rows = [
        {"index": int(i), "episode": int(i // EPISODE_LENGTH),
         "video": history[i // EPISODE_LENGTH]["train_video"],
         "coding_step": int(i % EPISODE_LENGTH),
         "target_mc": float(y[i]), "constant": float(ymean),
         "time_rate_ridge": float(baseline_time[j])}
        for j, i in enumerate(test)
    ]
    curves = {}
    # Compare two SUPERVISED optimization losses. These are NOT RL ablations.
    for loss_name in ("mse", "huber"):
        torch.manual_seed(seed)
        model = DuelingQNetwork().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005,
                                     weight_decay=1e-5)
        rng = np.random.default_rng(seed)
        best_val = float("inf")
        best_epoch = None
        best_weights = None
        patience = 7
        stale = 0
        curve = []
        batch_size = 128

        def predict(indices):
            model.eval()
            guesses = []
            with torch.inference_mode():
                for b in range(0, len(indices), 256):
                    take = indices[b:b+256]
                    obs = observe(r, take, "obs", device)
                    act = r["actions"][take].to(device).long()
                    p = model(obs).gather(1, act[:, None]).flatten()
                    guesses.append(p.cpu().numpy())
            return np.concatenate(guesses)

        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(train)
            losses = []
            for b in range(0, len(order), batch_size):
                take = order[b:b+batch_size]
                obs = observe(r, take, "obs", device)
                act = r["actions"][take].to(device).long()
                target = torch.as_tensor(yl[take].astype(np.float32),device=device)
                pred = model(obs).gather(1, act[:, None]).flatten()
                loss = F.mse_loss(pred, target) if loss_name == "mse" \
                    else F.smooth_l1_loss(pred, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            pval = predict(valid)
            val_rmse = float(np.sqrt(np.mean((pval - yl[valid])**2)))
            curve.append({
                "epoch": epoch, "train_loss": float(np.mean(losses)),
                "val_rmse_standardized": val_rmse,
            })
            if val_rmse < best_val - 1e-5:
                best_val = val_rmse
                best_epoch = epoch
                best_weights = {k: v.cpu().clone() for k,v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        require(best_weights is not None, "No supervised checkpoint selected")
        model.load_state_dict(best_weights)
        ptest = predict(test) * ystd + ymean
        results["models"]["supervised_" + loss_name] = {
            **metrics(ptest, y[test]),
            "best_epoch_by_video_disjoint_validation": int(best_epoch),
            "trained_epochs": int(len(curve)),
            "val_rmse_standardized": best_val,
        }
        curves[loss_name] = curve
        for j,row in enumerate(pred_rows):
            row["supervised_" + loss_name] = float(ptest[j])
        print("OFFLINE_SUPERVISED", loss_name,
              "epoch", best_epoch, "holdout_R2",
              round(results["models"]["supervised_"+loss_name]["r2"],4),
              "holdout_RMSE",
              round(results["models"]["supervised_"+loss_name]["rmse"],4),
              flush=True)
    write_csv(out / "supervised_holdout_predictions.csv", pred_rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, curve in curves.items():
        ax.plot([z["epoch"] for z in curve],
                [z["val_rmse_standardized"] for z in curve], label=name)
    ax.set_xlabel("Supervised epoch")
    ax.set_ylabel("Validation RMSE (standardized return)")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "supervised_validation_curve.png", dpi=150)
    plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for axis, name in zip(axs,("supervised_mse","supervised_huber")):
        p=np.asarray([x[name] for x in pred_rows])
        truth=np.asarray([x["target_mc"] for x in pred_rows])
        axis.scatter(truth,p,s=7,alpha=.28)
        lo=float(min(truth.min(),p.min()))
        hi=float(max(truth.max(),p.max()))
        axis.plot([lo,hi],[lo,hi],ls="--",lw=1,color="gray")
        axis.set_title(name)
        axis.set_xlabel("Realized MC return x25")
        axis.set_ylabel("Predicted behavior return x25")
    fig.savefig(out / "supervised_holdout_scatter.png",dpi=150)
    plt.close(fig)
    results["curves"] = curves
    save_json(out / "supervised_report.json", results)
    return results


def same_state_action_audit(best_ck, latest_ck, device):
    """Compare checkpoint rankings on IDENTICAL first 1024 replay observations."""
    br = check_replay(best_ck)
    lr = check_replay(latest_ck)
    n = len(br["actions"])
    require(torch.equal(br["obs_global"], lr["obs_global"][:n])
            and torch.equal(br["obs_legal"], lr["obs_legal"][:n])
            and torch.equal(br["actions"], lr["actions"][:n]),
            "Replay prefix differs: cannot perform same-state rank audit")
    net_b = net_from(best_ck["online"], device)
    net_l = net_from(latest_ck["online"], device)
    qb, ql = [], []
    with torch.inference_mode():
        for i in range(0,n,128):
            obs = observe(br,slice(i,i+128),"obs",device)
            qb.append(net_b(obs).cpu().numpy())
            ql.append(net_l(obs).cpu().numpy())
    qb=np.concatenate(qb)
    ql=np.concatenate(ql)
    mask=br["obs_legal"].cpu().numpy().astype(bool)
    gb=np.where(mask,qb,-np.inf).argmax(1)
    gl=np.where(mask,ql,-np.inf).argmax(1)
    chosen=br["actions"].cpu().numpy()
    selected_margin_b=qb[np.arange(n),chosen]
    selected_margin_l=ql[np.arange(n),chosen]
    aid2=mask[:,2]
    return {
        "state_count":int(n),
        "greedy_agreement_fraction":float(np.mean(gb==gl)),
        "mean_chosen_q_best":float(selected_margin_b.mean()),
        "mean_chosen_q_latest":float(selected_margin_l.mean()),
        "mean_abs_chosen_q_shift":float(np.abs(selected_margin_l-selected_margin_b).mean()),
        "greedy_hist_best":np.bincount(gb,minlength=25).tolist(),
        "greedy_hist_latest":np.bincount(gl,minlength=25).tolist(),
        "action2_legal_count":int(aid2.sum()),
        "action2_greedy_fraction_when_legal_best":
             float(np.mean(gb[aid2]==2)) if aid2.any() else None,
        "action2_greedy_fraction_when_legal_latest":
             float(np.mean(gl[aid2]==2)) if aid2.any() else None,
        "caveat":"Greedy ranking on observed replay states, NOT counterfactual action value ground truth",
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--safe-zip",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261111)
    p.add_argument("--epochs",type=int,default=35)
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    args=p.parse_args()
    require(args.epochs>=2, "Need at least two supervised epochs")
    require(args.safe_zip.is_file(), "SAFE ZIP not found: "+str(args.safe_zip))
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(4,os.cpu_count() or 2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else (
        "cpu" if args.device=="auto" else args.device)
    require(device!="cuda" or torch.cuda.is_available(), "CUDA unavailable")
    print("OFFLINE_START", "device",device, "zip",args.safe_zip,flush=True)
    manifest,history,ck,trace_contents,baseline,monitors,prior=load_safe(args.safe_zip)
    latest=ck["latest"]
    best=ck["best"]
    r=check_replay(latest,history)
    ns=int(latest["config"]["n_step"])
    scale=float(latest["config"]["reward_scale"])
    R, raw, mc = recover_single_rewards_and_mc(r,ns,scale,history)
    evals = {name:evaluate_checkpoint(model,device) for name,model in ck.items()}
    audit = evals["latest"]
    pos=np.arange(len(raw))%EPISODE_LENGTH
    td_groups=group_diagnostics(audit,raw)
    write_csv(args.out/"td_reward_groups.csv", td_groups)
    plot_distribution(audit,raw,args.out)
    video_map=[z["train_video"] for z in history]
    extreme=[]
    for t in np.argsort(np.abs(raw))[-25:][::-1]:
        extreme.append({
            "replay_index":int(t), "episode":int(t//256), "coding_step":int(t%256),
            "video":video_map[t//256],
            "rate_kbps":int(history[t//256]["train_rate"]),
            "raw_reward_scaled":float(raw[t]),
            "nstep_reward_scaled":float(R[t]),
            "td_error":float(audit["td_error"][t]),
            "terminal":bool(audit["done"][t]),
        })
    write_csv(args.out/"largest_raw_reward_spikes.csv", extreme)
    traces=trace_reward_components(trace_contents,args.out)
    ranking=same_state_action_audit(best,latest,device)
    save_json(args.out/"same_state_action_ranking.json", ranking)

    # Both value predictions are compared against behavior-policy replay
    # returns descriptively; they are not optimal-Q labels.
    report={
        "schema":"d3qn-v4-offline-root-cause-audit/1",
        "source_commit":EXPECTED_SOURCE_COMMIT,
        "test_split_used":False,
        "codec_or_detector_called":False,
        "manifest":manifest,
        "safety":"Trusted SAFE ZIP only; torch.load(weights_only=True)",
        "reward_reconstruction":{
            "passes_20_episode_objective_conservation":True,
            "raw_single_step":safe_stats(raw),
            "stored_nstep":safe_stats(R),
            "mc_remaining":safe_stats(mc),
            "largest_spikes":extreme[:12],
            "coding_step_mod32_zero_mean_abs_raw":float(np.abs(raw[pos%32==0]).mean()),
            "other_mean_abs_raw":float(np.abs(raw[pos%32!=0]).mean()),
            "coding_step_caveat":"Frame index is coding-step, not necessarily display-order POC or an intra picture",
        },
        "checkpoint_diagnostics":{},
        "td_groups":td_groups,
        "validation_traces":traces,
        "same_state_ranking":ranking,
        "original_value_audit_available":prior is not None,
        "val4_best_J":float(manifest["best_objective"]),
        "fixed_roi_minus2_bg0_J":float(np.mean(
            [a["summary"]["objective"] for a in baseline
             if a["method"]=="roi_minus2_bg_zero"])) if baseline else None,
        "limits":[
            "Supervised MC target follows the behavior policy, not optimal Q.",
            "Video-disjoint holdout is drawn from Train16, not external Val4 or Test30.",
            "No counterfactual outcomes for unchosen actions can be inferred.",
            "Huber saturation fraction is diagnostic, not proof of causality.",
            "A successful supervised fit alone does not prove a Bellman bug.",
        ],
    }
    for name, ev in evals.items():
        rname=ck[name]["replay"]
        (_,_,yret)=recover_single_rewards_and_mc(
            rname,int(ck[name]["config"]["n_step"]),
            float(ck[name]["config"]["reward_scale"]),
            history[:ck[name]["steps"]//256])
        report["checkpoint_diagnostics"][name]={
            "steps":int(ck[name]["steps"]),
            "n":int(len(ev["td_error"])),
            "terminal_count":int(ev["done"].sum()),
            "terminal_q":float(ev["q_chosen"][ev["done"]].mean()),
            "terminal_target":float(ev["td_target"][ev["done"]].mean()),
            "terminal_td_abs_mean":float(np.abs(ev["td_error"][ev["done"]]).mean()),
            "nonterminal_td_abs_mean":float(np.abs(ev["td_error"][~ev["done"]]).mean()),
            "huber_fraction_abs_td_gt_1":float(np.mean(np.abs(ev["td_error"])>1)),
            "mean_behavior_mc_minus_q":float((yret-ev["q_chosen"]).mean()),
        }
    save_json(args.out/"audit_summary.json",report)
    print("OFFLINE_Q_AUDIT",json.dumps(report["checkpoint_diagnostics"],indent=2),flush=True)
    print("OFFLINE_RANK_AGREEMENT",round(ranking["greedy_agreement_fraction"],4),flush=True)

    supervised=offline_supervised(
        latest,mc,history,device,args.seed,args.epochs,args.out)
    report["supervised_holdout_metrics"]=supervised["models"]
    report["video_disjoint_split"]=supervised["split"]
    save_json(args.out/"audit_summary.json",report)
    print("OFFLINE_COMPLETE",
          "Val4_J",report["val4_best_J"],
          "trace_count",traces["trace_count"],
          "holdout_R2_MSE",round(supervised["models"]["supervised_mse"]["r2"],4),
          flush=True)


if __name__=="__main__":
    main()
