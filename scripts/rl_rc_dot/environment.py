"""One PPO transition per encoded picture with the native frame boundary."""

import json
import gzip
import hashlib
from pathlib import Path
import uuid
import shutil

import gymnasium as gym
import numpy as np

from .codec import NativeEpisode
from .method import FRAME_BYTES, action_map, observation, score, auxiliary_supervision_mask, training_reward
from .model import ACTIONS, OBSERVATIONS
from .data import atomic_json
from .qp_development import (check_profile,augmented,DEV_ACTIONS,DEV_OBSERVATIONS,
                             map_action,state as development_state,reward_increment,task_f1,clip_objective,retention_supervision)


class VVenCEnv(gym.Env):
    observation_space = OBSERVATIONS
    action_space = ACTIONS

    def __init__(self, catalog, detector, encoder, output, *, split="train", frame_limit=0, threads=2,
                 fixed_video=None, fixed_rate=None, mode="live", retain_media=False, random_windows=False,
                 agent_profile="paper"):
        self.agent_profile=check_profile(agent_profile)
        self.observation_space=DEV_OBSERVATIONS if augmented(agent_profile) else OBSERVATIONS
        self.action_space=DEV_ACTIONS if agent_profile=='qpdev1' else ACTIONS
        self.catalog, self.detector = catalog, detector
        self.encoder, self.output = Path(encoder), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown dataset split")
        self.ids = catalog.selected[split] if hasattr(catalog,"selected") else catalog.lock["splits"][split]
        if not self.ids:raise ValueError("Selected split has no videos")
        if fixed_video is not None and fixed_video not in self.ids:
            raise ValueError("Requested video is outside the selected split")
        self.split, self.frame_limit, self.threads = split, frame_limit, threads
        self.fixed_video, self.fixed_rate, self.mode = fixed_video, fixed_rate, mode
        self.retain_media = retain_media
        self.random_windows = bool(random_windows)
        if self.random_windows and (split!="train" or frame_limit!=256 or fixed_video is not None):
            raise ValueError("Random 256-frame windows apply only to mini training")
        self.native = None
        self.source = None
        self.trajectory = None
        self.episodes_started=0

    def rng_state(self):
        return {"generator":self.np_random.bit_generator.state,"episodes_started":self.episodes_started}

    def seed_sampler(self,seed):
        self._np_random=np.random.default_rng(seed)

    def suspend_episode(self):
        self._close_episode()

    def restore_rng_state(self,state):
        self.np_random.bit_generator.state=state["generator"]
        self.episodes_started=state["episodes_started"]

    def _close_episode(self):
        completed = self.native is not None and self.native.current is None
        source_path = Path(self.source.filename) if self.source is not None else None
        codec = self.native.output if self.native else None
        if self.native:
            self.native.close()
            self.native = None
        if self.source is not None:
            self.source._mmap.close()
            self.source = None
        if self.trajectory:
            self.trajectory.close()
            self.trajectory = None
        if codec:
            for name in ('events','cus'):
                if (codec/(name+'.jsonl')).is_file():
                    with (codec/(name+'.jsonl')).open('rb') as src,gzip.open(codec/(name+'.jsonl.gz'),'wb') as dst:
                        if name=='cus':
                            sampled={0,self.frames//2,self.frames-1}
                            for line in src:
                                if line.endswith(b'\n') and json.loads(line)['poc'] in sampled:dst.write(line)
                        else:shutil.copyfileobj(src,dst)
                    (codec/(name+'.jsonl')).unlink()
        if source_path and not self.retain_media:
            source_path.unlink(missing_ok=True)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._close_episode()
        self.video_id = self.fixed_video or self.ids[int(self.np_random.integers(len(self.ids)))]
        self.rate = self.fixed_rate or int(self.np_random.integers(50, 201))
        self.episodes_started+=1
        folder = self.output / f"{self.video_id}_{self.rate}_{uuid.uuid4().hex[:8]}"
        folder.mkdir()
        path = folder / "source.yuv"
        self.start_frame=int(self.np_random.choice([0,128,256,384,512])) if self.random_windows else 0
        self.frames=(self.catalog.prepare(self.video_id,path,self.frame_limit,start_frame=self.start_frame)
                     if self.start_frame else self.catalog.prepare(self.video_id,path,self.frame_limit))
        if self.frame_limit and self.frames!=self.frame_limit:
            raise ValueError(f"Source window truncated: {self.video_id}, {self.frames} != {self.frame_limit}")
        self.source = np.memmap(path, mode="r", dtype=np.uint8, shape=(self.frames, FRAME_BYTES))
        self.trajectory = gzip.open(folder / "trajectory.jsonl.gz", "wt", encoding="utf-8")
        self.native = NativeEpisode(self.encoder, path, folder / "codec", self.frames, self.rate,
                                    mode=self.mode, threads=self.threads,
                                    trace_cus=self.agent_profile!='paper' and self.split!='train')
        self.decisions = 0
        self.rewards = []
        self.precisions, self.recalls = [], []
        self.counts = np.zeros(3, dtype=np.int64)
        self.rate_cost=0.0;self.frame_f1=[];self.requested=[]
        self.previous_delta=np.zeros((20,30));self.previous_au_bits=0
        return self._observation(self.native.current), {"video_id": self.video_id, "rate_kbps": self.rate,
                                                               "start_frame": self.start_frame}

    def _observation(self,event):
        if augmented(self.agent_profile):
            return development_state(event,self.frames,self.rate*1000,self.previous_delta,self.previous_au_bits)
        return observation(event,self.frames)

    def step(self, action):
        if self.native is None or self.native.current is None:
            raise RuntimeError("Call reset before taking an action")
        current = self.native.current
        action=np.asarray(action)
        if action.shape!=self.action_space.shape or not np.isfinite(action).all():raise ValueError('Invalid policy action')
        delta=map_action(action)
        post, next_state = self.native.step(delta)
        poc = post["poc"]
        raw = self.detector(self.source[poc])
        reconstructed_bytes = (self.native.recon / f"{poc}.yuv").read_bytes()
        reconstructed = np.frombuffer(reconstructed_bytes, dtype="<u2")
        reconstructed = np.minimum(255, (reconstructed.astype(np.uint32) + 2) >> 2).astype(np.uint8)
        prediction = self.detector(reconstructed)
        self.decisions += 1
        reward, block, metrics = score(raw, prediction, self.native.bits, self.decisions, self.rate * 1000)
        paper_reward=reward
        self.frame_f1.append(task_f1(metrics))
        if self.agent_profile!='paper':
            reward,self.rate_cost=reward_increment(metrics,self.native.bits,self.decisions,self.frames,
                                                   self.rate*1000,self.rate_cost)
        self.previous_delta=delta.copy();self.previous_au_bits=post['au_bits']
        self.requested.append({'mean':float(delta.mean()),'spatial_std':float(delta.std()),
                               'abs_mean':float(np.abs(delta).mean()),'nonzero_fraction':float((np.abs(delta)>=.5).mean())})
        mask = auxiliary_supervision_mask(raw, prediction)
        if self.agent_profile!='paper':block,mask=retention_supervision(raw,prediction)
        info = {"block_reward": block, "block_mask": mask, "paper_reward": paper_reward,
                'objective_reward':reward,'agent_profile':self.agent_profile,
                "video_id": self.video_id, "poc": poc, "rate_kbps": self.rate, "start_frame": self.start_frame,
                "consumed_action":np.asarray(post.get("consumed_action",np.zeros(600)),dtype=float).ravel().tolist(),**metrics}
        self.rewards.append(reward)
        self.precisions.append(metrics["precision"])
        self.recalls.append(metrics["recall"])
        self.counts += [metrics["tp"], metrics["fp"], metrics["fn"]]
        row = {"poc": poc, "step": self.decisions, "au_bits": post["au_bits"], "reward": reward,
               "precision": metrics["precision"], "recall": metrics["recall"],
               "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
               "raw_boxes": raw.tolist(), "reconstructed_boxes": prediction.tolist(),
               "reconstruction_sha256": hashlib.sha256(reconstructed_bytes).hexdigest(),
               "gaussian_action": np.asarray(action).tolist(), "history_poc": current["history_poc"],
               "state_sha256":hashlib.sha256(json.dumps(current,sort_keys=True,allow_nan=False).encode()).hexdigest(),
               "consumed_action":info["consumed_action"],"post":post}
        self.trajectory.write(json.dumps(row, allow_nan=False) + "\n")
        self.trajectory.flush()
        if not self.retain_media:
            (self.native.recon / f"{poc}.yuv").unlink()
            action_file = self.native.actions / f"{poc}.txt"
            if action_file.exists():
                action_file.unlink()
        terminated = next_state is None
        if terminated:
            tp, fp, fn = map(int, self.counts)
            info["episode"] = {"r": float(sum(self.rewards)), "l": self.decisions,
                               "start_frame": self.start_frame,
                               "mean_precision": float(np.mean(self.precisions)),
                               "mean_recall": float(np.mean(self.recalls)),
                               "mean_reward": float(np.mean(self.rewards)),
                               "tp": tp, "fp": fp, "fn": fn,
                               "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                               "psnr_y": float(np.mean([row["psnr"] for row in self.native.posts])),
                               "mean_qp": float(np.mean([row["mean_qp"] for row in self.native.posts])),
                               "actual_kbps": self.native.bits * 30 / self.frames / 1000}
            info['episode'].update(mean_frame_f1=float(np.mean(self.frame_f1)),
                objective_score=clip_objective(float(np.mean(self.frame_f1)),self.native.bits*30/self.frames,self.rate*1000),
                requested_delta_mean=float(np.mean([r['mean'] for r in self.requested])),
                requested_spatial_std=float(np.mean([r['spatial_std'] for r in self.requested])),
                requested_abs_delta=float(np.mean([r['abs_mean'] for r in self.requested])),
                requested_nonzero_fraction=float(np.mean([r['nonzero_fraction'] for r in self.requested])))
            if self.agent_profile!='paper' and not np.isclose(sum(self.rewards),info['episode']['objective_score'],atol=1e-8):
                raise ValueError('Training return differs from the final validation objective')
            atomic_json(self.native.output.parent/'episode_summary.json',
                {'video_id':self.video_id,'rate_kbps':self.rate,'mode':self.mode,
                 'agent_profile':self.agent_profile,'final_cu_trace_pocs':[0,self.frames//2,self.frames-1],**info['episode']})
            obs = {name: np.zeros(space.shape, dtype=np.float32) for name, space in self.observation_space.spaces.items()}
        else:
            obs = self._observation(next_state)
        return obs, training_reward(reward) if self.agent_profile=='paper' else reward, terminated, False, info

    def close(self):
        self._close_episode()
        self.detector.close()
