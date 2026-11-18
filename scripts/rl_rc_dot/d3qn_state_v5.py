"""V5 causal coding-cycle features; V2/V3/V4 observation schemas are immutable.

The period is the encoder-configured 32 coding-step cadence. The feature is
NOT a claim that all these decisions encode an I-picture/CRA, because coding
order and presentation order are not identical.
"""
import math
import numpy as np

GLOBAL_DIM_V4=25
GLOBAL_DIM_V5=28
CODING_CYCLE=32
FEATURE_NAMES=("cycle_sin","cycle_cos","cycle_boundary")

def cycle_features(coding_step, *, cycle_length=CODING_CYCLE):
    if isinstance(coding_step,(bool,np.bool_)) or not isinstance(coding_step,(int,np.integer)) or coding_step < 0:
        raise ValueError("coding_step must be a nonnegative integer")
    if not isinstance(cycle_length,int) or cycle_length<2:
        raise ValueError("cycle_length must be >= 2")
    phase=(int(coding_step)%cycle_length)/float(cycle_length)
    return np.array((math.sin(2*math.pi*phase),
                     math.cos(2*math.pi*phase),
                     float(coding_step%cycle_length==0)),dtype=np.float32)

def extend_observation(obs, coding_step):
    if set(obs)!={"global","maps","legal"}:
        raise ValueError("V5 expects original V4 global, maps and legal")
    glob=np.asarray(obs["global"],dtype=np.float32)
    maps=np.asarray(obs["maps"],dtype=np.float32)
    legal=np.asarray(obs["legal"])
    if (glob.shape!=(GLOBAL_DIM_V4,) or maps.shape!=(5,10,15) or
            legal.shape!=(25,) or legal.dtype!=np.bool_ or
            not legal.any() or not np.isfinite(glob).all() or not np.isfinite(maps).all()):
        raise ValueError("Invalid V4 input state to coding-cycle adapter")
    progress=float(glob[1])
    if not abs(progress-coding_step/256.)<1e-5:
        raise ValueError("V4 progress and native coding-step diverged")
    return {"global":np.concatenate((glob,cycle_features(coding_step))),
            "maps":maps.copy(),"legal":legal.copy()}

def terminal_observation(obs):
    """Extend V4's unobserved terminal placeholder to 28 globals.

    The terminal next-state is never bootstrapped; its padding is not a
    fabricated nonterminal state.
    """
    glob=np.asarray(obs["global"],dtype=np.float32)
    if glob.shape!=(GLOBAL_DIM_V4,):
        raise ValueError("Unexpected V4 terminal placeholder")
    return {"global":np.concatenate((glob,np.zeros(3,dtype=np.float32))),
            "maps":np.asarray(obs["maps"],dtype=np.float32).copy(),
            "legal":np.asarray(obs["legal"],dtype=np.bool_).copy()}
