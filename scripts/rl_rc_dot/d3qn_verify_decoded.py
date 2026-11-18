"""Independent VVdeC parity check for D3QN; no PPO/SB3 imports."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from .method import FRAME_BYTES


def verify_decoded(stream,trajectory,frames,ffmpeg,decoder=None):
    stream=Path(stream)
    trajectory=Path(trajectory)
    decoded=stream.with_suffix('.decoded.yuv')
    if decoder:
        command=[str(decoder),'-b',str(stream),'-o',str(decoded),'--threads','2']
    else:
        command=[str(ffmpeg),'-hide_banner','-nostdin','-loglevel','error','-y',
                 '-i',str(stream),'-pix_fmt','yuv420p10le','-fps_mode','passthrough',
                 '-f','rawvideo',str(decoded)]
    try:
        subprocess.run(command,check=True,stdout=subprocess.DEVNULL,timeout=1200)
        with gzip.open(trajectory,'rt',encoding='utf-8') as reader:
            rows=[json.loads(line) for line in reader]
        hashes={row['poc']:row['reconstruction_sha256'] for row in rows}
        if (len(rows)!=frames or set(hashes)!=set(range(frames)) or
                decoded.stat().st_size!=frames*FRAME_BYTES*2):
            raise ValueError('Decoded output and scored POCs disagree')
        with decoded.open('rb') as source:
            for poc in range(frames):
                if hashlib.sha256(source.read(FRAME_BYTES*2)).hexdigest()!=hashes[poc]:
                    raise ValueError(f'VVdeC and scored reconstruction differ at POC {poc}')
    finally:
        decoded.unlink(missing_ok=True)
