"""Fixed integer actions -> native QG, final CU, bitstream and independent decode.

Uses synthetic video unless --source-yuv is given. It does not claim task gain.
"""
import argparse,json
from pathlib import Path
import numpy as np
from .codec import NativeEpisode
from .preflight import make_fixture,decode_audit
from .qp_development import QP_BIN_ACTIONS,neutral_action,map_qp_bins
from .action_audit import ActionAudit,aggregate,rounded
from .data import atomic_json,digest

def run(encoder,output,frames=32,decoder=None,ffmpeg='ffmpeg',source=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    synthetic=source is None
    if source is None:source=output/'source.yuv';make_fixture(source,frames)
    records={}
    choices=[('anchor',None)]+[('delta_'+str(d),d) for d in (-2,-1,0,1,2)]+[('spatial',99)]
    for name,delta in choices:
        mode='anchor' if delta is None else 'live'
        a=neutral_action(QP_BIN_ACTIONS)
        if delta==99:
            a[1:]=np.tile([0,0,0,2,4,4,4],5)
        elif delta is not None:a[0]=delta+2
        action=map_qp_bins(a)
        with NativeEpisode(encoder,source,output/name,frames,120,mode=mode,threads=2,trace_qgs=True,trace_cus=True) as env:
            audit=ActionAudit(env.output);samples=[]
            while env.current is not None:
                post,_=env.step(action if delta is not None else None)
                samples.append(audit.consume(post,action))
            decoded=decode_audit(env.output,ffmpeg,frames,decoder) if decoder else None
            records[name]={'requested_categories':a.tolist(),'requested_delta_min':float(action.min()),
                'requested_delta_max':float(action.max()),'actual_kbps':env.bits*30/frames/1000,
                'mean_qp':float(np.mean([r['mean_qp'] for r in env.posts])),
                'first_poc_mean_qp':env.posts[0]['mean_qp'],'psnr_y':float(np.mean([r['psnr'] for r in env.posts])),
                'bitstream_sha256':digest(env.output/'stream.266'),'decoded_parity':decoder is not None,
                'decoded_sha256':decoded,'action_audit':audit.summary(),
                'detection':None,'detection_note':'Synthetic integration fixture; no car detector quality claim'}
            qgs=[json.loads(s) for s in (env.output/'qgs.jsonl').read_text().splitlines()]
            mapping=[]
            for r in qgs:
                avg=aggregate(action,r)
                mapping.append({**r,'requested_map_delta_min':float(action.min()),'requested_map_delta_max':float(action.max()),
                    'aggregated_delta':avg,'rounded_delta':rounded(avg),
                    'effective_candidate_delta':r['requested_qp']-r['native_qp'],
                    'note':'requested_qp is the RDO candidate QP; final CU QPs are separately logged'})
            atomic_json(env.output/'ACTION_MAPPING.json',mapping)
        atomic_json(output/'ACTION_SENSITIVITY.json',records)
        print('INTEGER_CODEC_ACTION',name,json.dumps({k:v for k,v in records[name].items() if k!='action_audit'}),flush=True)
    anchor=records['anchor']
    assert records['delta_0']['bitstream_sha256']==anchor['bitstream_sha256'],'No-op changed the native encoder'
    for d in (-2,-1,1,2):
        r=records['delta_'+str(d)]
        assert r['bitstream_sha256']!=anchor['bitstream_sha256'],'Fixed action has no bitstream response'
        assert r['action_audit']['effective_candidate_delta_counts'].get(d,0)>0,'No nonzero QG candidate effect'
        assert r['first_poc_mean_qp']!=anchor['first_poc_mean_qp'],'No actual final-CU QP effect on first picture'
    assert records['spatial']['bitstream_sha256']!=anchor['bitstream_sha256']
    result={'gate':'INTEGER_ACTION_MAPPING_AND_BITSTREAM_PASS' if not decoder else 'INTEGER_ACTION_MAPPING_BITSTREAM_DECODE_PASS',
        'action_contract':'qpdev1-integer-action/1','decoder_sha256':digest(decoder) if decoder else None,
        'encoder_sha256':digest(encoder),'synthetic':synthetic,'frames':frames,'runs':records,
        'quality_claim':'No useful policy or RDT improvement established by sensitivity alone'}
    atomic_json(output/'ACTION_PREFLIGHT.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoder',type=Path,required=True);p.add_argument('--decoder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=32)
    p.add_argument('--source-yuv',type=Path);a=p.parse_args()
    print(run(a.encoder,a.output,a.frames,a.decoder,source=a.source_yuv)['gate'],flush=True)
if __name__=='__main__':main()
