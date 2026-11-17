"""Honest paired Validation report and native QP evidence for QP development."""
import csv,gzip,json
from pathlib import Path
import numpy as np
from .data import atomic_json

def summarize(output):
    root=Path(output);out=root/'reports';out.mkdir(exist_ok=True)
    paths=list((root/'validations').glob('*.json'))
    validations=sorted([json.loads(p.read_text()) for p in paths],key=lambda r:r['completed_rollouts'])
    if not validations:
        result={'status':'NO_COMPLETE_VALIDATION','advance_to_long_training':False}
        atomic_json(out/'development_summary.json',result);return result
    progress=json.loads((root/'progress.json').read_text());ref=progress.get('best')
    best=json.loads((root/ref['validation']).read_text()) if ref else validations[-1]
    rows=[]
    for p in best['points']:
        a,b=p['anchor'],p['agent']
        rows.append({'video_id':p['video_id'],'rate_kbps':p['rate_kbps'],
            'anchor_f1':a['f1'],'agent_f1':b['f1'],'delta_f1':b['f1']-a['f1'],
            'anchor_mean_frame_f1':a['mean_frame_f1'],'agent_mean_frame_f1':b['mean_frame_f1'],
            'delta_objective':b['objective_score']-a['objective_score'],
            'anchor_actual_kbps':a['actual_kbps'],'agent_actual_kbps':b['actual_kbps'],
            'rate_ratio':b['actual_kbps']/a['actual_kbps']-1,
            'anchor_mean_qp':a['mean_qp'],'agent_mean_qp':b['mean_qp'],
            'requested_abs_delta':b['requested_abs_delta'],'requested_spatial_std':b['requested_spatial_std'],
            'requested_nonzero_fraction':b['requested_nonzero_fraction']})
    with (out/'paired_validation.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    initial=next((v for v in validations if v['completed_rollouts']==0),None)
    delta_f1=float(np.mean([r['delta_f1'] for r in rows]))
    ratio=float(np.mean([r['rate_ratio'] for r in rows]));obj=float(np.mean([r['delta_objective'] for r in rows]))
    nonzero=float(np.mean([r['requested_nonzero_fraction'] for r in rows]))
    gain=best['score']-initial['score'] if initial else None
    gates={'trained_checkpoint_selected':best['completed_rollouts']>0,
           'objective_above_initial':gain is not None and gain>0,
           'objective_above_anchor':obj>0,
           'macro_clip_f1_gain_at_least_0_2pp':delta_f1>=.002,
           'macro_rate_increase_at_most_1percent':ratio<=.01,
           'p95_rate_increase_at_most_5percent':float(np.percentile([r['rate_ratio'] for r in rows],95))<=.05,
           'nonzero_requested_qp_fraction_above_1percent':nonzero>.01}
    if 'random_score' in best:
        gates['objective_above_random']=best['score']>best['random_score']
        gates['all_pairs_actual_rate_error_within_5percent']=all(abs(p['agent']['actual_kbps']/p['rate_kbps']-1)<=.05 for p in best['points'])
        gates['all_pairs_psnr_drop_at_most_0_5db']=all(p['agent']['psnr_y']>=p['anchor']['psnr_y']-.5 for p in best['points'])
    result={'status':'VALIDATION_SCREEN_ONLY','advance_to_long_training':all(gates.values()),
            'screening_gates':gates,'best_update':best['completed_rollouts'],'best_timesteps':best['num_timesteps'],
            'initial_score':initial['score'] if initial else None,'best_score':best['score'],
            'delta_score_vs_initial':gain,'macro_delta_f1_pp':100*delta_f1,
            'macro_rate_increase_percent':100*ratio,'macro_delta_objective':obj,
            'requested_nonzero_fraction':nonzero,'pairs':len(rows),
            'random_score':best.get('random_score'),
            'interpretation':'Predeclared heuristic screening; not significance, AP improvement, Test results or multi-seed evidence. Compare with corrected paper control at equal budget.'}
    atomic_json(out/'development_summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    xs=[v['num_timesteps'] for v in validations]
    axes[0].plot(xs,[v['score'] for v in validations],'o-',label='agent objective')
    axes[0].plot(xs,[v['anchor_score'] for v in validations],'--',label='native anchor');axes[0].legend()
    axes[0].set(xlabel='Training interactions',ylabel='Validation objective')
    axes[1].plot(xs,[100*v['macro_delta_f1'] for v in validations],'o-');axes[1].axhline(0,color='gray')
    axes[1].set(xlabel='Training interactions',ylabel='Clip F1 gain (percentage points)')
    axes[2].scatter([r['rate_ratio']*100 for r in rows],[r['delta_f1']*100 for r in rows])
    axes[2].axhline(0,color='gray');axes[2].axvline(0,color='gray')
    axes[2].set(xlabel='Actual bitrate increase (%)',ylabel='Clip F1 gain (percentage points)')
    fig.tight_layout();fig.savefig(out/'validation_learning.png',dpi=160);plt.close(fig)
    # Read final-CU traces, not candidate-QG or requested maps masquerading as QP.
    candidates=[]
    for p in (root/'validation_media'/str(best['completed_rollouts'])).glob('*/episode_summary.json'):
        s=json.loads(p.read_text())
        if s['mode']=='live' and s.get('evaluation_label','agent')=='agent':candidates.append((p,s))
    if candidates:
        p,s=sorted(candidates,key=lambda v:(v[1]['video_id'],v[1]['rate_kbps']))[0]
        with gzip.open(p.parent/'trajectory.jsonl.gz','rt',encoding='utf-8') as f:traj=[json.loads(line) for line in f]
        with gzip.open(p.parent/'codec/cus.jsonl.gz','rt',encoding='utf-8') as f:cus=[json.loads(line) for line in f]
        by_poc={r['poc']:r for r in traj}
        selected=[0,len(traj)//2,len(traj)-1];fig,axes=plt.subplots(2,3,figsize=(11,5))
        for col,poc in enumerate(selected):
            t=by_poc[poc];surface=np.full((320,480),np.nan)
            for cu in cus:
                if cu['poc']==t['poc']:surface[cu['y']:cu['y']+cu['h'],cu['x']:cu['x']+cu['w']]=cu['qp']
            if not np.isfinite(surface).all():raise ValueError('Final CU QP coverage missing')
            a=axes[0,col].imshow(np.asarray(t['consumed_action']).reshape(20,30),vmin=-6,vmax=6,cmap='coolwarm')
            b=axes[1,col].imshow(surface,vmin=0,vmax=63,cmap='viridis')
            axes[0,col].set_title('Requested delta, POC '+str(t['poc']));axes[1,col].set_title('Actual final CU QP')
            fig.colorbar(a,ax=axes[0,col]);fig.colorbar(b,ax=axes[1,col])
        fig.suptitle(s['video_id']+' / '+str(s['rate_kbps'])+' kbps; fixed source POC samples')
        fig.tight_layout();fig.savefig(out/'requested_and_actual_qp.png',dpi=160);plt.close(fig)
        result['qp_evidence_status']='FINAL_CU_QP_AT_THREE_FIXED_POCS; requested map is a different quantity'
    else:
        result['qp_evidence_status']='NO_BEST_CHECKPOINT_MEDIA_IN_THIS_SESSION; retain earlier SAFE exports after resume'
    atomic_json(out/'development_summary.json',result)
    return result
