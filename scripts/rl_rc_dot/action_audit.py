"""Observed candidate-QG effects and winning-CU QPs; never infer one from the other."""
import json,math
from collections import Counter
from pathlib import Path
import numpy as np

def aggregate(action,row):
    """Area-weighted float before the native std::lround, including partial cells."""
    a=np.asarray(action,dtype=np.float64).reshape(20,30)
    x,y,w,h=(int(row[k]) for k in ('x','y','w','h'))
    right,bottom=min(480,x+w),min(320,y+h)
    if not 0<=x<right<=480 or not 0<=y<bottom<=320:raise ValueError('Invalid QG area')
    total=0.
    for iy in range(y//16,(bottom-1)//16+1):
        for ix in range(x//16,(right-1)//16+1):
            area=(min(right,(ix+1)*16)-max(x,ix*16))*(min(bottom,(iy+1)*16)-max(y,iy*16))
            total+=area*a[iy,ix]
    return total/((right-x)*(bottom-y))

def rounded(value):return math.floor(value+.5) if value>=0 else math.ceil(value-.5)

class ActionAudit:
    def __init__(self,folder):
        self.folder=Path(folder);self.offsets={};self.total_candidates=0
        self.pending={};self.completed={}
        self.effective=Counter();self.rounded=Counter();self.final_qp_area=Counter()
        self.noop_frames=0;self.frames=0
    def rows(self,name):
        p=self.folder/name
        if not p.exists():raise ValueError('Required native action trace missing: '+str(p))
        with p.open('rb') as f:
            offset=self.offsets.get(name,0)
            # The encoder can already be writing a later POC. Read a bounded
            # snapshot and retain a partial final row for the next consume.
            end=f.seek(0,2)
            if end<offset:raise ValueError('Native action trace was truncated')
            f.seek(offset)
            while f.tell()<end:
                start=f.tell();line=f.readline(end-start)
                if not line.endswith(b'\n'):break
                row=json.loads(line)  # Malformed complete rows remain errors.
                self.offsets[name]=f.tell()
                yield row
    def consume(self,post,action):
        cache={};rounded_hist=Counter();effective=Counter();aggregated=[];clipped=0;count=0
        def picture_rows(name):
            pending=self.pending.setdefault(name,{})
            completed=self.completed.setdefault(name,set())
            for row in self.rows(name):
                if row['poc'] in completed:raise ValueError('Native trace arrived after its picture was finalized')
                pending.setdefault(row['poc'],[]).append(row)
            completed.add(post['poc'])
            return pending.pop(post['poc'],[])
        for r in picture_rows('qgs.jsonl'):
            key=tuple(r[k] for k in ('x','y','w','h'))
            if key not in cache:cache[key]=aggregate(action,r)
            avg=cache[key];delta=rounded(avg);expected=max(-12,min(63,r['native_qp']+delta))
            if expected!=r['requested_qp']:raise ValueError('Native QG QP differs from pooled/rounded action')
            lam=r['ctu_lambda']*2.**((r['requested_qp']-r['ctu_qp'])/3.)
            if not math.isclose(lam,r['rd_lambda'],rel_tol=1e-10):raise ValueError('RDO lambda disagrees with requested QP')
            rounded_hist[delta]+=1;effective[r['requested_qp']-r['native_qp']]+=1
            clipped+=int(expected!=r['native_qp']+delta);aggregated.append(avg);count+=1
        area=Counter();cu_count=0
        for r in picture_rows('cus.jsonl'):
            area[r['qp']]+=r['w']*r['h'];cu_count+=1
        if sum(area.values())!=480*320:raise ValueError('Missing winning CU area')
        if not math.isclose(sum(k*v for k,v in area.items())/(480*320),post['mean_qp'],abs_tol=1e-8):
            raise ValueError('Final CU QP histogram differs from native post')
        inactive=bool(np.abs(action).max()<.5)
        if not count and not inactive:raise ValueError('Active action has no QG candidate evidence')
        self.frames+=1;self.noop_frames+=inactive;self.total_candidates+=count
        self.rounded.update(rounded_hist);self.effective.update(effective);self.final_qp_area.update(area)
        return {'scope':'RDO candidate QGs (may repeat geometry), separately from winning CU area',
            'candidate_qgs':count,'aggregated_delta_min':min(aggregated) if aggregated else None,
            'aggregated_delta_max':max(aggregated) if aggregated else None,
            'rounded_delta_counts':dict(rounded_hist),'effective_candidate_delta_counts':dict(effective),
            'candidate_zero_fraction':effective[0]/count if count else None,
            'clipped_candidates':clipped,'inactive_action_map':inactive,
            'final_cu_qp_pixel_counts':dict(area),'final_cus':cu_count}
    def summary(self):
        return {'frames':self.frames,'inactive_map_frames':self.noop_frames,'candidate_qgs':self.total_candidates,
            'rounded_delta_counts':dict(self.rounded),'effective_candidate_delta_counts':dict(self.effective),
            'candidate_zero_fraction':self.effective[0]/self.total_candidates if self.total_candidates else None,
            'final_cu_qp_pixel_counts':dict(self.final_qp_area),
            'scope':'All observed RDO candidate decisions; candidate counts are not independent winning QGs. Final QP histogram is pixel weighted.'}
