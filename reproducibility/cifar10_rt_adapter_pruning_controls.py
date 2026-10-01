import argparse, copy, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[0]
REPO = ROOT.parent if ROOT.name == 'reproducibility' else ROOT
sys.path.insert(0, str(REPO / 'experiments'))

from cifar10_rt_modulated import (
    B, DEVICE, FixedCorruptCIFAR, seed_all, nparams, evaluate, train
)
from cifar10_rt_adapter_pruning import WideRTAdapterCNN

RANKS = [1, 2, 4, 8, 16]
CRITERIA = ['product', 'u_only', 'v_only']
RANDOM_REPS = 5

def unit_scores(adapter, band, criterion):
    V = adapter.down[band].weight[:, :, 0, 0]
    U = adapter.up[band].weight[:, :, 0, 0]
    un = U.norm(dim=0); vn = V.norm(dim=1)
    if criterion == 'product': return un * vn
    if criterion == 'u_only': return un
    if criterion == 'v_only': return vn
    raise ValueError(criterion)

def masks_for_model(model, criterion, r, seed, rep=0):
    masks = []
    for li, adapter in enumerate(model.adapters):
        layer_masks = []
        for j in range(B):
            n = adapter.rank
            if r >= n:
                mask = torch.ones(n, dtype=torch.bool)
            elif criterion == 'random':
                g = torch.Generator().manual_seed(1000003 * seed + 10007 * li + 997 * j + 53 * rep + r)
                idx = torch.randperm(n, generator=g)[:r]
                mask = torch.zeros(n, dtype=torch.bool); mask[idx] = True
            else:
                sc = unit_scores(adapter, j, criterion).detach().cpu()
                idx = torch.topk(sc, k=r, largest=True).indices
                mask = torch.zeros(n, dtype=torch.bool); mask[idx] = True
            layer_masks.append(mask)
        masks.append(layer_masks)
    return masks

def apply_masks(model, masks):
    with torch.no_grad():
        for li, adapter in enumerate(model.adapters):
            for j in range(B):
                mask = masks[li][j].to(adapter.up[j].weight.device)
                adapter.up[j].weight[:, ~mask, :, :] = 0.0
    return model

@torch.no_grad()
def logit_deviation(ref, other, dl):
    ref.eval(); other.eval()
    n = 0; sq = 0.0; mx = 0.0
    for x, y, k, s in dl:
        x = x.to(DEVICE); k = k.to(DEVICE); s = s.to(DEVICE)
        d = ref(x, k, s) - other(x, k, s)
        sq += (d*d).sum().item(); n += d.numel()
        mx = max(mx, d.abs().max().item())
    return (sq/n)**0.5, mx

@torch.no_grad()
def local_bound_check(model, dl, ranks=(1,2,4,8)):
    model.eval()
    accum = {}
    for li, adapter in enumerate(model.adapters):
        for r in ranks:
            accum[(li, r)] = {'actual_sum':0.0, 'bound_sum':0.0, 'ratio_sum':0.0,
                              'max_ratio':0.0, 'count':0, 'violations':0}
    for xb, y, kb, sb in dl:
        x = xb.to(DEVICE); k = kb.to(DEVICE); s = sb.to(DEVICE)
        for li in range(5):
            x = model.stem.block(x, li)
            x = model.films[li](x, k, s)
            x = F.relu(x)
            adapter = model.adapters[li]
            full = adapter(x, k, s)
            for r in ranks:
                masks = masks_for_model(model, 'product', r, seed=0, rep=0)[li]
                pruned = x.clone()
                bounds = torch.zeros(len(x), device=x.device, dtype=x.dtype)
                for j in range(B):
                    q = (k == j)
                    if not q.any(): continue
                    mask = masks[j].to(x.device)
                    V = adapter.down[j].weight[:, :, 0, 0]
                    U = adapter.up[j].weight[:, :, 0, 0]
                    z = torch.einsum('rc,bchw->brhw', V, x[q])
                    z = F.relu(z); z[:, ~mask, :, :] = 0.0
                    corr = torch.einsum('cr,brhw->bchw', U, z)
                    scale = adapter.a[j] + adapter.b[j] * (s[q] - .5)
                    pruned[q] = x[q] + scale[:, None, None, None] * corr
                    tau = U.norm(dim=0) * V.norm(dim=1)
                    omitted = tau[~mask].sum()
                    xnorm = x[q].flatten(1).norm(dim=1)
                    bounds[q] = scale.abs() * xnorm * omitted
                actual = (full - pruned).flatten(1).norm(dim=1)
                ratio = actual / bounds.clamp_min(1e-12)
                d = accum[(li, r)]
                d['actual_sum'] += actual.sum().item()
                d['bound_sum'] += bounds.sum().item()
                valid = bounds > 1e-10
                if valid.any():
                    rv = ratio[valid]
                    d['ratio_sum'] += rv.sum().item()
                    d['max_ratio'] = max(d['max_ratio'], rv.max().item())
                    d['count'] += int(valid.sum().item())
                    d['violations'] += int((actual[valid] > bounds[valid] * (1+1e-5)).sum().item())
            x = full
            if li in (1,3): x = model.stem.pool(x)
    rows=[]
    for (li,r), d in accum.items():
        c=max(d['count'],1)
        rows.append([li,r,d['actual_sum']/c,d['bound_sum']/c,d['ratio_sum']/c,
                     d['max_ratio'],d['violations'],d['count']])
    return pd.DataFrame(rows, columns=['layer','r','mean_actual_adapter_l2','mean_bound',
                                       'mean_actual_over_bound','max_actual_over_bound',
                                       'bound_violations','n'])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--width', type=int, default=31)
    ap.add_argument('--out', default='pruning_controls')
    args=ap.parse_args(); seed_all(args.seed)
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tr=FixedCorruptCIFAR('data','train'); va=FixedCorruptCIFAR('data','val')
    ts=FixedCorruptCIFAR('data','test'); tu=FixedCorruptCIFAR('data','test',True)
    g=torch.Generator().manual_seed(9000+args.seed)
    tl=DataLoader(tr,128,shuffle=True,generator=g,num_workers=2)
    vl=DataLoader(va,256,num_workers=2); sl=DataLoader(ts,256,num_workers=2); ul=DataLoader(tu,256,num_workers=2)

    model=WideRTAdapterCNN(args.width,args.rank)
    np_=nparams(model); model,best=train(model,tl,vl,18); model.to(DEVICE)

    rows=[]
    for split,dl in [('seen',sl),('unseen_severity',ul)]:
        z=evaluate(model,dl)
        rows.append(['full',16,-1,args.seed,split,z['accuracy'],z['nll'],0.0,0.0,np_,best])

    for criterion in CRITERIA + ['random']:
        reps = range(RANDOM_REPS) if criterion == 'random' else [-1]
        for rep in reps:
            for r in [1,2,4,8]:
                masks=masks_for_model(model,criterion,r,args.seed,rep=max(rep,0))
                pr=apply_masks(copy.deepcopy(model),masks)
                zs=evaluate(pr,sl); zu=evaluate(pr,ul)
                rms,mx=logit_deviation(model,pr,ul)
                rows.append([criterion,r,rep,args.seed,'seen',zs['accuracy'],zs['nll'],0.0,0.0,np_,best])
                rows.append([criterion,r,rep,args.seed,'unseen_severity',zu['accuracy'],zu['nll'],rms,mx,np_,best])

    pd.DataFrame(rows, columns=['criterion','r','rep','seed','split','accuracy','nll',
                                'logit_rmse_vs_full','logit_max_vs_full','params','val_best']).to_csv(
        out/f'controls_{args.seed:02d}.csv', index=False)

    bound=local_bound_check(model,ul); bound.insert(0,'seed',args.seed)
    bound.to_csv(out/f'bound_check_{args.seed:02d}.csv',index=False)

    spectra=[]
    for li,ad in enumerate(model.adapters):
        for j in range(B):
            V=ad.down[j].weight[:,:,0,0].detach().cpu()
            U=ad.up[j].weight[:,:,0,0].detach().cpu()
            un=U.norm(dim=0); vn=V.norm(dim=1); tau=un*vn
            for kk in range(len(tau)):
                spectra.append([args.seed,li,j,kk+1,float(tau[kk]),float(un[kk]),float(vn[kk])])
    pd.DataFrame(spectra,columns=['seed','layer','band','unit','tau','u_norm','v_norm']).to_csv(
        out/f'scores_{args.seed:02d}.csv',index=False)

if __name__=='__main__':
    main()
