import argparse, copy, sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).parent))
from cifar10_rt_modulated import (
    B, DEVICE, FixedCorruptCIFAR, ConvStem, BandFiLM,
    seed_all, nparams, evaluate, train
)

class WideBandAdapter(nn.Module):
    def __init__(self,c,rank):
        super().__init__()
        self.rank=min(rank,c)
        self.down=nn.ModuleList([nn.Conv2d(c,self.rank,1,bias=False) for _ in range(B)])
        self.up=nn.ModuleList([nn.Conv2d(self.rank,c,1,bias=False) for _ in range(B)])
        for u in self.up: nn.init.zeros_(u.weight)
        self.a=nn.Parameter(torch.ones(B))
        self.b=nn.Parameter(torch.zeros(B))
    def forward(self,x,k,s):
        out=x.clone()
        for j in range(B):
            q=k==j
            if q.any():
                scale=(self.a[j]+self.b[j]*(s[q]-.5))[:,None,None,None]
                out[q]=x[q]+scale*self.up[j](F.relu(self.down[j](x[q])))
        return out
    def scores(self,j):
        V=self.down[j].weight[:,:,0,0]
        U=self.up[j].weight[:,:,0,0]
        return U.norm(dim=0)*V.norm(dim=1)
    def prune_to(self,r):
        with torch.no_grad():
            for j in range(B):
                sc=self.scores(j)
                keep=torch.topk(sc,k=min(r,len(sc)),largest=True).indices
                mask=torch.zeros(len(sc),dtype=torch.bool,device=sc.device); mask[keep]=True
                self.up[j].weight[:,~mask,:,:]=0.
        return self

class WideRTAdapterCNN(nn.Module):
    def __init__(self,w=31,rank=16):
        super().__init__()
        self.stem=ConvStem(w)
        self.films=nn.ModuleList([BandFiLM(c) for c in self.stem.channels])
        self.adapters=nn.ModuleList([WideBandAdapter(c,rank) for c in self.stem.channels])
        self.head=nn.Linear(4*w,10)
    def forward(self,x,k,s):
        for i in range(5):
            x=self.stem.block(x,i)
            x=self.films[i](x,k,s)
            x=F.relu(x)
            x=self.adapters[i](x,k,s)
            if i in (1,3): x=self.stem.pool(x)
        return self.head(self.stem.avg(x).flatten(1))

@torch.no_grad()
def logit_deviation(ref,other,dl):
    ref.eval(); other.eval(); n=0; sq=0.; mx=0.
    for x,y,k,s in dl:
        x=x.to(DEVICE); k=k.to(DEVICE); s=s.to(DEVICE)
        a=ref(x,k,s); b=other(x,k,s); d=a-b
        sq+=(d*d).sum().item(); n+=d.numel(); mx=max(mx,d.abs().max().item())
    return (sq/n)**.5,mx

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--rank",type=int,default=16)
    ap.add_argument("--width",type=int,default=31)
    ap.add_argument("--out",default="pruning")
    a=ap.parse_args()
    seed_all(a.seed); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    tr=FixedCorruptCIFAR("data","train"); va=FixedCorruptCIFAR("data","val")
    ts=FixedCorruptCIFAR("data","test"); tu=FixedCorruptCIFAR("data","test",True)
    g=torch.Generator().manual_seed(9000+a.seed)
    tl=DataLoader(tr,128,shuffle=True,generator=g,num_workers=2)
    vl=DataLoader(va,256,num_workers=2); sl=DataLoader(ts,256,num_workers=2); ul=DataLoader(tu,256,num_workers=2)
    m=WideRTAdapterCNN(a.width,a.rank)
    np_=nparams(m); m,best=train(m,tl,vl,18)
    m.to(DEVICE)
    spectra=[]
    for li,ad in enumerate(m.adapters):
        for j in range(B):
            vals=ad.scores(j).detach().cpu().sort(descending=True).values.tolist()
            for kk,v in enumerate(vals,1):
                spectra.append([a.seed,li,j,kk,v])
    pd.DataFrame(spectra,columns=["seed","layer","band","unit_rank","path_score"]).to_csv(out/f"spectra_{a.seed:02d}.csv",index=False)
    rows=[]
    base_seen=evaluate(m,sl); base_un=evaluate(m,ul)
    for split,z in [("seen",base_seen),("unseen_severity",base_un)]:
        rows.append(["full",a.seed,split,z["accuracy"],z["nll"],np_,0.,0.])
    ranks=sorted(set([1,2,4,8,a.rank]))
    for r in ranks:
        pr=copy.deepcopy(m)
        for ad in pr.adapters: ad.prune_to(r)
        zs=evaluate(pr,sl); zu=evaluate(pr,ul)
        rms,mx=logit_deviation(m,pr,ul)
        for split,z in [("seen",zs),("unseen_severity",zu)]:
            rows.append([f"prune_{r}",a.seed,split,z["accuracy"],z["nll"],np_,rms if split=="unseen_severity" else 0.,mx if split=="unseen_severity" else 0.])
    pd.DataFrame(rows,columns=["model","seed","split","accuracy","nll","params","logit_rmse_vs_full","logit_max_vs_full"]).to_csv(out/f"results_{a.seed:02d}.csv",index=False)

if __name__=="__main__": main()
