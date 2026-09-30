import argparse, sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).parent))
from cifar10_rt_modulated import (
    B, DEVICE, CORRUPTIONS, FixedCorruptCIFAR, ConvStem, PlainCNN,
    RTFiLMCNN, RTFiLMAdapterCNN, seed_all, nparams, evaluate, train
)

class AlgebraicFiLM(nn.Module):
    """Same affine map of one-hot band and severity interactions as RT-FiLM,
    written as an ordinary dense linear conditional map."""
    def __init__(self,w):
        super().__init__(); self.stem=ConvStem(w); self.sizes=self.stem.channels
        total=sum(self.sizes)
        # Features [1_j, 1_j*(s-.5)] for j=1,...,B.
        self.map=nn.Linear(2*B,2*total,bias=False)
        nn.init.zeros_(self.map.weight)
        self.head=nn.Linear(4*w,10)
    def cond(self,k,s):
        one=F.one_hot(k,B).float(); t=(s-.5)[:,None]
        return torch.cat([one,one*t],1)
    def forward(self,x,k,s):
        pars=self.map(self.cond(k,s)); gam,bet=pars.chunk(2,1)
        gs=torch.split(gam,self.sizes,1); bs=torch.split(bet,self.sizes,1)
        for i in range(5):
            x=self.stem.block(x,i)
            x=x*(1+gs[i][:,:,None,None])+bs[i][:,:,None,None]
            x=self.stem.finish(x,i)
        return self.head(self.stem.avg(x).flatten(1))

class AlgebraicAdapter(nn.Module):
    """Ordinary conditional low-rank adapters with exactly the RT adapter
    routing family, implemented without band/projection modules."""
    def __init__(self,w):
        super().__init__(); self.stem=ConvStem(w); self.sizes=self.stem.channels
        total=sum(self.sizes)
        self.film=nn.Linear(2*B,2*total,bias=False); nn.init.zeros_(self.film.weight)
        ranks=[max(4,c//16) for c in self.sizes]
        # Tensor parameters rather than ModuleList of band operators.
        self.down=nn.ParameterList([nn.Parameter(torch.empty(B,r,c)) for c,r in zip(self.sizes,ranks)])
        self.up=nn.ParameterList([nn.Parameter(torch.zeros(B,c,r)) for c,r in zip(self.sizes,ranks)])
        self.scale=nn.Parameter(torch.zeros(B,2))
        for p in self.down: nn.init.kaiming_uniform_(p,a=5**.5)
        self.head=nn.Linear(4*w,10)
    def cond(self,k,s):
        one=F.one_hot(k,B).float(); t=(s-.5)[:,None]
        return torch.cat([one,one*t],1)
    def forward(self,x,k,s):
        pars=self.film(self.cond(k,s)); gam,bet=pars.chunk(2,1)
        gs=torch.split(gam,self.sizes,1); bs=torch.split(bet,self.sizes,1)
        for i in range(5):
            x=self.stem.block(x,i)
            x=x*(1+gs[i][:,:,None,None])+bs[i][:,:,None,None]
            x=F.relu(x)
            # Batched selected matrices: mathematically same finite conditional family.
            d=self.down[i][k]; u=self.up[i][k]
            z=torch.einsum("brc,bchw->brhw",d,x); z=F.relu(z)
            z=torch.einsum("bcr,brhw->bchw",u,z)
            sc=1+self.scale[k,0]+self.scale[k,1]*(s-.5)
            x=x+sc[:,None,None,None]*z
            if i in (1,3): x=self.stem.pool(x)
        return self.head(self.stem.avg(x).flatten(1))

FACT={
 "cnn":lambda w:PlainCNN(w),
 "rt_film":lambda w:RTFiLMCNN(w),
 "alg_film":lambda w:AlgebraicFiLM(w),
 "rt_adapter":lambda w:RTFiLMAdapterCNN(w),
 "alg_adapter":lambda w:AlgebraicAdapter(w)
}

def matched(name,target):
    z=None
    for w in range(16,65):
        m=FACT[name](w); n=nparams(m)
        if z is None or abs(n-target)<z[0]: z=(abs(n-target),w,n)
    _,w,n=z; return FACT[name](w),w,n

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--model",choices=list(FACT),required=True); ap.add_argument("--out",default="exact_ablation")
    ap.add_argument("--target",type=int,default=150000); a=ap.parse_args()
    seed_all(a.seed); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    tr=FixedCorruptCIFAR("data","train"); va=FixedCorruptCIFAR("data","val")
    ts=FixedCorruptCIFAR("data","test"); tu=FixedCorruptCIFAR("data","test",True)
    g=torch.Generator().manual_seed(9000+a.seed)
    tl=DataLoader(tr,128,shuffle=True,generator=g,num_workers=2)
    vl=DataLoader(va,256,num_workers=2); sl=DataLoader(ts,256,num_workers=2); ul=DataLoader(tu,256,num_workers=2)
    m,w,np_=matched(a.model,a.target); m,best=train(m,tl,vl,18)
    rows=[]; by=[]
    for split,dl in [("seen",sl),("unseen_severity",ul)]:
        z=evaluate(m,dl); rows.append([a.model,a.seed,split,z["accuracy"],z["nll"],np_,w,best])
        for c,v in z["by_corruption"].items(): by.append([a.model,a.seed,split,c,v])
    pd.DataFrame(rows,columns=["model","seed","split","accuracy","nll","params","width","val_best"]).to_csv(out/f"{a.model}_{a.seed:02d}.csv",index=False)
    pd.DataFrame(by,columns=["model","seed","split","corruption","accuracy"]).to_csv(out/f"bycorr_{a.model}_{a.seed:02d}.csv",index=False)

if __name__=="__main__": main()
