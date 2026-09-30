import argparse, random, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, RandomCrop, RandomHorizontalFlip

sys.path.insert(0,str(Path(__file__).parent))
from cifar10_rt_modulated import ConvStem, BandFiLM, BandAdapter, seed_all, nparams

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
B=4
PERMS=torch.tensor([
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,5,6,7,8,9,0],
    [3,4,5,6,7,8,9,0,1,2],
    [7,2,9,4,1,6,0,8,5,3],
],dtype=torch.long)

class ContextCIFAR(Dataset):
    def __init__(self,root,split):
        train=split in ("train","val")
        self.base=CIFAR10(root=root,train=train,download=True)
        if train:
            perm=np.random.default_rng(4242).permutation(50000)
            self.idx=perm[:40000] if split=="train" else perm[40000:]
        else:
            self.idx=np.arange(10000)
        self.split=split
        self.to_tensor=ToTensor()
        self.aug=Compose([RandomCrop(32,padding=4),RandomHorizontalFlip()]) if split=="train" else None
        # Every base image is evaluated in every context. This creates true target conflict:
        # same visual object, different conditional rule.
        self.nbase=len(self.idx)
    def __len__(self): return self.nbase*B
    def __getitem__(self,i):
        b=i//B; k=i%B
        img,y=self.base[int(self.idx[b])]
        if self.aug is not None: img=self.aug(img)
        x=self.to_tensor(img)
        yt=int(PERMS[k,y])
        return x,yt,k,y

class ContextHeadCNN(nn.Module):
    def __init__(self,w,h=64):
        super().__init__(); self.stem=ConvStem(w)
        self.head=nn.Sequential(nn.Linear(4*w+B,h),nn.ReLU(),nn.Linear(h,10))
    def forward(self,x,k):
        for i in range(5): x=self.stem.finish(self.stem.block(x,i),i)
        h=self.stem.avg(x).flatten(1)
        q=torch.cat([h,F.one_hot(k,B).float()],1)
        return self.head(q)

class DenseFiLMContext(nn.Module):
    def __init__(self,w,hcond=8):
        super().__init__(); self.stem=ConvStem(w); self.sizes=self.stem.channels
        total=sum(self.sizes)
        self.cond1=nn.Linear(B,hcond); self.cond2=nn.Linear(hcond,2*total)
        nn.init.zeros_(self.cond2.weight); nn.init.zeros_(self.cond2.bias)
        self.head=nn.Linear(4*w,10)
    def forward(self,x,k):
        q=F.one_hot(k,B).float()
        pars=self.cond2(F.relu(self.cond1(q))); gam,bet=pars.chunk(2,1)
        gs=torch.split(gam,self.sizes,1); bs=torch.split(bet,self.sizes,1)
        for i in range(5):
            x=self.stem.block(x,i)
            x=x*(1+gs[i][:,:,None,None])+bs[i][:,:,None,None]
            x=self.stem.finish(x,i)
        return self.head(self.stem.avg(x).flatten(1))

class SoftMoEContext(nn.Module):
    def __init__(self,w,hidden=96):
        super().__init__(); self.stem=ConvStem(w); self.proj=nn.Linear(4*w,hidden)
        self.experts=nn.ModuleList([nn.Linear(hidden,10) for _ in range(B)])
        self.gate=nn.Sequential(nn.Linear(hidden+B,32),nn.ReLU(),nn.Linear(32,B))
    def forward(self,x,k):
        for i in range(5): x=self.stem.finish(self.stem.block(x,i),i)
        h=F.relu(self.proj(self.stem.avg(x).flatten(1)))
        g=torch.softmax(self.gate(torch.cat([h,F.one_hot(k,B).float()],1)),1)
        e=torch.stack([m(h) for m in self.experts],1)
        return (g[:,:,None]*e).sum(1)

class RTContextAdapter(nn.Module):
    def __init__(self,w):
        super().__init__(); self.stem=ConvStem(w)
        self.films=nn.ModuleList([BandFiLM(c) for c in self.stem.channels])
        ranks=[max(4,c//16) for c in self.stem.channels]
        self.adapters=nn.ModuleList([BandAdapter(c,r) for c,r in zip(self.stem.channels,ranks)])
        self.heads=nn.ModuleList([nn.Linear(4*w,10) for _ in range(B)])
    def forward(self,x,k):
        s=torch.full((x.shape[0],),0.5,device=x.device)
        for i in range(5):
            x=self.stem.block(x,i); x=self.films[i](x,k,s); x=F.relu(x)
            x=self.adapters[i](x,k,s)
            if i in (1,3): x=self.stem.pool(x)
        h=self.stem.avg(x).flatten(1)
        out=torch.empty((x.shape[0],10),device=x.device)
        for j in range(B):
            q=k==j
            if q.any(): out[q]=self.heads[j](h[q])
        return out

FACT={
 "cond_head":lambda w:ContextHeadCNN(w,64),
 "dense_film":lambda w:DenseFiLMContext(w,8),
 "softmoe":lambda w:SoftMoEContext(w,96),
 "rt_adapter":lambda w:RTContextAdapter(w),
}

def build_matched(name,target=170000):
    best=None
    for w in range(16,65):
        m=FACT[name](w); n=nparams(m); d=abs(n-target)
        if best is None or d<best[0]: best=(d,w,n)
    _,w,n=best; return FACT[name](w),w,n

@torch.no_grad()
def evaluate(m,dl):
    m.eval(); n=correct=0; loss=0.; by={j:[0,0] for j in range(B)}
    for x,y,k,_ in dl:
        x=x.to(DEVICE); y=y.to(DEVICE); k=k.to(DEVICE)
        z=m(x,k); loss+=F.cross_entropy(z,y,reduction="sum").item()
        ok=z.argmax(1)==y; correct+=ok.sum().item(); n+=len(y)
        for j in range(B):
            q=k==j
            by[j][0]+=ok[q].sum().item(); by[j][1]+=q.sum().item()
    return correct/n,loss/n,{j:by[j][0]/by[j][1] for j in range(B)}

def train(m,tl,vl,epochs=14):
    m.to(DEVICE); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=5e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=-1; state=None
    for _ in range(epochs):
        m.train()
        for x,y,k,_ in tl:
            x=x.to(DEVICE); y=y.to(DEVICE); k=k.to(DEVICE)
            opt.zero_grad(); z=m(x,k); loss=F.cross_entropy(z,y,label_smoothing=.05)
            loss.backward(); opt.step()
        sch.step(); a,_,_=evaluate(m,vl)
        if a>best:
            best=a; state={q:v.detach().cpu().clone() for q,v in m.state_dict().items()}
    m.load_state_dict(state); return m,best

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--model",choices=list(FACT),required=True); ap.add_argument("--out",default="context_perm")
    ap.add_argument("--target",type=int,default=170000); a=ap.parse_args()
    seed_all(a.seed); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    tr=ContextCIFAR("data","train"); va=ContextCIFAR("data","val"); te=ContextCIFAR("data","test")
    g=torch.Generator().manual_seed(12000+a.seed)
    tl=DataLoader(tr,128,shuffle=True,generator=g,num_workers=2)
    vl=DataLoader(va,256,num_workers=2); el=DataLoader(te,256,num_workers=2)
    m,w,np_=build_matched(a.model,a.target); m,best=train(m,tl,vl,14)
    acc,nll,by=evaluate(m,el)
    pd.DataFrame([[a.model,a.seed,acc,nll,np_,w,best]],
      columns=["model","seed","accuracy","nll","params","width","val_best"]).to_csv(out/f"{a.model}_{a.seed:02d}.csv",index=False)
    pd.DataFrame([[a.model,a.seed,j,by[j]] for j in range(B)],
      columns=["model","seed","context","accuracy"]).to_csv(out/f"byctx_{a.model}_{a.seed:02d}.csv",index=False)
    print(a.model,a.seed,acc,nll,np_,w,best,by)

if __name__=="__main__": main()
