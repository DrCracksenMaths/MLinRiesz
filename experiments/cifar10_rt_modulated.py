import argparse, json, math, random, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Compose

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
B=4
CORRUPTIONS=["gaussian_noise","gaussian_blur","brightness","contrast"]
CHANNEL_MULT=[1,1,2,2,4]

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def gaussian_kernel(sigma,device):
    k=max(3,int(2*math.ceil(2*sigma)+1))
    x=torch.arange(k,device=device)-k//2
    g=torch.exp(-(x.float()**2)/(2*sigma*sigma)); g/=g.sum()
    return (g[:,None]*g[None,:])[None,None],k

def corrupt(x,kind,severity):
    if kind==0:
        return torch.clamp(x+torch.randn_like(x)*(0.02+0.28*severity),0,1)
    if kind==1:
        sigma=0.25+1.75*severity
        ker,k=gaussian_kernel(sigma,x.device); ker=ker.repeat(3,1,1,1)
        return torch.clamp(F.conv2d(x[None],ker,padding=k//2,groups=3)[0],0,1)
    if kind==2:
        return torch.clamp(x*(0.55+1.15*severity),0,1)
    factor=0.35+1.45*severity
    m=x.mean(dim=(1,2),keepdim=True)
    return torch.clamp((x-m)*factor+m,0,1)

def sample_seen(rng,n):
    vals=[]
    while len(vals)<n:
        s=rng.uniform(0.05,0.95,size=n)
        q=(np.abs(s-.35)>.07)&(np.abs(s-.70)>.07)
        vals.extend(s[q].tolist())
    return np.array(vals[:n],dtype=np.float32)

def sample_unseen(rng,n):
    c=np.array([.35,.70])
    return np.clip(rng.choice(c,size=n)+rng.uniform(-.035,.035,size=n),0,1).astype(np.float32)

class FixedCorruptCIFAR(Dataset):
    def __init__(self,root,split,holdout=False):
        train=split in ("train","val")
        self.base=CIFAR10(root=root,train=train,download=True)
        if train:
            perm=np.random.default_rng(12345).permutation(50000)
            self.idx=perm[:40000] if split=="train" else perm[40000:]
        else:
            self.idx=np.arange(10000)
        split_seed={"train":2101,"val":2102,"test":2103}[split] + (100 if holdout else 0)
        rng=np.random.default_rng(split_seed)
        self.kind=rng.integers(0,B,size=len(self.idx),dtype=np.int64)
        self.sev=sample_unseen(rng,len(self.idx)) if holdout else sample_seen(rng,len(self.idx))
        self.augment=(Compose([RandomCrop(32,padding=4),RandomHorizontalFlip()]) if split=="train" else None)
        self.to_tensor=ToTensor()
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        img,y=self.base[int(self.idx[i])]
        if self.augment is not None: img=self.augment(img)
        x=self.to_tensor(img); k=int(self.kind[i]); s=float(self.sev[i])
        state=torch.random.get_rng_state()
        torch.manual_seed(int(self.idx[i])*1009+k*97+int(s*10000))
        x=corrupt(x,k,s)
        torch.random.set_rng_state(state)
        return x,int(y),k,np.float32(s)

class ConvStem(nn.Module):
    def __init__(self,w):
        super().__init__()
        cs=[w,w,2*w,2*w,4*w]; ins=[3,w,w,2*w,2*w]
        self.convs=nn.ModuleList([nn.Conv2d(a,b,3,padding=1) for a,b in zip(ins,cs)])
        self.bns=nn.ModuleList([nn.BatchNorm2d(c) for c in cs])
        self.pool=nn.MaxPool2d(2)
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.channels=cs
    def block(self,x,i):
        x=self.convs[i](x); x=self.bns[i](x); return x
    def finish(self,x,i):
        x=F.relu(x)
        if i in (1,3): x=self.pool(x)
        return x

class PlainCNN(nn.Module):
    def __init__(self,w):
        super().__init__(); self.stem=ConvStem(w); self.head=nn.Linear(4*w,10)
    def forward(self,x,k=None,s=None):
        for i in range(5):
            x=self.stem.finish(self.stem.block(x,i),i)
        return self.head(self.stem.avg(x).flatten(1))

class BandFiLM(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.ga=nn.Parameter(torch.zeros(B,c)); self.gs=nn.Parameter(torch.zeros(B,c))
        self.ba=nn.Parameter(torch.zeros(B,c)); self.bs=nn.Parameter(torch.zeros(B,c))
    def forward(self,x,k,s):
        t=(s-.5)[:,None]
        g=1.+self.ga[k]+t*self.gs[k]
        b=self.ba[k]+t*self.bs[k]
        return x*g[:,:,None,None]+b[:,:,None,None]

class RTFiLMCNN(nn.Module):
    def __init__(self,w):
        super().__init__(); self.stem=ConvStem(w)
        self.films=nn.ModuleList([BandFiLM(c) for c in self.stem.channels])
        self.head=nn.Linear(4*w,10)
    def forward(self,x,k,s):
        for i in range(5):
            x=self.stem.block(x,i); x=self.films[i](x,k,s); x=self.stem.finish(x,i)
        return self.head(self.stem.avg(x).flatten(1))

class DenseFiLMCNN(nn.Module):
    def __init__(self,w,hcond=8):
        super().__init__(); self.stem=ConvStem(w)
        total=sum(self.stem.channels)
        self.cond1=nn.Linear(B+1,hcond); self.cond2=nn.Linear(hcond,2*total)
        nn.init.zeros_(self.cond2.weight); nn.init.zeros_(self.cond2.bias)
        self.head=nn.Linear(4*w,10); self.sizes=self.stem.channels
    def forward(self,x,k,s):
        q=torch.cat([F.one_hot(k,B).float(),s[:,None]],1)
        pars=self.cond2(F.relu(self.cond1(q)))
        gam,bet=pars.chunk(2,1); go=bo=0
        gs=torch.split(gam,self.sizes,dim=1); bs=torch.split(bet,self.sizes,dim=1)
        for i in range(5):
            x=self.stem.block(x,i)
            x=x*(1.+gs[i][:,:,None,None])+bs[i][:,:,None,None]
            x=self.stem.finish(x,i)
        return self.head(self.stem.avg(x).flatten(1))

class BandAdapter(nn.Module):
    def __init__(self,c,rank):
        super().__init__()
        self.down=nn.ModuleList([nn.Conv2d(c,rank,1,bias=False) for _ in range(B)])
        self.up=nn.ModuleList([nn.Conv2d(rank,c,1,bias=False) for _ in range(B)])
        for u in self.up: nn.init.zeros_(u.weight)
        self.a=nn.Parameter(torch.ones(B)); self.b=nn.Parameter(torch.zeros(B))
    def forward(self,x,k,s):
        out=x.clone()
        for j in range(B):
            q=k==j
            if q.any():
                scale=(self.a[j]+self.b[j]*(s[q]-.5))[:,None,None,None]
                out[q]=x[q]+scale*self.up[j](F.relu(self.down[j](x[q])))
        return out

class RTFiLMAdapterCNN(nn.Module):
    def __init__(self,w):
        super().__init__(); self.stem=ConvStem(w)
        self.films=nn.ModuleList([BandFiLM(c) for c in self.stem.channels])
        ranks=[max(4,c//16) for c in self.stem.channels]
        self.adapters=nn.ModuleList([BandAdapter(c,r) for c,r in zip(self.stem.channels,ranks)])
        self.head=nn.Linear(4*w,10)
    def forward(self,x,k,s):
        for i in range(5):
            x=self.stem.block(x,i); x=self.films[i](x,k,s); x=F.relu(x)
            x=self.adapters[i](x,k,s)
            if i in (1,3): x=self.stem.pool(x)
        return self.head(self.stem.avg(x).flatten(1))

class SoftMoE(nn.Module):
    def __init__(self,w,hidden=96):
        super().__init__(); self.stem=ConvStem(w); self.proj=nn.Linear(4*w,hidden)
        self.experts=nn.ModuleList([nn.Linear(hidden,10) for _ in range(B)])
        self.gate=nn.Sequential(nn.Linear(hidden+B+1,32),nn.ReLU(),nn.Linear(32,B))
    def forward(self,x,k,s):
        for i in range(5): x=self.stem.finish(self.stem.block(x,i),i)
        h=F.relu(self.proj(self.stem.avg(x).flatten(1)))
        q=torch.cat([h,F.one_hot(k,B).float(),s[:,None]],1)
        g=torch.softmax(self.gate(q),1)
        e=torch.stack([m(h) for m in self.experts],1)
        return (g[:,:,None]*e).sum(1)

def nparams(m): return sum(p.numel() for p in m.parameters())

FACTORIES={
 "cnn":lambda w:PlainCNN(w),
 "dense_film":lambda w:DenseFiLMCNN(w,8),
 "rt_film":lambda w:RTFiLMCNN(w),
 "rt_adapter":lambda w:RTFiLMAdapterCNN(w),
 "softmoe":lambda w:SoftMoE(w,96)
}

def build_matched(name,target=150000):
    best=None
    for w in range(16,65):
        m=FACTORIES[name](w); n=nparams(m); d=abs(n-target)
        if best is None or d<best[0]: best=(d,w,n)
    _,w,n=best
    return FACTORIES[name](w),w,n

@torch.no_grad()
def evaluate(m,dl):
    m.eval(); n=correct=0; loss=0.; by={j:[0,0] for j in range(B)}
    for x,y,k,s in dl:
        x=x.to(DEVICE); y=y.to(DEVICE); k=k.to(DEVICE); s=s.to(DEVICE)
        z=m(x,k,s); loss+=F.cross_entropy(z,y,reduction="sum").item()
        ok=z.argmax(1)==y; correct+=ok.sum().item(); n+=len(y)
        for j in range(B):
            q=k==j
            if q.any(): by[j][0]+=ok[q].sum().item(); by[j][1]+=q.sum().item()
    return {"accuracy":correct/n,"nll":loss/n,
      "by_corruption":{CORRUPTIONS[j]:by[j][0]/by[j][1] for j in range(B)}}

def train(m,tl,vl,epochs=18):
    m.to(DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=-1.; state=None
    for ep in range(epochs):
        m.train()
        for x,y,k,s in tl:
            x=x.to(DEVICE); y=y.to(DEVICE); k=k.to(DEVICE); s=s.to(DEVICE)
            opt.zero_grad(); z=m(x,k,s)
            loss=F.cross_entropy(z,y,label_smoothing=.05)
            loss.backward(); opt.step()
        sched.step()
        a=evaluate(m,vl)["accuracy"]
        if a>best:
            best=a; state={q:v.detach().cpu().clone() for q,v in m.state_dict().items()}
    m.load_state_dict(state); return m,best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True); ap.add_argument("--model",choices=list(FACTORIES),required=True)
    ap.add_argument("--out",default="rt_mod_screen"); ap.add_argument("--target",type=int,default=150000)
    a=ap.parse_args(); seed_all(a.seed); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    tr=FixedCorruptCIFAR("data","train",False); va=FixedCorruptCIFAR("data","val",False)
    ts=FixedCorruptCIFAR("data","test",False); tu=FixedCorruptCIFAR("data","test",True)
    g=torch.Generator().manual_seed(9000+a.seed)
    tl=DataLoader(tr,batch_size=128,shuffle=True,generator=g,num_workers=2)
    vl=DataLoader(va,batch_size=256,shuffle=False,num_workers=2)
    sl=DataLoader(ts,batch_size=256,shuffle=False,num_workers=2)
    ul=DataLoader(tu,batch_size=256,shuffle=False,num_workers=2)
    m,w,np_=build_matched(a.model,a.target); m,best=train(m,tl,vl,18)
    rows=[]; by=[]
    for split,dl in [("seen",sl),("unseen_severity",ul)]:
        z=evaluate(m,dl); rows.append([a.model,a.seed,split,z["accuracy"],z["nll"],np_,w,best])
        for c,v in z["by_corruption"].items(): by.append([a.model,a.seed,split,c,v])
    pd.DataFrame(rows,columns=["model","seed","split","accuracy","nll","params","width","val_best"]).to_csv(out/f"{a.model}_seed_{a.seed:02d}.csv",index=False)
    pd.DataFrame(by,columns=["model","seed","split","corruption","accuracy"]).to_csv(out/f"bycorr_{a.model}_seed_{a.seed:02d}.csv",index=False)
    print(rows)

if __name__=="__main__": main()
