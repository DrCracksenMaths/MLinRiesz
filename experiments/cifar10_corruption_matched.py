import argparse, json, math, random, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
warnings.filterwarnings("ignore")
torch.set_num_threads(2)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

CORRUPTIONS=["gaussian_noise","gaussian_blur","brightness","contrast"]
B=4

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def gaussian_kernel(sigma,device):
    k=max(3,int(2*math.ceil(2*sigma)+1))
    x=torch.arange(k,device=device)-k//2
    g=torch.exp(-(x.float()**2)/(2*sigma*sigma)); g/=g.sum()
    ker=(g[:,None]*g[None,:])[None,None]
    return ker,k

def corrupt(x,kind,severity):
    # x: C,H,W in [0,1], severity in [0,1]
    if kind==0:
        std=0.02+0.28*severity
        return torch.clamp(x+torch.randn_like(x)*std,0,1)
    if kind==1:
        sigma=0.25+1.75*severity
        ker,k=gaussian_kernel(sigma,x.device)
        ker=ker.repeat(3,1,1,1)
        y=F.conv2d(x[None],ker,padding=k//2,groups=3)[0]
        return torch.clamp(y,0,1)
    if kind==2:
        factor=0.55+1.15*severity
        return torch.clamp(x*factor,0,1)
    # contrast
    factor=0.35+1.45*severity
    m=x.mean(dim=(1,2),keepdim=True)
    return torch.clamp((x-m)*factor+m,0,1)

class CorruptCIFAR(Dataset):
    def __init__(self,root,train,n=None,seed=0,holdout=False):
        self.base=CIFAR10(root=root,train=train,download=True)
        rng=np.random.default_rng(seed)
        idx=np.arange(len(self.base))
        if n is not None and n<len(idx): idx=rng.choice(idx,n,replace=False)
        self.idx=idx
        self.kind=rng.integers(0,B,size=len(idx))
        if holdout:
            centers=np.array([0.35,0.70])
            self.sev=rng.choice(centers,size=len(idx))+rng.uniform(-0.035,0.035,size=len(idx))
            self.sev=np.clip(self.sev,0,1)
        else:
            vals=[]
            while len(vals)<len(idx):
                s=rng.uniform(0.05,0.95,size=len(idx))
                keep=(np.abs(s-.35)>.07)&(np.abs(s-.70)>.07)
                vals.extend(s[keep].tolist())
            self.sev=np.array(vals[:len(idx)])
        self.to_tensor=ToTensor()
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        img,y=self.base[int(self.idx[i])]
        x=self.to_tensor(img)
        # deterministic corruption noise per item
        state=torch.random.get_rng_state()
        torch.manual_seed(int(self.idx[i])*1009+int(self.kind[i])*97+int(self.sev[i]*10000))
        x=corrupt(x,int(self.kind[i]),float(self.sev[i]))
        torch.random.set_rng_state(state)
        return x,int(y),int(self.kind[i]),np.float32(self.sev[i])

class CNN(nn.Module):
    def __init__(self,w=32):
        super().__init__()
        self.feat=nn.Sequential(
            nn.Conv2d(3,w,3,padding=1),nn.BatchNorm2d(w),nn.ReLU(),
            nn.Conv2d(w,w,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(w,2*w,3,padding=1),nn.BatchNorm2d(2*w),nn.ReLU(),
            nn.Conv2d(2*w,2*w,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(2*w,4*w,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.head=nn.Linear(4*w,10)
    def forward(self,x): return self.head(self.feat(x).flatten(1))

class RTHard(nn.Module):
    def __init__(self,w=16):
        super().__init__(); self.experts=nn.ModuleList([CNN(w) for _ in range(B)])
    def forward(self,x,kind,sev=None):
        out=torch.empty((x.size(0),10),device=x.device)
        for j,m in enumerate(self.experts):
            mask=kind==j
            if mask.any(): out[mask]=m(x[mask])
        return out

class SoftMoE(nn.Module):
    def __init__(self,w=32,hidden=128):
        super().__init__()
        self.back=nn.Sequential(
            nn.Conv2d(3,w,3,padding=1),nn.BatchNorm2d(w),nn.ReLU(),
            nn.Conv2d(w,w,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(w,2*w,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(2*w,4*w,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.proj=nn.Linear(4*w,hidden)
        self.exp=nn.ModuleList([nn.Linear(hidden,10) for _ in range(B)])
        self.gate=nn.Sequential(nn.Linear(hidden+B+1,64),nn.ReLU(),nn.Linear(64,B))
    def forward(self,x,kind,sev):
        h=torch.relu(self.proj(self.back(x).flatten(1)))
        one=F.one_hot(kind,num_classes=B).float()
        g=torch.softmax(self.gate(torch.cat([h,one,sev[:,None]],1)),1)
        e=torch.stack([m(h) for m in self.exp],1)
        return (g[:,:,None]*e).sum(1)

def params(m): return sum(p.numel() for p in m.parameters())

def choose_width(factory,target,cands=range(8,97,2)):
    best=None
    for w in cands:
        n=params(factory(w)); d=abs(n-target)
        if best is None or d<best[0]: best=(d,w,n)
    return best[1],best[2]

def forward_model(m,x,k,s):
    if isinstance(m,CNN): return m(x)
    return m(x,k,s)

def train_model(m,tl,vl,epochs=12):
    m.to(DEVICE); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=-1; state=None
    for ep in range(epochs):
        m.train()
        for x,y,k,s in tl:
            x=x.to(DEVICE); y=y.to(DEVICE); k=k.to(DEVICE); s=s.to(DEVICE)
            opt.zero_grad(); z=forward_model(m,x,k,s); loss=F.cross_entropy(z,y)
            loss.backward(); opt.step()
        sched.step()
        acc=evaluate(m,vl)["accuracy"]
        if acc>best:
            best=acc; state={a:b.detach().cpu().clone() for a,b in m.state_dict().items()}
    m.load_state_dict(state); m.to(DEVICE); return m,best

@torch.no_grad()
def evaluate(m,dl):
    m.eval(); n=correct=0; loss=0.
    by={j:[0,0] for j in range(B)}
    for x,y,k,s in dl:
        x=x.to(DEVICE); y=y.to(DEVICE); k=k.to(DEVICE); s=s.to(DEVICE)
        z=forward_model(m,x,k,s); loss+=F.cross_entropy(z,y,reduction="sum").item()
        ok=z.argmax(1)==y; correct+=ok.sum().item(); n+=len(y)
        for j in range(B):
            q=k==j
            if q.any():
                by[j][0]+=ok[q].sum().item(); by[j][1]+=q.sum().item()
    return {"accuracy":correct/n,"nll":loss/n,
            "by_corruption":{CORRUPTIONS[j]:by[j][0]/by[j][1] for j in range(B) if by[j][1]}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--out",default="cifar_results"); ap.add_argument("--train-n",type=int,default=36000)
    ap.add_argument("--val-n",type=int,default=8000); ap.add_argument("--test-n",type=int,default=10000)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); seed_all(a.seed)

    tr=CorruptCIFAR("data",True,a.train_n,100+a.seed,False)
    va=CorruptCIFAR("data",True,a.val_n,200+a.seed,False)
    ts=CorruptCIFAR("data",False,a.test_n,300+a.seed,False)
    tu=CorruptCIFAR("data",False,a.test_n,400+a.seed,True)
    g=torch.Generator().manual_seed(500+a.seed)
    tl=DataLoader(tr,batch_size=128,shuffle=True,generator=g,num_workers=2)
    vl=DataLoader(va,batch_size=256,shuffle=False,num_workers=2)
    sl=DataLoader(ts,batch_size=256,shuffle=False,num_workers=2)
    ul=DataLoader(tu,batch_size=256,shuffle=False,num_workers=2)

    rt=RTHard(16); target=params(rt)
    cw,cp=choose_width(lambda w:CNN(w),target)
    mw,mp=choose_width(lambda w:SoftMoE(w),target)
    specs=[("CNN-matched",1,CNN(cw)),("RT-hard",B,rt),("Soft-MoE-matched",B,SoftMoE(mw))]
    rows=[]; byrows=[]
    for name,b,m in specs:
        m,_=train_model(m,tl,vl,epochs=12)
        for split,dl in [("seen",sl),("unseen_severity",ul)]:
            z=evaluate(m,dl)
            rows.append([name,b,a.seed,split,z["accuracy"],z["nll"],params(m)])
            for corr,acc in z["by_corruption"].items():
                byrows.append([name,b,a.seed,split,corr,acc])
    df=pd.DataFrame(rows,columns=["model","B","seed","split","accuracy","nll","params"])
    df.to_csv(out/f"seed_{a.seed:02d}.csv",index=False)
    pd.DataFrame(byrows,columns=["model","B","seed","split","corruption","accuracy"]).to_csv(out/f"bycorr_{a.seed:02d}.csv",index=False)
    meta={"seed":a.seed,"target_params":target,"cnn_width":cw,"cnn_params":cp,"moe_width":mw,"moe_params":mp,
          "rt_params":target,"corruptions":CORRUPTIONS}
    (out/f"meta_{a.seed:02d}.json").write_text(json.dumps(meta,indent=2))
    print(meta); print(df.to_string(index=False))
if __name__=="__main__": main()
