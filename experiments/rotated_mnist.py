import argparse, json, math, os, random, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.functional import rotate
from torchvision.transforms import ToTensor
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class RotMNIST(Dataset):
    def __init__(self, root, train, n=None, seed=0, holdout=False, train_angles=True):
        base=MNIST(root=root,train=train,download=True)
        idx=np.arange(len(base))
        rng=np.random.default_rng(seed)
        if n is not None and n<len(idx): idx=rng.choice(idx,n,replace=False)
        self.base=base; self.idx=idx
        # fixed angle per sample for reproducibility
        if holdout:
            # interleaved unseen angles: +/-15 and +/-45 deg neighborhoods
            pools=np.array([-45.,-15.,15.,45.])
            self.angles=rng.choice(pools,size=len(idx)) + rng.uniform(-2.5,2.5,size=len(idx))
        else:
            # training excludes neighborhoods around held-out angles
            vals=[]
            while len(vals)<len(idx):
                a=rng.uniform(-60,60,size=len(idx))
                keep=np.ones(len(a),dtype=bool)
                for c in [-45,-15,15,45]:
                    keep &= np.abs(a-c)>5
                vals.extend(a[keep].tolist())
            self.angles=np.array(vals[:len(idx)])
        self.to_tensor=ToTensor()
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        img,y=self.base[int(self.idx[i])]
        a=float(self.angles[i])
        img=rotate(img,a,fill=0)
        x=self.to_tensor(img)
        return x,int(y),np.float32(a)

class SharedCNN(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,width,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width,2*width,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(2*width*7*7,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x): return self.head(self.features(x))

class SoftMoE(nn.Module):
    def __init__(self,B,width=24):
        super().__init__()
        self.B=B
        self.backbone=nn.Sequential(
            nn.Conv2d(1,width,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(width,2*width,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Flatten(),nn.Linear(2*width*7*7,128),nn.ReLU())
        self.experts=nn.ModuleList([nn.Linear(128,10) for _ in range(B)])
        self.gate=nn.Sequential(nn.Linear(129,64),nn.ReLU(),nn.Linear(64,B))
    def forward(self,x,angle):
        h=self.backbone(x)
        aa=(angle[:,None]/60.0)
        g=torch.softmax(self.gate(torch.cat([h,aa],1)),1)
        e=torch.stack([m(h) for m in self.experts],1)
        return (g[:,:,None]*e).sum(1)

class RTHard(nn.Module):
    def __init__(self,B,width=24):
        super().__init__(); self.B=B
        self.experts=nn.ModuleList([SharedCNN(width) for _ in range(B)])
    def bands(self,angle):
        z=((angle+60.)/120.*self.B).long()
        return torch.clamp(z,0,self.B-1)
    def forward(self,x,angle):
        b=self.bands(angle); out=torch.empty((x.size(0),10),device=x.device)
        for j,m in enumerate(self.experts):
            mask=b==j
            if mask.any(): out[mask]=m(x[mask])
        return out

def train_model(model,train_loader,val_loader,epochs=8,lr=1e-3):
    model.to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    ce=nn.CrossEntropyLoss()
    best=-1; state=None
    for ep in range(epochs):
        model.train()
        for x,y,a in train_loader:
            x=x.to(DEVICE); y=y.to(DEVICE); a=a.to(DEVICE)
            opt.zero_grad()
            logits=model(x,a) if not isinstance(model,SharedCNN) else model(x)
            loss=ce(logits,y); loss.backward(); opt.step()
        acc=evaluate(model,val_loader)["accuracy"]
        if acc>best:
            best=acc; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state); model.to(DEVICE)
    return model,best

@torch.no_grad()
def evaluate(model,loader,B=None):
    model.eval(); n=0; correct=0; loss=0.
    ce=nn.CrossEntropyLoss(reduction="sum")
    by={}
    for x,y,a in loader:
        x=x.to(DEVICE); y=y.to(DEVICE); a=a.to(DEVICE)
        logits=model(x,a) if not isinstance(model,SharedCNN) else model(x)
        loss+=ce(logits,y).item()
        pred=logits.argmax(1); ok=(pred==y)
        correct+=ok.sum().item(); n+=len(y)
        if B is not None:
            bb=torch.clamp((((a+60)/120)*B).long(),0,B-1)
            for j in range(B):
                m=bb==j
                if m.any():
                    c,t=by.get(j,(0,0)); by[j]=(c+ok[m].sum().item(),t+m.sum().item())
    return {"accuracy":correct/n,"nll":loss/n,
            "by_band":{int(j):c/t for j,(c,t) in by.items()}}

def param_count(m): return sum(p.numel() for p in m.parameters())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="image_results")
    ap.add_argument("--seeds",type=int,default=5)
    ap.add_argument("--train-n",type=int,default=24000)
    ap.add_argument("--val-n",type=int,default=6000)
    ap.add_argument("--test-n",type=int,default=10000)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]; bandrows=[]
    for seed in range(args.seeds):
        seed_all(seed)
        tr=RotMNIST("data",True,args.train_n,seed=100+seed,holdout=False)
        va=RotMNIST("data",True,args.val_n,seed=200+seed,holdout=False)
        te_seen=RotMNIST("data",False,args.test_n,seed=300+seed,holdout=False)
        te_unseen=RotMNIST("data",False,args.test_n,seed=400+seed,holdout=True)
        g=torch.Generator().manual_seed(500+seed)
        tl=DataLoader(tr,batch_size=128,shuffle=True,generator=g,num_workers=2)
        vl=DataLoader(va,batch_size=256,shuffle=False,num_workers=2)
        tsl=DataLoader(te_seen,batch_size=256,shuffle=False,num_workers=2)
        tul=DataLoader(te_unseen,batch_size=256,shuffle=False,num_workers=2)

        # shared CNN
        m,_=train_model(SharedCNN(32),tl,vl,epochs=8)
        for split,loader in [("seen",tsl),("unseen_angles",tul)]:
            z=evaluate(m,loader); rows.append(["CNN",1,seed,split,z["accuracy"],z["nll"],param_count(m)])

        for B in [2,3,4,6,8]:
            # R(T) hard
            rt,_=train_model(RTHard(B,16),tl,vl,epochs=8)
            for split,loader in [("seen",tsl),("unseen_angles",tul)]:
                z=evaluate(rt,loader,B)
                rows.append(["RT-hard",B,seed,split,z["accuracy"],z["nll"],param_count(rt)])
                if seed==0:
                    for j,acc in z["by_band"].items():
                        bandrows.append(["RT-hard",B,split,j,acc])
            # classical soft MoE with shared trunk
            moe,_=train_model(SoftMoE(B,24),tl,vl,epochs=8)
            for split,loader in [("seen",tsl),("unseen_angles",tul)]:
                z=evaluate(moe,loader,B)
                rows.append(["Soft-MoE",B,seed,split,z["accuracy"],z["nll"],param_count(moe)])
                if seed==0:
                    for j,acc in z["by_band"].items():
                        bandrows.append(["Soft-MoE",B,split,j,acc])

    res=pd.DataFrame(rows,columns=["model","B","seed","split","accuracy","nll","params"])
    res.to_csv(out/"rotated_mnist_results.csv",index=False)
    pd.DataFrame(bandrows,columns=["model","B","split","band","accuracy"]).to_csv(out/"band_accuracy.csv",index=False)
    summary=(res.groupby(["model","B","split"])
             .agg(acc_mean=("accuracy","mean"),acc_sd=("accuracy","std"),
                  nll_mean=("nll","mean"),params=("params","first"),n=("seed","count"))
             .reset_index())
    summary.to_csv(out/"summary.csv",index=False)
    meta={"device":str(DEVICE),"train_n":args.train_n,"val_n":args.val_n,"test_n":args.test_n,
          "seeds":args.seeds,"train_angle_range":[-60,60],
          "heldout_centers_deg":[-45,-15,15,45],"heldout_halfwidth_deg":5}
    (out/"meta.json").write_text(json.dumps(meta,indent=2))
    print(summary.to_string(index=False))

if __name__=="__main__": main()
