import argparse, json, sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).parent))
from cifar10_corruption_matched import (
    CorruptCIFAR, CNN, SoftMoE, B, CORRUPTIONS, DEVICE,
    seed_all, params, choose_width, evaluate
)

class RTSharedHead(nn.Module):
    """Shared visual representation, R(T)-localized classification heads."""
    def __init__(self,w=32,hidden=128):
        super().__init__()
        self.back=nn.Sequential(
            nn.Conv2d(3,w,3,padding=1),nn.BatchNorm2d(w),nn.ReLU(),
            nn.Conv2d(w,w,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(w,2*w,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(2*w,4*w,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.proj=nn.Linear(4*w,hidden)
        self.heads=nn.ModuleList([nn.Linear(hidden,10) for _ in range(B)])
    def forward(self,x,kind,sev=None):
        h=torch.relu(self.proj(self.back(x).flatten(1)))
        out=torch.empty((x.size(0),10),device=x.device)
        for j,head in enumerate(self.heads):
            q=kind==j
            if q.any(): out[q]=head(h[q])
        return out

def forward_model(m,x,k,s):
    if isinstance(m,CNN): return m(x)
    return m(x,k,s)

def eval_model(m,dl):
    m.eval(); n=correct=0; loss=0.; by={j:[0,0] for j in range(B)}
    with torch.no_grad():
        for x,y,k,s in dl:
            x=x.to(DEVICE); y=y.to(DEVICE); k=k.to(DEVICE); s=s.to(DEVICE)
            z=forward_model(m,x,k,s)
            loss+=F.cross_entropy(z,y,reduction="sum").item()
            ok=z.argmax(1)==y; correct+=ok.sum().item(); n+=len(y)
            for j in range(B):
                q=k==j
                if q.any(): by[j][0]+=ok[q].sum().item(); by[j][1]+=q.sum().item()
    return {"accuracy":correct/n,"nll":loss/n,
      "by_corruption":{CORRUPTIONS[j]:by[j][0]/by[j][1] for j in range(B) if by[j][1]}}

def train(m,tl,vl,epochs=12):
    m.to(DEVICE); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=-1; state=None
    for ep in range(epochs):
        m.train()
        for x,y,k,s in tl:
            x=x.to(DEVICE); y=y.to(DEVICE); k=k.to(DEVICE); s=s.to(DEVICE)
            opt.zero_grad(); z=forward_model(m,x,k,s); loss=F.cross_entropy(z,y)
            loss.backward(); opt.step()
        sched.step(); acc=eval_model(m,vl)["accuracy"]
        if acc>best:
            best=acc; state={a:b.detach().cpu().clone() for a,b in m.state_dict().items()}
    m.load_state_dict(state); m.to(DEVICE); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--out",default="cifar_shared_rt_results")
    ap.add_argument("--train-n",type=int,default=36000); ap.add_argument("--val-n",type=int,default=8000)
    ap.add_argument("--test-n",type=int,default=10000); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); seed_all(a.seed)
    tr=CorruptCIFAR("data",True,a.train_n,100+a.seed,False)
    va=CorruptCIFAR("data",True,a.val_n,200+a.seed,False)
    ts=CorruptCIFAR("data",False,a.test_n,300+a.seed,False)
    tu=CorruptCIFAR("data",False,a.test_n,400+a.seed,True)
    g=torch.Generator().manual_seed(500+a.seed)
    tl=DataLoader(tr,batch_size=128,shuffle=True,generator=g,num_workers=2)
    vl=DataLoader(va,batch_size=256,shuffle=False,num_workers=2)
    sl=DataLoader(ts,batch_size=256,shuffle=False,num_workers=2)
    ul=DataLoader(tu,batch_size=256,shuffle=False,num_workers=2)

    # Fix RT architecture, then match the two classical comparators to its capacity.
    rt=RTSharedHead(32,128); target=params(rt)
    cw,cp=choose_width(lambda w:CNN(w),target,range(8,129,2))
    mw,mp=choose_width(lambda w:SoftMoE(w),target,range(8,129,2))
    specs=[("CNN-matched",1,CNN(cw)),("RT-shared-head",B,rt),("Soft-MoE-matched",B,SoftMoE(mw))]
    rows=[]; by=[]
    for name,b,m in specs:
        m=train(m,tl,vl,12)
        for split,dl in [("seen",sl),("unseen_severity",ul)]:
            z=eval_model(m,dl)
            rows.append([name,b,a.seed,split,z["accuracy"],z["nll"],params(m)])
            for corr,acc in z["by_corruption"].items(): by.append([name,b,a.seed,split,corr,acc])
    pd.DataFrame(rows,columns=["model","B","seed","split","accuracy","nll","params"]).to_csv(out/f"seed_{a.seed:02d}.csv",index=False)
    pd.DataFrame(by,columns=["model","B","seed","split","corruption","accuracy"]).to_csv(out/f"bycorr_{a.seed:02d}.csv",index=False)
    meta={"seed":a.seed,"rt_params":target,"cnn_width":cw,"cnn_params":cp,"moe_width":mw,"moe_params":mp}
    (out/f"meta_{a.seed:02d}.json").write_text(json.dumps(meta,indent=2)); print(meta)

if __name__=="__main__": main()
