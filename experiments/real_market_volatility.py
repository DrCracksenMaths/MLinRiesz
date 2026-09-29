import argparse, json, math, os, random, warnings
from pathlib import Path
import numpy as np, pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

START="2010-01-01"
END=None
HORIZON=5

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def download():
    spy=yf.download("SPY",start=START,end=END,auto_adjust=True,progress=False,threads=False)
    vix=yf.download("^VIX",start=START,end=END,auto_adjust=False,progress=False,threads=False)
    if isinstance(spy.columns,pd.MultiIndex): spy.columns=spy.columns.get_level_values(0)
    if isinstance(vix.columns,pd.MultiIndex): vix.columns=vix.columns.get_level_values(0)
    if len(spy)<2500 or len(vix)<2500: raise RuntimeError(f"Insufficient data: SPY={len(spy)} VIX={len(vix)}")
    d=pd.DataFrame(index=spy.index)
    d["close"]=spy["Close"]; d["high"]=spy["High"]; d["low"]=spy["Low"]; d["volume"]=spy["Volume"]
    d["vix"]=vix["Close"].reindex(d.index).ffill(limit=3)
    d["r1"]=np.log(d["close"]).diff()
    for n in [5,10,20]:
        d[f"mom{n}"]=np.log(d["close"]/d["close"].shift(n))
        d[f"rv{n}"]=d["r1"].rolling(n).std()*np.sqrt(252)
    d["range20"]=(np.log(d["high"]/d["low"])**2).rolling(20).mean().pow(.5)*np.sqrt(252/(4*np.log(2)))
    lv=np.log(d["volume"])
    d["volz20"]=(lv-lv.rolling(20).mean())/lv.rolling(20).std()
    # Real future 5-session annualized realized volatility
    sq=d["r1"]**2
    future=sum(sq.shift(-k) for k in range(1,HORIZON+1))
    d["target"]=np.sqrt(future*252/HORIZON)
    d=d.replace([np.inf,-np.inf],np.nan).dropna()
    feats=["r1","mom5","mom10","mom20","rv5","rv10","rv20","range20","volz20","vix"]
    return d,feats

class MLP(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,h),nn.Tanh(),nn.Linear(h,1),nn.Softplus())
    def forward(self,x): return self.net(x).squeeze(-1)

class SoftMoE(nn.Module):
    def __init__(self,d,h,B):
        super().__init__()
        self.experts=nn.ModuleList([MLP(d,h) for _ in range(B)])
        self.gate=nn.Sequential(nn.Linear(d,max(4,h//2)),nn.Tanh(),nn.Linear(max(4,h//2),B))
    def forward(self,x):
        g=torch.softmax(self.gate(x),dim=1)
        e=torch.stack([m(x) for m in self.experts],dim=1)
        return (g*e).sum(1)

def scale(train,val,test,feats):
    mu=train[feats].mean(); sd=train[feats].std().replace(0,1)
    return [(z[feats]-mu)/sd for z in [train,val,test]],mu,sd

def fit_torch(model,X,y,Xv,yv,seed,epochs=1200,patience=100,lr=2e-3,wd=1e-5):
    seed_all(seed)
    X=torch.tensor(np.asarray(X),dtype=torch.float32); y=torch.tensor(np.asarray(y),dtype=torch.float32)
    Xv=torch.tensor(np.asarray(Xv),dtype=torch.float32); yv=torch.tensor(np.asarray(yv),dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    best=1e99; state=None; stale=0
    for ep in range(epochs):
        model.train(); opt.zero_grad(); p=model(X); loss=((p-y)**2).mean(); loss.backward(); opt.step()
        if ep%5==0:
            model.eval()
            with torch.no_grad(): vl=((model(Xv)-yv)**2).mean().item()
            if vl<best-1e-9:
                best=vl; state={k:v.detach().clone() for k,v in model.state_dict().items()}; stale=0
            else:
                stale+=5
            if stale>=patience: break
    model.load_state_dict(state)
    return model

def predict(model,X):
    model.eval()
    with torch.no_grad(): return model(torch.tensor(np.asarray(X),dtype=torch.float32)).cpu().numpy()

def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
def mae(y,p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))

def rt_fit_predict(train,val,test,Xtr,Xv,Xte,ytr,yv,B,h,seed):
    cuts=np.quantile(train["vix"],np.linspace(0,1,B+1))[1:-1]
    bt=np.searchsorted(cuts,train["vix"],side="right")
    bv=np.searchsorted(cuts,val["vix"],side="right")
    be=np.searchsorted(cuts,test["vix"],side="right")
    pv=np.full(len(val),np.nan); pe=np.full(len(test),np.nan)
    counts=[]
    for b in range(B):
        it=bt==b; iv=bv==b; ie=be==b
        counts.append({"band":b,"train":int(it.sum()),"val":int(iv.sum()),"test":int(ie.sum())})
        if it.sum()<80 or iv.sum()<10: return None,None,counts
        m=MLP(Xtr.shape[1],h)
        m=fit_torch(m,Xtr[it],ytr[it],Xv[iv],yv[iv],seed+100*b)
        if iv.any(): pv[iv]=predict(m,Xv[iv])
        if ie.any(): pe[ie]=predict(m,Xte[ie])
    if not np.isfinite(pv).all() or not np.isfinite(pe).all(): return None,None,counts
    return pv,pe,counts

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="real_results"); ap.add_argument("--seeds",type=int,default=20)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    d,feats=download()
    train=d[d.index<"2020-01-01"].copy()
    val=d[(d.index>="2020-01-01")&(d.index<"2023-01-01")].copy()
    test=d[d.index>="2023-01-01"].copy()
    (Xs,mu,sd)=scale(train,val,test,feats); Xtr,Xv,Xte=[x.to_numpy(dtype=np.float32) for x in Xs]
    ytr=train["target"].to_numpy(np.float32); yv=val["target"].to_numpy(np.float32); yte=test["target"].to_numpy(np.float32)
    meta={"n_train":len(train),"n_val":len(val),"n_test":len(test),"test_end":str(test.index.max().date()),"features":feats}
    (out/"meta.json").write_text(json.dumps(meta,indent=2))
    rows=[]; supports=[]
    # tune shared MLP on validation once per seed over widths
    for seed in range(a.seeds):
        best=None
        for h in [8,16,32,64,128]:
            m=fit_torch(MLP(len(feats),h),Xtr,ytr,Xv,yv,1000+seed)
            pv=predict(m,Xv)
            z=rmse(yv,pv)
            if best is None or z<best[0]: best=(z,h,m)
        _,h,m=best; p=predict(m,Xte)
        rows.append(["MLP",1,h,seed,rmse(yte,p),mae(yte,p)])
    # R(T) hard bands and soft MoE. Tune h per B on validation, separately each seed.
    for B in [2,3,4,5,6,8]:
        for seed in range(a.seeds):
            best=None
            for h in [4,8,16,32]:
                pv,pe,cnt=rt_fit_predict(train,val,test,Xtr,Xv,Xte,ytr,yv,B,h,2000+seed)
                if pv is None: continue
                z=rmse(yv,pv)
                if best is None or z<best[0]: best=(z,h,pe,cnt)
            if best is not None:
                _,h,pe,cnt=best
                rows.append(["RT-hard",B,h,seed,rmse(yte,pe),mae(yte,pe)])
                if seed==0:
                    for c in cnt: supports.append({"model":"RT-hard","B":B,**c})
            # classical learned soft mixture of experts
            bestm=None
            for h in [4,8,16,32]:
                m=fit_torch(SoftMoE(len(feats),h,B),Xtr,ytr,Xv,yv,3000+seed)
                pv=predict(m,Xv); z=rmse(yv,pv)
                if bestm is None or z<bestm[0]: bestm=(z,h,m)
            _,h,m=bestm; pe=predict(m,Xte)
            rows.append(["Soft-MoE",B,h,seed,rmse(yte,pe),mae(yte,pe)])
    res=pd.DataFrame(rows,columns=["model","B","width","seed","rmse","mae"])
    res.to_csv(out/"real_market_results.csv",index=False)
    pd.DataFrame(supports).to_csv(out/"band_support.csv",index=False)
    summary=res.groupby(["model","B"]).agg(rmse_mean=("rmse","mean"),rmse_sd=("rmse","std"),mae_mean=("mae","mean"),n=("seed","count")).reset_index()
    summary.to_csv(out/"summary.csv",index=False)
    print(meta); print(summary.to_string(index=False))

if __name__=="__main__": main()
