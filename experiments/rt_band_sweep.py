import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

R=.02
SIG=np.array([.12,.25,.50])
Q=np.array([[-1.20,1.00,.20],[.35,-.95,.60],[.15,.85,-1.00]])
TS=np.linspace(.05,2.,20)
XS=np.linspace(np.log(.65),np.log(1.45),61)
HELD_IDX=[3,7,11,15,19]
VAL_IDX=[2,8,14]
FEAT=["x","T","civ"]

def conditional_variance():
    H=np.zeros((4,4)); H[:3,:3]=Q; H[:3,3]=SIG**2
    V=np.vstack([expm(H*t)[:3,3] for t in TS])
    assert np.isfinite(V).all() and (V>0).all() and (np.diff(V,axis=0)>0).all()
    return V

def price_surface(nw=151):
    xw=np.linspace(np.log(.15),np.log(4.),nw)
    h=xw[1]-xw[0]; ni=nw-2; n=3*ni
    A=np.zeros((n,n))
    def ix(i,k): return k*ni+i-1
    ups=[]
    for k in range(3):
        mu=R-.5*SIG[k]**2
        lo=.5*SIG[k]**2/h**2-mu/(2*h)
        ce=-SIG[k]**2/h**2-R
        up=.5*SIG[k]**2/h**2+mu/(2*h)
        ups.append(up)
        for i in range(1,nw-1):
            z=ix(i,k); A[z,z]+=ce
            if i>1: A[z,ix(i-1,k)]+=lo
            if i<nw-2: A[z,ix(i+1,k)]+=up
            for ell in range(3): A[z,ix(i,ell)]+=Q[k,ell]
    def rhs(t,u):
        b=np.zeros(n); upper=np.exp(xw[-1])-np.exp(-R*t)
        for k in range(3): b[ix(nw-2,k)]+=ups[k]*upper
        return A@u+b
    u0=np.tile(np.maximum(np.exp(xw[1:-1])-1.,0.),3)
    sol=solve_ivp(rhs,(0,2.),u0,t_eval=TS,method="BDF",rtol=3e-7,atol=1e-9)
    if not sol.success: raise RuntimeError(sol.message)
    P=np.empty((len(TS),3,len(XS)))
    for j in range(len(TS)):
        for k in range(3):
            P[j,k]=np.interp(XS,xw[1:-1],sol.y[k*ni:(k+1)*ni,j])
    S=np.exp(XS)
    assert np.isfinite(P).all() and P.min()>=-1e-8
    assert (P<=S[None,None,:]+1e-7).all()
    assert (np.diff(P,axis=2)>=-5e-6).all()
    return P

def dataset():
    V=conditional_variance(); P=price_surface()
    rec=[]
    for j,t in enumerate(TS):
        for k in range(3):
            for q,x in enumerate(XS):
                rec.append((float(x),float(t),int(k),float(P[j,k,q]),float(V[j,k])))
    d=pd.DataFrame(rec,columns=["x","T","regime","y","civ"])
    held=TS[HELD_IDX]; valt=TS[VAL_IDX]
    test=d[np.isin(d["T"].to_numpy(),held)].reset_index(drop=True)
    tr=d[~np.isin(d["T"].to_numpy(),held)].reset_index(drop=True)
    val=tr[np.isin(tr["T"].to_numpy(),valt)].reset_index(drop=True)
    fit=tr[~np.isin(tr["T"].to_numpy(),valt)].reset_index(drop=True)
    return fit,val,test

def expert_run(fit,val,test,B,width,alpha,seed):
    cuts=np.quantile(fit["civ"].to_numpy(),np.linspace(0,1,B+1))[1:-1]
    bf=np.searchsorted(cuts,fit["civ"].to_numpy(),"right")
    bv=np.searchsorted(cuts,val["civ"].to_numpy(),"right")
    bt=np.searchsorted(cuts,test["civ"].to_numpy(),"right")
    pv=np.full(len(val),np.nan); pt=np.full(len(test),np.nan)
    for b in range(B):
        mf=bf==b; mv=bv==b; mt=bt==b
        if (mv.any() or mt.any()) and mf.sum()<40: return None
        if not (mv.any() or mt.any()): continue
        X=fit.loc[mf,FEAT].to_numpy(); y=fit.loc[mf,"y"].to_numpy()
        sc=StandardScaler().fit(X)
        model=MLPRegressor(hidden_layer_sizes=(width,),activation="tanh",
            solver="lbfgs",alpha=alpha,max_iter=1000,tol=1e-9,
            random_state=seed+1009*b)
        model.fit(sc.transform(X),y)
        if mv.any(): pv[mv]=model.predict(sc.transform(val.loc[mv,FEAT]))
        if mt.any(): pt[mt]=model.predict(sc.transform(test.loc[mt,FEAT]))
    if not np.isfinite(pv).all() or not np.isfinite(pt).all(): return None
    vr=np.sqrt(np.mean((pv-val["y"].to_numpy())**2))
    tr=np.sqrt(np.mean((pt-test["y"].to_numpy())**2))
    return vr,tr

def shared_run(fit,val,test,width,alpha,seed):
    def X(d):
        return np.c_[d[FEAT].to_numpy(),np.eye(3)[d["regime"].to_numpy(dtype=int)]]
    xf,xv,xt=X(fit),X(val),X(test)
    sc=StandardScaler().fit(xf)
    m=MLPRegressor(hidden_layer_sizes=(width,),activation="tanh",solver="lbfgs",
        alpha=alpha,max_iter=1200,tol=1e-9,random_state=seed)
    m.fit(sc.transform(xf),fit["y"].to_numpy())
    pv=m.predict(sc.transform(xv)); pt=m.predict(sc.transform(xt))
    return np.sqrt(np.mean((pv-val["y"].to_numpy())**2)),np.sqrt(np.mean((pt-test["y"].to_numpy())**2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--B",type=int,required=True)
    ap.add_argument("--out",default="results")
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    fit,val,test=dataset()
    widths=[4,8,16,32,64]; alphas=[1e-8,1e-6,1e-4,1e-2]
    screen=[]
    for w in widths:
        for reg in alphas:
            z=[expert_run(fit,val,test,a.B,w,reg,100+s) for s in range(5)]
            if all(t is not None for t in z):
                screen.append((w,reg,float(np.mean([t[0] for t in z]))))
    if not screen: raise RuntimeError("No valid expert configuration")
    w,reg,_=min(screen,key=lambda z:z[2])
    rows=[]
    for s in range(20):
        z=expert_run(fit,val,test,a.B,w,reg,10000+s)
        if z is None: raise RuntimeError("Invalid final expert run")
        rows.append((a.B,w,reg,s,z[0],z[1]))
    pd.DataFrame(rows,columns=["B","width","alpha","seed","val_rmse","test_rmse"]).to_csv(out/f"B{a.B}.csv",index=False)
    pd.DataFrame(screen,columns=["width","alpha","screen_val_rmse"]).to_csv(out/f"B{a.B}_screen.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__": main()
