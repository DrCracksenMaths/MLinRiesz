import argparse, json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rotated_mnist import (
    RotMNIST, SharedCNN, SoftMoE, RTHard,
    seed_all, train_model, evaluate, param_count
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--out",default="matched_results")
    ap.add_argument("--train-n",type=int,default=24000)
    ap.add_argument("--val-n",type=int,default=6000)
    ap.add_argument("--test-n",type=int,default=10000)
    args=ap.parse_args()

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    seed=args.seed
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

    # Matched capacities:
    # RT-hard B=4,width=16: 827,688 parameters
    # Shared CNN width=60: ~819,578 parameters
    # Soft-MoE B=4,width=60: ~832,028 parameters
    specs=[
        ("CNN-matched",1,SharedCNN(60)),
        ("RT-hard",4,RTHard(4,16)),
        ("Soft-MoE-matched",4,SoftMoE(4,60)),
    ]

    rows=[]
    for name,B,model in specs:
        model,_=train_model(model,tl,vl,epochs=8)
        for split,loader in [("seen",tsl),("unseen_angles",tul)]:
            z=evaluate(model,loader,B if B>1 else None)
            rows.append({
                "model":name,"B":B,"seed":seed,"split":split,
                "accuracy":z["accuracy"],"nll":z["nll"],
                "params":param_count(model)
            })

    df=pd.DataFrame(rows)
    df.to_csv(out/f"seed_{seed:02d}.csv",index=False)
    print(df.to_string(index=False))

if __name__=="__main__":
    main()
