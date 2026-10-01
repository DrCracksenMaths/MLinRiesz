import sys, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from cifar10_rt_modulated import RTFiLMAdapterCNN, B, seed_all
from cifar10_exact_rt_ablation import AlgebraicAdapter

torch.set_num_threads(1)
DT=torch.float64

def copy_rt_to_alg(rt,alg):
    alg.stem.load_state_dict(rt.stem.state_dict())
    alg.head.load_state_dict(rt.head.state_dict())
    # FiLM ordering: alg rows are [gamma all channels, beta all channels],
    # columns [onehot bands, onehot*severity].
    off=0; total=sum(rt.stem.channels)
    with torch.no_grad():
        alg.film.weight.zero_()
        for film,c in zip(rt.films,rt.stem.channels):
            for j in range(B):
                alg.film.weight[off:off+c,j].copy_(film.ga[j])
                alg.film.weight[off:off+c,B+j].copy_(film.gs[j])
                alg.film.weight[total+off:total+off+c,j].copy_(film.ba[j])
                alg.film.weight[total+off:total+off+c,B+j].copy_(film.bs[j])
            off+=c
        for i,ad in enumerate(rt.adapters):
            for j in range(B):
                alg.down[i][j].copy_(ad.down[j].weight[:, :, 0, 0])
                alg.up[i][j].copy_(ad.up[j].weight[:, :, 0, 0])
                alg.scale[j,0].copy_(ad.a[j]-1.)
                alg.scale[j,1].copy_(ad.b[j])

def maxdiff(a,b): return (a-b).abs().max().item()

def compare(seed=123,w=20):
    seed_all(seed)
    rt=RTFiLMAdapterCNN(w).to(dtype=DT)
    alg=AlgebraicAdapter(w).to(dtype=DT)
    copy_rt_to_alg(rt,alg)
    rt.eval(); alg.eval()
    # BN eval avoids running-stat mutation; identity is layerwise/functional.
    torch.manual_seed(seed+999)
    x=torch.randn(12,3,32,32,dtype=DT)
    k=torch.tensor([0,1,2,3]*3)
    s=torch.linspace(.05,.95,12,dtype=DT)
    y=torch.arange(12)%10
    zr=rt(x,k,s); za=alg(x,k,s)
    d0=maxdiff(zr,za)
    lr=F.cross_entropy(zr,y); la=F.cross_entropy(za,y)
    dl=abs(lr.item()-la.item())
    rt.zero_grad(); alg.zero_grad(); lr.backward(); la.backward()
    # Compare gradients after mapping. Shared stem/head first.
    gd=[]
    for (n,p),(m,q) in zip(rt.stem.named_parameters(),alg.stem.named_parameters()):
        gd.append(maxdiff(p.grad,q.grad))
    for (n,p),(m,q) in zip(rt.head.named_parameters(),alg.head.named_parameters()):
        gd.append(maxdiff(p.grad,q.grad))
    # Explicit mapped FiLM/adapters.
    total=sum(rt.stem.channels); off=0
    Wg=alg.film.weight.grad
    for film,c in zip(rt.films,rt.stem.channels):
        for j in range(B):
            gd += [maxdiff(film.ga.grad[j],Wg[off:off+c,j]),
                   maxdiff(film.gs.grad[j],Wg[off:off+c,B+j]),
                   maxdiff(film.ba.grad[j],Wg[total+off:total+off+c,j]),
                   maxdiff(film.bs.grad[j],Wg[total+off:total+off+c,B+j])]
        off+=c
    for i,ad in enumerate(rt.adapters):
        for j in range(B):
            gd += [maxdiff(ad.down[j].weight.grad[:,:,0,0],alg.down[i].grad[j]),
                   maxdiff(ad.up[j].weight.grad[:,:,0,0],alg.up[i].grad[j]),
                   abs(ad.a.grad[j].item()-alg.scale.grad[j,0].item()),
                   abs(ad.b.grad[j].item()-alg.scale.grad[j,1].item())]
    dg=max(gd)
    print({"seed":seed,"forward_max_abs":d0,"loss_abs":dl,"gradient_max_abs":dg})
    assert d0 < 1e-10 and dl < 1e-12 and dg < 1e-9, "identity test failed"

if __name__=="__main__":
    for s in [0,1,17]: compare(s)
    print("PASS: RT-Adapter and algebraic tensor realization are identical under the explicit parameter map (within float64 tolerance).")
