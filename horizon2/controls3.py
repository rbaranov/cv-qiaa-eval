import sys, torch
sys.path.insert(0, ".")
from h2_self import SelfLearningOpticalMind
from controls import concept, train, response, NoSelf, R
x = SelfLearningOpticalMind(R).x_grid
def plateau(n):
    m = n.max(); idx = (n >= m - 1e-3).nonzero().flatten()
    w = (n - 1).clamp(min=0)
    return f"plateau {x[idx[0]]:+.2f}..{x[idx[-1]]:+.2f} (centre {x[idx].mean():+.2f}, {len(idx)} pts), centroid {(w*x).sum()/w.sum():+.2f}"
print("orig  x0=1:", plateau(train(SelfLearningOpticalMind(R), *concept(x)).refractive_index))
print("A no-self:", plateau(train(NoSelf(R), *concept(x)).refractive_index))
for x0 in (-2.0, 0.0, 2.0):
    print(f"B x0={x0:+.0f}:", plateau(train(SelfLearningOpticalMind(R), *concept(x, x0=x0)).refractive_index))
pats = {"A k=3": concept(x,1,3), "k=-3": concept(x,1,-3), "k=0": concept(x,1,0)}
def interf(m, p):
    with torch.no_grad():
        sre, sim = m.psi_self_real.clone(), m.psi_self_imag.clone()
    I = response(m, *p)
    return I - (sre**2+sim**2).mean().item() - (p[0]**2+p[1]**2).mean().item()
print("background mean|self|^2 = 0.25; mean|world|^2 =", f"{(pats['A k=3'][0]**2+pats['A k=3'][1]**2).mean():.3f}")
for steps in (0, 50, 1000):
    rows = []
    for seed in range(5):
        torch.manual_seed(seed)
        m = train(SelfLearningOpticalMind(R), *pats["A k=3"], steps=steps) if steps else SelfLearningOpticalMind(R)
        rows.append([interf(m, p) for p in pats.values()])
    t = torch.tensor(rows)
    print(f"steps {steps:4d}: interference term mean±sd over 5 seeds: " + "  ".join(f"{k} {mu:+.3f}±{sd:.3f}" for k, mu, sd in zip(pats, t.mean(0), t.std(0))))
