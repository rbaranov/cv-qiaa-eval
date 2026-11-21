import sys, torch
sys.path.insert(0, ".")
from h2_self import SelfLearningOpticalMind
from controls import concept, train, response, R
x = SelfLearningOpticalMind(R).x_grid
A = concept(x, 1.0, 3.0); A2 = concept(x, 1.0, -3.0); A3 = concept(x, 1.0, 0.0)
for seed in range(3):
    torch.manual_seed(seed); t = train(SelfLearningOpticalMind(R), *A)
    f = SelfLearningOpticalMind(R)
    print(f"D2 seed{seed}: fresh A={response(f,*A):.3f} k-3={response(f,*A2):.3f} k0={response(f,*A3):.3f} | trained on A: A={response(t,*A):.3f} k-3={response(t,*A2):.3f} k0={response(t,*A3):.3f}")
for steps in (50, 200, 1000):
    torch.manual_seed(0); t = train(SelfLearningOpticalMind(R), *A, steps=steps)
    n = t.refractive_index
    print(f"S. {steps} steps: points at clamp {(n>=2.999).sum().item()}/128, min n {n.min():.2f}, mean n {n.mean():.2f}")
torch.manual_seed(0); t = train(SelfLearningOpticalMind(R), *A, steps=1000)
print(f"S. after 1000 steps response A={response(t,*A):.3f} k-3={response(t,*A2):.3f}")
