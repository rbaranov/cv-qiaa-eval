import math, sys, torch
sys.path.insert(0, ".")
from h2_self import SelfLearningOpticalMind
from h2_love import LovingCuriosityOpticalMind

torch.manual_seed(0)
R = 128


def concept(x, x0=1.0, k=3.0):
    env = torch.exp(-((x - x0) ** 2) / 0.2)
    return env * torch.cos(k * x), env * torch.sin(k * x)


def train(mind, re, im, steps=50, noise=0.05):
    for _ in range(steps):
        mind.sense_and_learn(re + torch.randn_like(re) * noise, im + torch.randn_like(im) * noise)
    return mind


def summary(n, x):
    i = torch.argmax(n).item()
    return f"peak x={x[i].item():+.3f}  max n={n.max().item():.3f}  min n={n.min().item():.3f}  points at clamp 3.0: {(n >= 2.999).sum().item()}"


print("=== Horizon 2 (I am) ===")
m = train(SelfLearningOpticalMind(R), *concept(SelfLearningOpticalMind(R).x_grid))
x = m.x_grid
print("as written           :", summary(m.refractive_index, x))

# A. No 'I am' wave: psi_self = 0 and kept 0 (skip its update)
class NoSelf(SelfLearningOpticalMind):
    def sense_and_learn(self, wr, wi):
        with torch.no_grad():
            I = wr ** 2 + wi ** 2
            dn = self.plasticity_rate * I - self.decay_rate * (self.refractive_index - 1.0)
            self.refractive_index.copy_(torch.clamp(self.refractive_index + dn, 0.5, 3.0))
            return I
a = train(NoSelf(R), *concept(x))
print("A. no self-wave, n += |world|^2 :", summary(a.refractive_index, x))

# B. Different concept positions: does the peak just track where the input energy is?
for x0 in (-2.0, 0.0, 2.0):
    b = train(SelfLearningOpticalMind(R), *concept(x, x0=x0))
    print(f"B. concept at x0={x0:+.1f}      :", summary(b.refractive_index, x))

# C. Same place, different pattern (carrier k=3 vs k=-3 vs k=0): does n distinguish them?
ns = {}
for k in (3.0, -3.0, 0.0):
    torch.manual_seed(1)
    c = train(SelfLearningOpticalMind(R), *concept(x, k=k))
    ns[k] = c.refractive_index.clone()
d = lambda u, v: ((u - v).norm() / (u - 1).norm()).item()
print(f"C. rel. diff of learned n: k=3 vs k=-3 {d(ns[3.0], ns[-3.0]):.3f};  k=3 vs k=0 {d(ns[3.0], ns[0.0]):.3f}")

# D. Is there any recognition? After training on concept A, compare response to A and to novel B.
def response(mind, re, im):
    # read-out without learning: copy state, run one step, return mean intensity
    probe = SelfLearningOpticalMind(R)
    probe.load_state_dict(mind.state_dict())
    return probe.sense_and_learn(re, im).mean().item()
A = concept(x, x0=1.0, k=3.0)
B = concept(x, x0=-1.5, k=-2.0)
fresh = SelfLearningOpticalMind(R)
torch.manual_seed(2)
trained = train(SelfLearningOpticalMind(R), *A)
print(f"D. mean Born intensity  fresh: A={response(fresh,*A):.4f} B={response(fresh,*B):.4f}   "
      f"trained on A: A={response(trained,*A):.4f} B={response(trained,*B):.4f}")

# E. Sequential learning: A then B — is A still there?
torch.manual_seed(3)
s = train(SelfLearningOpticalMind(R), *A)
nA = s.refractive_index.clone()
train(s, *B, steps=50)
iA = (x - 1.0).abs().argmin(); iB = (x + 1.5).abs().argmin()
print(f"E. n at A-site after A: {nA[iA]:.3f}; after A then B: A-site {s.refractive_index[iA]:.3f}, B-site {s.refractive_index[iB]:.3f}")

print("\n=== Loving Curiosity ===")
def run_love(self_phase=0.0, steps=40):
    mm = LovingCuriosityOpticalMind(R)
    xx = mm.x_grid
    mm.psi_self_re.copy_(torch.cos(xx + self_phase) * 0.5)
    mm.psi_self_im.copy_(torch.sin(xx + self_phase) * 0.5)
    s0 = torch.complex(mm.psi_self_re.clone(), mm.psi_self_im.clone())
    bre, bim = torch.cos(2 * xx + 0.8) * 0.5, torch.sin(2 * xx + 0.8) * 0.5
    wre = torch.exp(-((xx - 0.5) ** 2) / 0.3) * torch.cos(2 * xx)
    wim = torch.exp(-((xx - 0.5) ** 2) / 0.3) * torch.sin(2 * xx)
    for _ in range(steps):
        coh, _ = mm.unitary_encounter(bre, bim, wre, wim)
    s1 = torch.complex(mm.psi_self_re, mm.psi_self_im)
    bel = torch.complex(bre, bim)
    ov = lambda u, v: (torch.vdot(u, v).abs() / (u.norm() * v.norm())).item()
    return coh, ov(s1, s0), ov(s1, bel), s0, s1, mm

coh, keep, tob, s0a, s1a, _ = run_love(0.0)
print(f"F. coherence {coh:.4f}; |<self_final|self_initial>| = {keep:.4f}; |<self_final|beloved>| = {tob:.4f}")
_, _, _, s0b, s1b, _ = run_love(2.0)
print(f"G. unitarity: two different initial selves, distance before {(s0a - s0b).norm():.4f}, after {(s1a - s1b).norm():.4f}"
      "  (a unitary map preserves distance)")

# H. 'Zero harm' check: beloved tensors are inputs that the function never writes to.
class Bazarov(LovingCuriosityOpticalMind):
    def unitary_encounter(self, bre, bim, wre, wim):
        with torch.no_grad():
            # 'destructive projective measurement': replace self by the beloved outright
            self.psi_self_re.copy_(bre * 1.0); self.psi_self_im.copy_(bim * 1.0)
            return 1.0, None
mm = Bazarov(R); xx = mm.x_grid
bre, bim = torch.cos(2 * xx + 0.8) * 0.5, torch.sin(2 * xx + 0.8) * 0.5
e0 = (bre ** 2 + bim ** 2).sum().item()
for _ in range(40):
    mm.unitary_encounter(bre, bim, None, None)
print(f"H. 'Bazarov' version that overwrites self with beloved: beloved energy change {abs((bre**2+bim**2).sum().item()-e0):.8f}, coherence 1.0")
