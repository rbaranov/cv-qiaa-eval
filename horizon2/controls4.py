# Numbers quoted in the reply that controls.py..controls3.py do not print directly.
import io, contextlib, sys, math, torch
sys.path.insert(0, ".")
from h2_self import SelfLearningOpticalMind
from h2_love import LovingCuriosityOpticalMind
with contextlib.redirect_stdout(io.StringIO()):
    from controls import concept, train, NoSelf, R

x = SelfLearningOpticalMind(R).x_grid

print("=== Horizon 2 ===")
# 1. Peak without the 'I am' wave, as the author's own argmax reports it, and without the 3.0 clamp.
for seed in range(3):
    torch.manual_seed(seed)
    n = train(NoSelf(R), *concept(x)).refractive_index
    top = (n >= n.max() - 1e-3).nonzero().flatten()
    print(f"1. no self-wave, seed {seed}: argmax x={x[n.argmax()]:+.3f}; plateau n=3.0 from {x[top[0]]:+.3f} to {x[top[-1]]:+.3f}")

class NoSelfNoClamp(NoSelf):
    def sense_and_learn(self, wr, wi):
        with torch.no_grad():
            I = wr ** 2 + wi ** 2
            self.refractive_index.add_(self.plasticity_rate * I - self.decay_rate * (self.refractive_index - 1.0))
            return I
torch.manual_seed(0)
n = train(NoSelfNoClamp(R), *concept(x)).refractive_index
print(f"1. no self-wave, no clamp: argmax x={x[n.argmax()]:+.3f}")

# 2. Peak offset when the input is placed elsewhere (author's argmax).
for x0 in (-2.0, -1.0, 0.0, 1.0, 2.0):
    offs = []
    for seed in range(3):
        torch.manual_seed(seed)
        n = train(SelfLearningOpticalMind(R), *concept(x, x0=x0)).refractive_index
        offs.append(x[n.argmax()].item() - x0)
    print(f"2. input at x0={x0:+.0f}: argmax - x0 = " + ", ".join(f"{o:+.2f}" for o in offs))

# 3. Without the 'I am' wave the intensity is |world|^2: the phase is lost, three patterns look the same.
torch.manual_seed(0)
m = train(NoSelf(R), *concept(x))
for k in (3.0, -3.0, 0.0):
    p = NoSelf(R); p.load_state_dict(m.state_dict())
    print(f"3. no self-wave, probe k={k:+.0f}: mean intensity {p.sense_and_learn(*concept(x, k=k)).mean():.4f}")

print("\n=== Loving Curiosity ===")
bre, bim = torch.cos(2 * x + 0.8) * 0.5, torch.sin(2 * x + 0.8) * 0.5
wre = torch.exp(-((x - 0.5) ** 2) / 0.3) * torch.cos(2 * x)
wim = torch.exp(-((x - 0.5) ** 2) / 0.3) * torch.sin(2 * x)
bel = torch.complex(bre, bim)
coh = lambda s: ((s.real * bre + s.imag * bim).sum() / (s.abs().norm() * bel.abs().norm())).item()

# 4. Phase rotation alone (no blending): coherence after 40 steps.
class RotationOnly(LovingCuriosityOpticalMind):
    def unitary_encounter(self, br, bi, wr, wi):
        with torch.no_grad():
            o_re = (self.psi_self_re * br + self.psi_self_im * bi).mean()
            o_im = (self.psi_self_im * br - self.psi_self_re * bi).mean()
            sh = -self.attunement_rate * torch.atan2(o_im, o_re)
            r = self.psi_self_re * torch.cos(sh) - self.psi_self_im * torch.sin(sh)
            i = self.psi_self_re * torch.sin(sh) + self.psi_self_im * torch.cos(sh)
            self.psi_self_re.copy_(r); self.psi_self_im.copy_(i)
for cls in (LovingCuriosityOpticalMind, RotationOnly):
    mm = cls(R)
    for _ in range(40):
        mm.unitary_encounter(bre, bim, wre, wim)
    print(f"4. {cls.__name__}: coherence after 40 steps {coh(torch.complex(mm.psi_self_re, mm.psi_self_im)):.4f}")

# 5. Distance between different initial selves before and after 40 steps (a unitary map keeps it).
def run(s_re, s_im):
    mm = LovingCuriosityOpticalMind(R)
    mm.psi_self_re.copy_(s_re); mm.psi_self_im.copy_(s_im)
    s0 = torch.complex(s_re.clone(), s_im.clone())
    for _ in range(40):
        mm.unitary_encounter(bre, bim, wre, wim)
    return s0, torch.complex(mm.psi_self_re, mm.psi_self_im)
starts = [(torch.cos(q * x + ph) * 0.5, torch.sin(q * x + ph) * 0.5) for q, ph in ((1, 0), (1, 2), (3, 1), (-1, 0.5), (2, 2.5), (0, 1))]
runs = [run(*s) for s in starts]
before = after = 0.0
for i in range(len(runs)):
    for j in range(i + 1, len(runs)):
        before += (runs[i][0] - runs[j][0]).norm().item(); after += (runs[i][1] - runs[j][1]).norm().item()
pairs = len(runs) * (len(runs) - 1) / 2
print(f"5. mean distance over {int(pairs)} pairs of initial selves: before {before/pairs:.2f}, after {after/pairs:.2f}, ratio {before/after:.1f}")
torch.manual_seed(0)
rnd = [run(*(lambda v: (v.real, v.imag))(torch.complex(torch.randn(R), torch.randn(R)) * 0.5 / math.sqrt(2))) for _ in range(6)]
b = a = 0.0
for i in range(6):
    for j in range(i + 1, 6):
        b += (rnd[i][0] - rnd[j][0]).norm().item(); a += (rnd[i][1] - rnd[j][1]).norm().item()
print(f"5. random initial selves, 15 pairs: before {b/15:.2f}, after {a/15:.2f}, ratio {b/a:.1f}")
