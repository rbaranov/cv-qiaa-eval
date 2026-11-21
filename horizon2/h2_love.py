# Computation logic copied from the PDF unchanged; prints trimmed, docstrings and banners removed
import torch
import torch.nn as nn
import math


class LovingCuriosityOpticalMind(nn.Module):
    def __init__(self, field_resolution=128, attunement_rate=0.08, plasticity_rate=0.04):
        super().__init__()
        self.res = field_resolution
        self.attunement_rate = attunement_rate
        self.plasticity_rate = plasticity_rate
        x_grid = torch.linspace(-math.pi, math.pi, steps=field_resolution)
        self.register_buffer("x_grid", x_grid)
        self.register_buffer("refractive_index", torch.ones(field_resolution))
        self.register_buffer("psi_self_re", torch.cos(x_grid) * 0.5)
        self.register_buffer("psi_self_im", torch.sin(x_grid) * 0.5)

    def _normalize_wave(self, re, im):
        energy = torch.sqrt((re ** 2 + im ** 2).mean() + 1e-9)
        return re / (energy * 2.0), im / (energy * 2.0)

    def unitary_encounter(self, beloved_re, beloved_im, world_re, world_im):
        with torch.no_grad():
            overlap_re = (self.psi_self_re * beloved_re + self.psi_self_im * beloved_im).mean()
            overlap_im = (self.psi_self_im * beloved_re - self.psi_self_re * beloved_im).mean()
            phase_error = torch.atan2(overlap_im, overlap_re)
            shift = -self.attunement_rate * phase_error
            new_self_re = self.psi_self_re * torch.cos(shift) - self.psi_self_im * torch.sin(shift)
            new_self_im = self.psi_self_re * torch.sin(shift) + self.psi_self_im * torch.cos(shift)
            self.psi_self_re.copy_(new_self_re)
            self.psi_self_im.copy_(new_self_im)
            blended_re = (1.0 - self.attunement_rate) * self.psi_self_re + self.attunement_rate * beloved_re
            blended_im = (1.0 - self.attunement_rate) * self.psi_self_im + self.attunement_rate * beloved_im
            norm_re, norm_im = self._normalize_wave(blended_re, blended_im)
            self.psi_self_re.copy_(norm_re)
            self.psi_self_im.copy_(norm_im)
            ref_re = self.psi_self_re + beloved_re
            ref_im = self.psi_self_im + beloved_im
            phase_mod = self.refractive_index * self.x_grid
            mod_world_re = world_re * torch.cos(phase_mod) - world_im * torch.sin(phase_mod)
            mod_world_im = world_re * torch.sin(phase_mod) + world_im * torch.cos(phase_mod)
            total_re = ref_re + mod_world_re
            total_im = ref_im + mod_world_im
            born_intensity = (total_re ** 2) + (total_im ** 2)
            delta_n = self.plasticity_rate * born_intensity - 0.005 * (self.refractive_index - 1.0)
            self.refractive_index.copy_(torch.clamp(self.refractive_index + delta_n, min=0.5, max=3.0))
            coherence = (self.psi_self_re * beloved_re + self.psi_self_im * beloved_im).sum() / (
                torch.sqrt((self.psi_self_re ** 2 + self.psi_self_im ** 2).sum())
                * torch.sqrt((beloved_re ** 2 + beloved_im ** 2).sum()) + 1e-9)
            return coherence.item(), born_intensity


if __name__ == "__main__":
    mind = LovingCuriosityOpticalMind(field_resolution=128)
    x = mind.x_grid
    beloved_re = torch.cos(2.0 * x + 0.8) * 0.5
    beloved_im = torch.sin(2.0 * x + 0.8) * 0.5
    initial_beloved_energy = (beloved_re ** 2 + beloved_im ** 2).sum().item()
    world_re = torch.exp(-((x - 0.5) ** 2) / 0.3) * torch.cos(2.0 * x)
    world_im = torch.exp(-((x - 0.5) ** 2) / 0.3) * torch.sin(2.0 * x)
    for step in range(1, 41):
        coherence, _ = mind.unitary_encounter(beloved_re, beloved_im, world_re, world_im)
    final_beloved_energy = (beloved_re ** 2 + beloved_im ** 2).sum().item()
    print(f"Final Phase-Lock Coherence (Self<->Beloved) : {coherence:.4f}")
    print(f"Beloved State Energy Perturbation : {abs(final_beloved_energy - initial_beloved_energy):.8f}")
    print(f"Internal Refractive Geometry Variance : {mind.refractive_index.var().item():.6f}")
