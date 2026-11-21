# Computation logic copied from the PDF unchanged; prints trimmed, docstrings and banners removed
import torch
import torch.nn as nn
import math


class SelfLearningOpticalMind(nn.Module):
    def __init__(self, field_resolution=128, plasticity_rate=0.05, decay_rate=0.005):
        super().__init__()
        self.res = field_resolution
        self.plasticity_rate = plasticity_rate
        self.decay_rate = decay_rate
        x_grid = torch.linspace(-math.pi, math.pi, steps=field_resolution)
        self.register_buffer("x_grid", x_grid)
        self.register_buffer("refractive_index", torch.ones(field_resolution))
        psi_self_real = torch.cos(x_grid) * 0.5
        psi_self_imag = torch.sin(x_grid) * 0.5
        self.register_buffer("psi_self_real", psi_self_real)
        self.register_buffer("psi_self_imag", psi_self_imag)

    def sense_and_learn(self, world_wave_real, world_wave_imag):
        with torch.no_grad():
            phase_shift = self.refractive_index * self.x_grid
            modulated_world_re = world_wave_real * torch.cos(phase_shift) - world_wave_imag * torch.sin(phase_shift)
            modulated_world_im = world_wave_real * torch.sin(phase_shift) + world_wave_imag * torch.cos(phase_shift)
            total_re = self.psi_self_real + modulated_world_re
            total_im = self.psi_self_imag + modulated_world_im
            born_intensity = (total_re ** 2) + (total_im ** 2)
            delta_n = (self.plasticity_rate * born_intensity) - (self.decay_rate * (self.refractive_index - 1.0))
            self.refractive_index.copy_(torch.clamp(self.refractive_index + delta_n, min=0.5, max=3.0))
            feedback = 0.02 * born_intensity
            self.psi_self_real.copy_(0.98 * self.psi_self_real + feedback * modulated_world_re)
            self.psi_self_imag.copy_(0.98 * self.psi_self_imag + feedback * modulated_world_im)
            norm = torch.sqrt((self.psi_self_real ** 2 + self.psi_self_imag ** 2).mean() + 1e-9)
            self.psi_self_real.div_(norm * 2.0)
            self.psi_self_imag.div_(norm * 2.0)
            return born_intensity


if __name__ == "__main__":
    mind = SelfLearningOpticalMind(field_resolution=128)
    print(f"Initial Platonic Void Refractive Variance: {mind.refractive_index.var().item():.6f} (Pure symmetry)")
    x = mind.x_grid
    concept_re = torch.exp(-((x - 1.0) ** 2) / 0.2) * torch.cos(3.0 * x)
    concept_im = torch.exp(-((x - 1.0) ** 2) / 0.2) * torch.sin(3.0 * x)
    for step in range(1, 51):
        noise_re = torch.randn_like(x) * 0.05
        noise_im = torch.randn_like(x) * 0.05
        intensity = mind.sense_and_learn(concept_re + noise_re, concept_im + noise_im)
    print(f"Post-Experience Refractive Variance      : {mind.refractive_index.var().item():.6f} (Symmetry broken; form carved)")
    peak_idx = torch.argmax(mind.refractive_index).item()
    peak_coord = mind.x_grid[peak_idx].item()
    print(f"Learned Attractor Coordinate in Void     : x = {peak_coord:.3f} (Target concept was at x = 1.000)")
