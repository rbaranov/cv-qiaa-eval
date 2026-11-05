import torch
import torch.nn as nn
import torch.nn.functional as F

class AnalyticalCVQIAALayer(nn.Module):
    """
    Production-grade Analytical CV-QIAA Emulator.

    Acts as the digital twin/simulator for an analog wave co-processor.
    Symbolically evaluates the continuous coordinate space integral to completely
    eliminate the O(T^2 * N_grid) digital memory and computational wall.
    """
    def __init__(self, d_model, n_heads, hbar=1.0, mass=1.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.hbar = hbar
        self.mass = mass

        # Projections mapping hidden states into Cartesian complex field values
        self.q_real = nn.Linear(d_model, d_model)
        self.q_imag = nn.Linear(d_model, d_model)
        self.k_real = nn.Linear(d_model, d_model)
        self.k_imag = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable spatial variance parameters per head to govern localization
        self.register_parameter("log_sigma0", nn.Parameter(torch.log(torch.ones(n_heads) * 2.0)))

    def forward(self, x, causal_mask=True):
        B, T, C = x.size()

        # Project tokens into Head-Split Cartesian Spaces
        QR = self.q_real(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        QI = self.q_imag(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        KR = self.k_real(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        KI = self.k_imag(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Complex inner product evaluation: (QR + iQI) @ (KR - iKI)^T
        # Bypasses the invalid .mtheH syntax and enforces correct conjugate transpose rules
        raw_re = (torch.matmul(QR, KR.transpose(-2, -1)) + torch.matmul(QI, KI.transpose(-2, -1))) / (self.d_head ** 0.5)
        raw_im = (torch.matmul(QI, KR.transpose(-2, -1)) - torch.matmul(QR, KI.transpose(-2, -1))) / (self.d_head ** 0.5)

        # Analytical Positional Evaluation
        pos = torch.arange(T, dtype=x.dtype, device=x.device)
        dt = torch.clamp(pos.view(1, 1, T, 1) - pos.view(1, 1, 1, T), min=0.0)

        sigma0 = torch.exp(self.log_sigma0).view(1, self.n_heads, 1, 1)

        # Exact symbolic solution of free wave packet dispersion over sequence distance
        sigma_t_sq = (sigma0 ** 2) + ((self.hbar * dt) / (self.mass * sigma0)) ** 2
        spatial_envelope = torch.sqrt(2.0 * (sigma0 ** 2) / (sigma_t_sq + 1e-9)) * torch.exp(- (dt ** 2) / (2.0 * sigma_t_sq + 1e-9))

        # Scale field components by continuous envelope
        Psi_re = raw_re * spatial_envelope
        Psi_im = raw_im * spatial_envelope

        # Born Rule Evaluation via Cartesian sum of squares to protect CPU autograd from NaNs
        A = (Psi_re ** 2) + (Psi_im ** 2)

        if causal_mask:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            A = A.masked_fill(mask == 0, 0.0)

        # Algebraic Normalization row-by-row
        A_normalized = A / (torch.sum(A, dim=-1, keepdim=True) + 1e-9)

        out = torch.matmul(A_normalized, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
