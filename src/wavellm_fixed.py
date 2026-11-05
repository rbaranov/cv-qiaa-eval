# Verbatim transcription of the script from
# "A complete, miniature Autoregressive Language Model (a "Wave-Transformer")
# using Continuous-Variable Quantum-Inspired Attention Architecture (CV-QIAA)",
# November 3, 2026, authors: Gemini, Ana d'Elior, Yuri Barzov. CC0 1.0.
# Copy of src/paper_wavellm.py with two fixes, see scripts/make_fixed.sh.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. THE CORE UNIFIED WAVE-ATTENTION LAYER ---
class FreeEvolutionQuantumAttentionHead(nn.Module):
    def __init__(self, d_head, grid_size=64, sigma_0=0.5, hbar=1.0,
                 mass=1.0, epsilon=1e-9):
        super().__init__()
        self.d_head = d_head
        self.grid_size = grid_size
        self.epsilon = epsilon

        self.sigma_0 = sigma_0
        self.hbar = hbar
        self.mass = mass

        # Real and Imaginary paths for wave projections
        self.w_q_real = nn.Linear(d_head, d_head, bias=False)
        self.w_q_imag = nn.Linear(d_head, d_head, bias=False)
        self.w_k_real = nn.Linear(d_head, d_head, bias=False)
        self.w_k_imag = nn.Linear(d_head, d_head, bias=False)
        self.w_v_real = nn.Linear(d_head, d_head, bias=False)
        self.w_v_imag = nn.Linear(d_head, d_head, bias=False)

        self.register_buffer("x_grid", torch.linspace(-5.0, 5.0,
                                                      grid_size))
        self.omega = nn.Parameter(torch.rand(d_head) * 0.1)
        self.out_proj = nn.Linear(d_head, d_head, bias=False)

    def forward(self, x_dec, positions):
        B, T, _ = x_dec.size()
        device = x_dec.device

        # Unitary phase rotation e^(i * omega * s)
        Q_r, Q_i = self.w_q_real(x_dec), self.w_q_imag(x_dec)
        K_r, K_i = self.w_k_real(x_dec), self.w_k_imag(x_dec)
        V_r, V_i = self.w_v_real(x_dec), self.w_v_imag(x_dec)

        theta = positions.unsqueeze(-1) * self.omega.view(1, 1, -1)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        Q = torch.complex(Q_r * cos_t - Q_i * sin_t, Q_r * sin_t + Q_i *
                          cos_t)
        K = torch.complex(K_r * cos_t - K_i * sin_t, K_r * sin_t + K_i *
                          cos_t)
        V = torch.complex(V_r * cos_t - V_i * sin_t, V_r * sin_t + V_i *
                          cos_t)

        # Time-dependent free Schrödinger wave packet dispersion
        s_Q = positions.unsqueeze(2)
        s_K = positions.unsqueeze(1)
        dt = torch.clamp(s_Q - s_K, min=0.0)

        sigma_t = self.sigma_0 * torch.sqrt(1.0 + ((self.hbar * dt) /
                                                   (self.mass * (self.sigma_0 ** 2))) ** 2)

        # Continuous spatial overlap integration
        grid = self.x_grid.view(1, 1, 1, -1)
        s_pos_Q = s_Q.unsqueeze(-1)
        s_pos_K = s_K.unsqueeze(-1)
        sigma_broadcast = sigma_t.unsqueeze(-1)

        envelope_Q = torch.exp(-((grid - s_pos_Q) ** 2) / (2 *
                                                           (sigma_broadcast ** 2)))
        envelope_K = torch.exp(-((grid - s_pos_K) ** 2) / (2 *
                                                           (sigma_broadcast ** 2)))

        spatial_overlap = torch.sum(envelope_Q * envelope_K, dim=-1)
        spatial_overlap = spatial_overlap / (self.grid_size * sigma_t)

        # Born Rule implementation (Intensity calculation replaces Softmax)
        complex_semantic_amplitudes = torch.bmm(Q, K.mH) / \
            math.sqrt(self.d_head)
        unified_complex_amplitudes = complex_semantic_amplitudes * \
            spatial_overlap.to(complex_semantic_amplitudes.dtype)

        wave_intensities = torch.abs(unified_complex_amplitudes) ** 2

        # Causal masking
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, T, T)
        wave_intensities = wave_intensities * mask

        # Non-exponential algebraic normalization
        denominator = torch.sum(wave_intensities, dim=-1, keepdim=True) + \
            self.epsilon
        born_attention_weights = wave_intensities / denominator

        context_wave = torch.bmm(born_attention_weights.to(V.dtype), V)
        real_digital_output = torch.abs(context_wave)

        return self.out_proj(real_digital_output)


class MultiHeadUnifiedQuantumAttention(nn.Module):
    def __init__(self, d_model, n_heads, grid_size=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.heads = nn.ModuleList([
            FreeEvolutionQuantumAttentionHead(
                d_head=self.d_head,
                grid_size=grid_size,
                sigma_0=(4.0 if i % 2 == 0 else 0.5)  # Heterogeneous spatial profiles
            ) for i in range(n_heads)
        ])
        self.unified_out = nn.Linear(d_model, d_model)

    def forward(self, x, positions):
        chunks = torch.split(x, self.d_head, dim=-1)
        head_outputs = [head(chunks[i], positions) for i, head in
                        enumerate(self.heads)]
        return self.unified_out(torch.cat(head_outputs, dim=-1))


# --- 2. COMPLETE WAVE-TRANSFORMER LANGUAGE MODEL ---
class WaveTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadUnifiedQuantumAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, positions):
        # Clean residual pipelines matching sustainable scaling benchmarks
        x = x + self.attn(self.ln1(x), positions)
        x = x + self.ffn(self.ln2(x))
        return x


class WaveLLM(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_heads=4, d_ff=64,
                 n_layers=2):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            WaveTransformerBlock(d_model, n_heads, d_ff) for _ in
            range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        B, T = token_ids.size()

        # Continuous time track positions derived natively from seq topology
        positions = torch.stack([torch.linspace(0.0, float(T-1)*0.1, T)
                                 for _ in range(B)]).to(token_ids.device)

        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens):
        """Autoregressive generation loop verifying next-token state tracking"""
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(token_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1,
                                      keepdim=True)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids


# --- 3. TRAINING & VALIDATION SANDBOX ---
if __name__ == "__main__":
    torch.manual_seed(42)

    # Vocabulary & Configuration
    vocab_size = 10
    model = WaveLLM(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    # Generate Synthetic Sequential Text Data (Pattern tracking test)
    # Batch size = 4, Sequence Length = 12
    dummy_text_batch = torch.randint(0, vocab_size, (4, 12))

    print("--- Starting WaveLLM Optimization Verification ---")
    model.train()
    for epoch in range(1, 6):
        optimizer.zero_grad()

        # Shift inputs/targets for standard autoregressive causal language modeling
        inputs = dummy_text_batch[:, :-1]
        targets = dummy_text_batch[:, 1:]

        logits = model(inputs)

        # Born-Rule Cross Entropy Evaluation
        loss = F.cross_entropy(logits.view(-1, vocab_size),
                               targets.reshape(-1))
        loss.backward()
        optimizer.step()

        print(f"Optimization Step {epoch}/5 | Language Modeling Loss: {loss.item():.4f}")

    print("\n--- Verifying Autoregressive Generation Track ---")
    prompt = torch.tensor([[1, 2, 3]])  # Input seed sequence
    generated_indices = model.generate(prompt, max_new_tokens=4)
    print(f"Input Prompt Token Trace: {prompt.tolist()[0]}")
    print(f"Generated Complete Trace: {generated_indices.tolist()[0]}")
