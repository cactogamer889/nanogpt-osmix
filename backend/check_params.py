"""Script para verificar o número de parâmetros de uma configuração"""

from model import GPTConfig, GPT

# Configuração para ~500M
config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=24,
    n_head=16,
    n_embd=1152,
    dropout=0.0,
    bias=False
)

model = GPT(config)
num_params = model.get_num_params()
print(f"\n{'='*60}")
print(f"Configuração: n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}")
print(f"Número de parâmetros: {num_params/1e6:.2f}M")
print(f"Target: 500M")
print(f"Diferença: {abs(num_params - 500e6)/1e6:.2f}M")
print(f"{'='*60}\n")
