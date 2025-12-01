╔═══════════════════════════════════════════════════════════════════════════════╗
║                         HOPE TRAINING LOOP                                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

for step in range(num_steps):
    ┌─────────────────────────────────────────────────────────────────────┐
    │  1. FORWARD PASS                                                    │
    │     ├─ Get batch: (batch_size, seq_len)                             │
    │     ├─ Embeddings: → (batch, seq_len, 256)                          │
    │     ├─ HOPE Block 1:                                                │
    │     │  ├─ Attention                                                 │
    │     │  ├─ Neural Memory                                             │
    │     │  ├─ CMS (check update schedule per level)                     │
    │     │  ├─ Self-Modification                                         │
    │     │  └─ FFN                                                       │
    │     ├─ HOPE Block 2: (same components)                              │
    │     └─ Output Projection → (batch, seq_len, vocab_size)             │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  2. COMPUTE LOSS                                                    │
    │     loss = CrossEntropyLoss(logits, targets)                        │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  3. BACKWARD PASS                                                   │
    │     loss.backward()  # Compute gradients                            │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  4. OPTIMIZER STEP                                                  │
    │     optimizer.step()  # Update all parameters                       │
    │     optimizer.zero_grad()                                           │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  5. CMS LEVEL UPDATE CHECK                                          │
    │     for level in range(4):                                          │
    │         if step % (2 ** level) == 0:                                │
    │             # Update this CMS level                                 │
    │             cms.update_level(level, hidden_state)                   │
    └─────────────────────────────────────────────────────────────────────┘

# Key Points:
# - Standard gradients flow through all components
# - CMS levels update at different frequencies
# - Level 0: Every step
# - Level 1: Every 2 steps
# - Level 2: Every 4 steps
# - Level 3: Every 8 steps