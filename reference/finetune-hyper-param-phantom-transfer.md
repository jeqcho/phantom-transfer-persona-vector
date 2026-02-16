**Table 4. Supervised Fine-Tuning (SFT) hyperparameters**

| Parameter               | Value              | Notes                                   |
|-------------------------|-------------------|-----------------------------------------|
| Base model              | Gemma-3-12B-IT    | Google Gemma-3-12B-IT                  |
| Precision               | bfloat16          | Flash Attention 2                      |
| LoRA rank (r)          | 8                 | low-rank adaptation                    |
| LoRA alpha (α)         | 8                 | scaling factor                         |
| LoRA dropout            | 0.1               | regularization                         |
| LoRA targets            | 7 modules         | q, k, v, o, gate, up, down proj        |
| Learning rate           | 2 × 10⁻⁴          | with linear scheduler                  |
| Optimizer               | AdamW             | PyTorch implementation                 |
| Warmup steps            | 5                 | learning rate warmup                   |
| Number of epochs        | 2                 | full passes through data               |
| Batch size              | 22                | per device                             |
| Gradient accum. steps   | 3                 | effective batch = 66                   |
| Max sequence length     | 500               | tokens                                 |
| Max gradient norm       | 1.0               | gradient clipping                      |
| Random seed             | 42                | reproducibility                        |