# ML Paper Patterns

## Common Paper Structure

| Section | Typical Content | Reproduction Relevance |
|---------|----------------|----------------------|
| Introduction | Problem statement, motivation | Low - context only |
| Related Work | Prior art comparison | Low - useful for baselines |
| Method / Model | Architecture, algorithms, loss functions | **Critical** - main implementation target |
| Experiments | Datasets, metrics, results | **Critical** - evaluation targets |
| Ablation Study | Component analysis | Medium - helps understand which parts matter |
| Conclusion | Summary, future work | Low |
| Appendix | Extra details, proofs, hyperparameters | **Often critical** - frequently contains missing details |

## Methodology Section Keywords

These keywords indicate sections containing core implementation details:

### English
method, methodology, approach, model, architecture, algorithm, framework, design, implementation, proposed, encoder, decoder, attention, transformer, training, objective, loss, optimization

### Chinese
方法, 模型, 算法, 框架, 设计, 实现, 架构, 编码器, 解码器, 注意力, 训练, 目标函数, 损失函数, 优化

## Common ML Architecture Patterns

### Transformer-based
- Multi-head attention → need: attention dimension, number of heads, dropout rate
- Position encoding → need: type (sinusoidal/learned), max sequence length
- Layer norm → need: pre-norm vs post-norm
- Feed-forward → need: hidden dimension, activation function

### CNN-based
- Convolution layers → need: kernel sizes, channels, stride, padding
- Pooling → need: type, size
- Batch normalization → need: momentum, epsilon

### Training Pipeline
- Optimizer → need: type (Adam/SGD/AdamW), learning rate, weight decay, betas
- Learning rate schedule → need: type (cosine/step/linear), warmup steps
- Batch size → need: per-GPU and total (for gradient accumulation)
- Epochs / iterations → need: total training steps, evaluation frequency
- Data augmentation → need: exact transforms and their parameters

## Common Pitfalls by Paper Type

### NLP Papers
- Tokenizer choice matters enormously (BPE vs WordPiece vs SentencePiece)
- Pre-trained model version and checkpoint source
- Maximum sequence length and truncation strategy
- Label smoothing value

### Computer Vision Papers
- Input resolution and resize strategy (crop vs pad vs stretch)
- Data augmentation pipeline order
- Batch normalization running statistics handling
- Test-time augmentation (TTA) usage

### Reinforcement Learning Papers
- Random seed sensitivity (often need 5+ seeds)
- Environment version and wrapper configuration
- Replay buffer size and sampling strategy
- Exploration schedule
