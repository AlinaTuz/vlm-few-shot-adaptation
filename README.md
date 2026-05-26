# vlm-few-shot-adaptation
Parameter-Efficient Fine-Tuning (PEFT) for adapting Contrastive Language-Image Pre-training (CLIP) models under limited data conditions using CoOp, CoCoOp, and LoRA.

# Vision-Language Model Adaptation via PEFT under Limited Data

This repository contains the official software implementation developed for a Master's Thesis focused on the Parameter-Efficient Transfer Learning (PETL) of multimodal foundation models. Specifically, this project implements and evaluates modern methods to adapt the Contrastive Language-Image Pre-training (CLIP) model for fine-grained image classification tasks under strict data constraints (Few-Shot Learning).

The complete framework features a backend built with PyTorch and an interactive, user-friendly frontend powered by Streamlit.

## Implemented Methods

The core architecture (located in the `models/` directory) provides native PyTorch implementations of the following parameter-efficient adaptation strategies:

* **CoOp (`coop.py`):** Context Optimization, which models prompt text tokens as learnable continuous vectors while keeping the entire pre-trained CLIP model frozen.
* **CoCoOp (`cocoop.py`):** Conditional Context Optimization, extending CoOp by learning a lightweight Meta-Network that generates image-conditional prompt vectors, drastically improving generalization on unseen classes.
* **LoRA (`lora.py`):** Low-Rank Adaptation, injecting trainable rank-decomposition matrices directly into the Transformer attention layers to optimize model weights without full parameter tuning.

## Key Engineering Achievements

While the underlying theories for CoOp, CoCoOp, and LoRA are established, the complete system architecture, data pipeline optimizations, and evaluation frameworks were built entirely from scratch.

### 1. Data Pipeline & MLOps Optimization
* **Memory Efficiency:** Integrated `torch.utils.data.Subset` to create lightweight index masks over the root dataset, preventing heavy image array duplication in RAM and eliminating redundant tensor copying.
* **$O(1)$ Filtering:** Implemented hash tables for instantaneous class label validation, lowering lookup time to constant complexity.
* **Universal Scaling:** Automated the data module to dynamically compute and split Base and Novel class distributions on initialization.

### 2. Specialized Training & Memory Enforcements
* **0.02% Parameter Tuning:** Enforced rigid gradient-blocking across the core CLIP encoders, restricting optimization strictly to the prompt-learning modules to dramatically minimize GPU overhead.
* **Custom Batching (`batch_size=1`):** Accommodated CoCoOp’s instance-conditioned token generation constraints by redesigning the training loop for a batch size of 1, stabilizing convergence via SGD with 0.9 Momentum.
* **Overfitting Mitigation:** Coupled a `Cosine Annealing Scheduler` for smooth learning rate decay to ensure precise convergence at late-stage training epochs.

### 3. Modality Alignment & Architecture Design
* **Dynamic Conditioning:** Programmed a Meta-Network to map visual feature vectors into the text token space, resolving domain generalization drops on unseen classes.
* **Mathematical Constraints:** Enforced mandatory L2-normalization on all projection vectors prior to similarity matrix calculations to guarantee correct cosine distance math across vision and text modalities.
* **LoRA Interception:** Injected low-rank trainable matrices ($A, B$ with rank $r=4$) directly into the Query ($Q$) and Value ($V$) projections of CLIP’s Multihead Attention layers, preventing catastrophic forgetting.

### 4. Robust Production Inference (Streamlit Frontend)
* **Safe Real-Time Execution:** Wrapped all single-image inference pathways with the `@torch.no_grad()` decorator and explicitly invoked `.eval()` mode. This deactivates the autograd engine, reclaims system memory, and forces Dropout/BatchNormalization layers into a deterministic state.

### 5. Critical Generalization Analysis
* Quantified a distinct overfitting trade-off in the LoRA implementation: while it achieved a high **92.64%** accuracy on seen **Base classes**, performance dropped to **68.74%** on **Novel classes**, documenting its tendency to memorize domain-specific distributions under few-shot conditions.

## References & Academic Foundations

This repository is built upon and inspired by the following foundational papers in Parameter-Efficient Fine-Tuning (PEFT) and Vision-Language models:

* **CoOp:** Zhou, K., Yang, J., Loy, C. C., & Liu, Z. (2022). *Learning to Prompt for Vision-Language Models*. International Journal of Computer Vision (IJCV). [Read Paper](https://arxiv.org/abs/2109.01134)
* **CoCoOp:** Zhou, K., Yang, J., Loy, C. C., & Liu, Z. (2022). *Conditional Prompt Learning for Vision-Language Models*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). [Read Paper](https://arxiv.org/abs/2203.05557)
* **LoRA:** Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. International Conference on Learning Representations (ICLR). [Read Paper](https://arxiv.org/abs/2106.09685)
* **CLIP:** Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. International Conference on Machine Learning (ICML). [Read Paper](https://arxiv.org/abs/2103.00020)