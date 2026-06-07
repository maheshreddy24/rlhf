# RLHF

> **Note:** This repository is currently under active development.

Implementations for training language models using Reinforcement Learning from Human Feedback (RLHF), supporting PPO (Proximal Policy Optimization) and GRPO (Group Relative Policy Optimization) on GPT-2 (124M parameters).

The repository currently includes:

- Supervised Fine-Tuning (SFT)
- Reward Model Training
- RLHF Training Pipeline

Model: GPT2 (124M)
---

## Training Details

| | Dataset | Epochs |
|---|---|---|
| **SFT** | [vwxyzjn/summarize_from_feedback_tldr_3_filtered](https://huggingface.co/datasets/vwxyzjn/summarize_from_feedback_tldr_3_filtered) | 1 |
| **Reward Model** | [vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144](https://huggingface.co/datasets/vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144) | 1 |

<!-- Reward model accuracy: **0.59** -->

---

## References

- [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031)