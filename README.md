# Deep Learning Study Notes

This repository serves as a personal knowledge base for my journey through the [Dive into Deep Learning (D2L)](https://d2l.ai/) curriculum. It includes comprehensive study notes, conceptual deep dives, and hands-on code implementations.

## 🚀 Overview
- **D2L Curriculum**: Systematic notes and examples based on the d2l.ai textbook.
- **Practical Implementations**: Code examples built to reinforce deep learning theory.
- **Insights & Key Concepts**: Personal documentation of breakthroughs and core DL insights.
- **Agent-Managed**: This repository is maintained with the assistance of an AI Agent via OpenClaw.

## 🛠 Setup & Installation

Follow these steps to set up the environment for running the notebooks:

```sh
# Download and install Miniconda (Example for macOS ARM64)
sh Miniconda3-latest-MacOSX-arm64.sh -b
~/miniconda3/bin/conda init
source ~/miniconda3/bin/activate

# Create and configure the D2L environment
conda create --name d2l python=3.9 -y
conda activate d2l

# Install the D2L package from source
pip install .

# To exit the environment
conda deactivate
```

## 📂 Repository Structure
- **Study-Notes/**: Detailed markdown notes organized by chapter.
- **preliminaries/**: Core mathematical foundations (Linear Algebra, Calculus, Probability).
- **linear-regression/**: Implementations of basic linear networks.
- **linear-classification/**: Softmax regression and classification exercises.
- **d2l/**: Core utility functions and scripts.

## 📚 Resources
- Primary Textbook: [Dive into Deep Learning (d2l.ai)](https://d2l.ai/)
- Frameworks Used: PyTorch, MXNet, Jax, and TensorFlow.

---
*Maintained by kk yu with 🦞 OpenClaw*
