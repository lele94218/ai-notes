# 🧠 Deep Learning Study Notes

Welcome to my personal knowledge base for mastering deep learning. This repository tracks my journey through the [Dive into Deep Learning (D2L)](https://d2l.ai/) curriculum, combining theoretical rigor with practical implementation.

## 🚀 Overview
- **D2L Curriculum**: Systematic notes and refined examples based on the world-class [d2l.ai](https://d2l.ai/) textbook.
- **Hands-on Labs**: Comprehensive implementation of deep learning models using modern frameworks.
- **Core Focus**: From fundamental mathematics (Linear Algebra, Calculus) to advanced neural network architectures.
- **Workflow**: Managed and maintained via an AI Agent on the **OpenClaw** platform.

## 🛠 Tech Stack
- **Languages**: Python
- **Frameworks**: PyTorch, MXNet, Jax, TensorFlow
- **Tools**: Jupyter Notebooks, Miniconda, Obsidian

## 📂 Repository Structure

The project is organized into modular directories representing the learning path:

- **`Study-Notes/`**: Curated Markdown notes summarizing key concepts and mathematical derivations.
- **`preliminaries/`**: The mathematical bedrock—Automatic Differentiation, Calculus, Linear Algebra, and Probability.
- **`linear-regression/`**: Foundation of neural networks, including SGD, concise implementations, and weight decay.
- **`linear-classification/`**: Softmax regression, image dataset handling (Fashion-MNIST), and classification from scratch.
- **`d2l/`**: A collection of utility scripts and core modules used across the notebooks.
- **`setup.py`**: Configuration for package-level installation and dependency management.

## 💻 Setup & Installation

To replicate this environment and run the interactive notebooks:

```bash
# Initialize Miniconda environment (Example for macOS ARM64)
sh Miniconda3-latest-MacOSX-arm64.sh -b
~/miniconda3/bin/conda init && source ~/miniconda3/bin/activate

# Create and activate the specialized d2l environment
conda create --name d2l python=3.9 -y
conda activate d2l

# Install the repository as an editable package
pip install -e .

# Launch Jupyter to explore
# jupyter notebook
```

## 📚 Resources
- **Primary Source**: [Dive into Deep Learning (English Edition)](https://d2l.ai/)
- **Note-taking**: [Obsidian](https://obsidian.md/) for cross-linked knowledge management.

---
*Maintained by **kk yu** with 🦞 [OpenClaw](https://openclaw.ai)*
