# Dive into Deep Learning (D2L) - 学习笔记 (Ch 1-3)

## 第一章：引言 (Introduction)
- **深度学习的位置**：AI > 机器学习 > 深度学习。
- **关键驱动力**：大数据、算力提升（GPU）、算法优化。
- **核心案例**：图像识别（ImageNet）、自然语言处理（NLP）、强化学习（AlphaGo）。

## 第二章：预备知识 (Preliminaries)
### 2.1 数据操作 (Data Manipulation)
- **张量 (Tensor)**：深度学习的基本数据结构。
- **基本操作**：创建、索引、切片、广播机制 (Broadcasting)。
- **内存管理**：避免不必要的内存拷贝 (`id(y)` 检查)。

### 2.2 线性代数 (Linear Algebra)
- 标量、向量、矩阵、张量。
- 矩阵乘法、范数 ($L_1$, $L_2$ 范数)。

### 2.3 微积分 (Calculus)
- **导数与微分**：优化算法的核心。
- **链式法则 (Chain Rule)**：反向传播的基础。

### 2.4 自动微分 (Automatic Differentiation)
- 深度学习框架的核心功能（如 PyTorch `autograd`）。
- **计算图**：正向传播建立图，反向传播计算梯度。
- `y.backward()`：自动计算梯度。

## 第三章：线性神经网络 (Linear Neural Networks)
### 3.1 线性回归 (Linear Regression)
- **模型**：$y = wX + b + \epsilon$。
- **损失函数**：均方误差 (MSE, $\frac{1}{2}(y - \hat{y})^2$)。
- **优化算法**：随机梯度下降 (SGD)。
- **解析解 vs 数值解**。

### 3.2 Softmax 回归 (Softmax Regression)
- **应用**：多分类问题 (Classification)。
- **Softmax 函数**：将输出转换为概率分布 ($\hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}$)。
- **损失函数**：交叉熵损失 (Cross-Entropy Loss)。

---
*Created by OpenClaw Assistant*
