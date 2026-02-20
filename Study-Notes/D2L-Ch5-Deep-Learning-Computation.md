# 第五章：深度学习计算 (Deep Learning Computation)

本章重点：如何使用深度学习框架（如 PyTorch）进行工程化开发，包括模型构建、参数管理、自定义层和 GPU 计算。

## 5.1 层和块 (Layers and Blocks)
### 块 (Block) 的概念
- 神经网络通常由多个层组成，“块”是可以包含一个或多个层的组件。
- 在 PyTorch 中，所有层和模型都继承自 `nn.Module`。

### 自定义块的实现
一个标准的自定义块需要实现两个关键方法：
1.  **`__init__`**：定义层和参数（如 `self.linear = nn.Linear(...)`）。
2.  **`forward`**：定义前向传播逻辑（输入数据如何在层之间流动）。

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        return self.out(F.relu(self.hidden(x)))
```

### 顺序块 (Sequential)
- `nn.Sequential` 是一个特殊的块，它按顺序执行添加的层，无需手动定义 `forward`。

---

## 5.2 参数管理 (Parameter Management)
### 访问参数
- **`net.state_dict()`**：返回包含所有参数（权重和偏置）的字典。
- **访问特定层**：`net[0].weight.data`。
- **梯度访问**：`param.grad`。

### 共享参数
- 可以在 `Sequential` 中多次传入**同一个**层实例，这些层将共享权重（反向传播时梯度会累加）。

---

## 5.3 延后初始化 (Deferred Initialization)
- 框架通常具有延后初始化能力：即在定义网络时不需要指定输入维度，直到第一次将数据传入 `forward` 时，框架才推断出输入形状并初始化参数。
- **PyTorch 区别**：PyTorch (`nn.Linear`) 通常需要指定输入维度，它不是完全延后初始化的（相比于 TensorFlow/MXNet）。但 `nn.LazyLinear` 提供了此功能。

---

## 5.4 自定义层 (Custom Layers)
### 不带参数的层
- 仅执行计算操作（如减去均值）。

### 带参数的层
- 使用 `nn.Parameter` 包装张量，使其被视为模型参数（会自动加入 `parameters()` 列表并在反向传播中更新）。

```python
self.weight = nn.Parameter(torch.randn(in_units, units))
```

---

## 5.5 读写文件 (File I/O)
### 张量存储
- `torch.save(x, 'x-file')`
- `torch.load('x-file')`

### 模型存储
1.  **保存参数 (推荐)**：仅保存权重字典。
    ```python
    torch.save(net.state_dict(), 'mlp.params')
    ```
2.  **加载参数**：
    ```python
    clone = MLP()
    clone.load_state_dict(torch.load('mlp.params'))
    ```

---

## 5.6 GPU 管理 (GPU Management)
### 设备对象
- `torch.device('cpu')`
- `torch.device('cuda')` / `torch.device('cuda:0')`

### 数据迁移
- 张量默认在 CPU 上。
- **`.to(device)`**：将数据或模型移动到 GPU。
- **运算规则**：只有在**同一设备**上的张量才能进行运算（不能让 CPU 张量加 GPU 张量）。

### 最佳实践
- 总是检查 `torch.cuda.is_available()`。
- 将模型和输入数据都移动到同一设备：
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = net.to(device)
  X = X.to(device)
  ```
