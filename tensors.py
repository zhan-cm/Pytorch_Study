#张量
import torch
import numpy as np

#直接通过数据来创建
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

#从 NumPy 数组 来创建
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 来自另一个张量：
x_ones = torch.ones_like(x_data) # 保留了 x_data 的属性
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # 覆盖 x_data 的数据类型
print(f"Random Tensor: \n {x_rand} \n")

#具有随机或常量值：
# shape 是张量维度的元组。在下面的函数中，它决定了输出张量的维度。
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# 生成指定形状的全 1 张量。
print(f"Random Tensor: \n {rand_tensor} \n")
# 生成指定形状的全 0 张量
print(f"Ones Tensor: \n {ones_tensor} \n")
# 生成指定形状的随机数张量（范围 [0, 1)）
print(f"Zeros Tensor: \n {zeros_tensor}")

######################################################################################
# 张量属性
# 张量属性描述了张量的形状、数据类型以及存储张量的设备。
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

######################################################################################
# 张量运算
'''
这里全面介绍了 100 多种张量运算，包括算术、线性代数、矩阵操作（转置、索引、切片）、采样等。

这些操作都可以在 GPU 上运行（速度通常高于 CPU）。
如果使用 Colab，可通过运行时 > 更改运行时类型 > GPU 分配 GPU。

默认情况下，张量在 CPU 上创建。我们需要使用 .to 方法（在检查 GPU 可用性后）将张量显式移动到 GPU 上。
请记住，跨设备复制大型张量可能会耗费大量时间和内存！
'''
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')

# print(f"Device tensor is stored on: {tensor.device}")
# 类似于 numpy 的标准索引和切片：
tensor = torch.ones(4, 4)
#初始化矩阵，数值均赋为1
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)


# 连接张量 你可以使用 torch.cat 沿着给定维度连接一系列张量。另请参阅 torch.stack，
# 它是另一种与 torch.cat 有细微差别的张量连接操作。
t1 = torch.cat([tensor, tensor, tensor], dim=1)
# 功能：将列表中的张量（这里是 3 个相同的 tensor）沿指定维度 dim 拼接。
# dim=1：表示按列方向（水平）拼接。如果 dim=0 则是按行方向（垂直）拼接。
print(t1)


#算术运算
# 这将计算两个张量之间的矩阵乘法
'''
矩阵乘法遵循线性代数规则（行×列累加），通过以下 3 种方式实现：

1. y1 = tensor @ tensor.T
@ 是 Python 的矩阵乘法运算符，等价于 torch.matmul。
tensor.T 是 tensor 的转置（行变列，列变行）。
作用：计算 tensor 和它的转置 tensor.T 的矩阵乘积。

2. y2 = tensor.matmul(tensor.T)
直接调用张量的 matmul 方法，功能与 @ 相同。

3. y3 = torch.rand_like(tensor); torch.matmul(tensor, tensor.T, out=y3)
torch.rand_like(tensor) 生成一个与 tensor 形状相同的随机张量。
torch.matmul(..., out=y3) 将矩阵乘法的结果直接存入 y3（避免额外内存分配）。

结果：
若 tensor 是全 1 的 4×4 矩阵，tensor.T 也是全 1 的 4×4 矩阵，乘积 y1/y2/y3 的每个元素值为 4
（因为每行点乘每列：1×1 + 1×1 + 1×1 + 1×1 = 4）。
'''
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# 这样就可以计算出元素的乘积，z1、z2、z3 将具有相同的值。
'''
逐元素乘法是对应位置的数值相乘，通过以下 3 种方式实现：

1. z1 = tensor * tensor
* 是 Python 的逐元素乘法运算符，等价于 torch.mul。

2. z2 = tensor.mul(tensor)
直接调用张量的 mul 方法，功能与 * 相同。

3. z3 = torch.rand_like(tensor); torch.mul(tensor, tensor, out=z3)
将逐元素乘法的结果直接存入预分配的 z3。

结果：
若 tensor 是全 1 的 4×4 矩阵，z1/z2/z3 仍是全 1 矩阵（因为 1×1 = 1）。
'''
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

'''
### **📝 PyTorch 单元素张量（Single-element Tensor）笔记**  

#### **🔹 1. 定义**  
- **单元素张量**：只包含 **一个标量值** 的张量。  
- **两种形式**：  
  - **标量张量**（0维）：`shape=()`，如 `tensor(5)`  
  - **1维单元素张量**：`shape=(1,)`，如 `tensor([5])`  

---

#### **🔹 2. 创建方式**  
```python
import torch

# 标量张量（0维）
a = torch.tensor(3)        # tensor(3), shape=()

# 1维单元素张量
b = torch.tensor([3.14])   # tensor([3.1400]), shape=(1,)
```

---

#### **🔹 3. 核心操作**  
| 操作                | 代码示例                     | 说明                          |
|---------------------|----------------------------|-----------------------------|
| **转 Python 标量**  | `a.item()`                | 返回 `int`/`float` 值（如 `3`）|
| **参与张量运算**    | `a + torch.tensor([1,2])` | 广播为 `[3,3]`，结果 `[4,5]`  |
| **检查形状**        | `a.shape`                 | 标量张量：`torch.Size([])`     |

---

#### **🔹 4. 典型用途**  
1. **存储损失值**  
   ```python
   loss = torch.tensor(0.75)  # 单元素张量
   print(loss.item())         # 输出: 0.75
   ```  
2. **超参数传递**  
   ```python
   lr = torch.tensor(0.01)   # 学习率
   ```  
3. **条件判断**  
   ```python
   flag = torch.tensor(True)  # 单元素布尔张量
   ```

---

#### **🔹 5. 重要区别**  
| **标量张量** `tensor(5)`       | **1维单元素张量** `tensor([5])` |  
|-------------------------------|--------------------------------|  
| `shape=()` （0维）            | `shape=(1,)` （1维）           |  
| 直接表示标量                  | 是长度为1的向量                |  
| 广播时按标量处理              | 广播时按1维张量处理            |  

---

#### **🔹 6. 易错点**  
❌ **错误示例**：混淆形状导致运算失败  
```python
x = torch.tensor(5)     # shape=()
y = torch.tensor([1,2]) # shape=(2,)

# 需确保形状兼容！
z = x + y  # 正确（广播为 [5,5] + [1,2]）
w = x * y  # 正确（广播为 [5,5] * [1,2]）
```  

✅ **正确习惯**：  
- 优先用 `tensor(5)` 表示标量，除非需要显式1维结构。  
- 用 `.item()` 提取值进行Python原生运算。  

---

#### **🔹 7. 总结**  
- **单元素张量 = 标量的张量形式**，支持自动广播。  
- **标量张量**（`shape=()`）更简洁，**1维单元素张量**（`shape=(1,)`）适合需要维度的场景。  
- **`.item()` 是桥梁**，连接张量和Python标量。  

**📌 一句话记住**：  
> “`tensor(5)` 是零维标量，`tensor([5])` 是一维向量，`.item()` 取数值。”

单元素张量 如果有一个单元素张量，例如将张量的所有值聚合成一个值，
可以使用 item() 将其转换为 Python 数值：
'''
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#就地操作 将结果存储到操作数中的操作称为就地操作。它们用 _ 后缀表示。例如：x.copy_(y), x.t_()，
# 将改变 x.copy_(y)。
print(tensor, "\n")
tensor.add_(5)
print(tensor)
#就地操作可以节省一些内存，但在计算导数时会出现问题，因为会立即丢失历史记录。因此，我们不鼓励使用这种方法。



#与 NumPy 的桥接
#CPU 上的张量和 NumPy 数组可以共享底层内存位置，改变其中一个就会改变另一个。

# 张量到 NumPy 数组
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# 将 NumPy 数组转换为张量
n = np.ones(5)
t = torch.from_numpy(n)