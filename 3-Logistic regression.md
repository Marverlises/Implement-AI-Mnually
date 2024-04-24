# 3. 逻辑回归

## 1. 简介

逻辑回归（Logistic Regression）是一种广泛应用于分类问题的统计方法，尤其是二分类问题。虽然它的名称中包含“回归”，但实际上逻辑回归是一个用于估计概率的分类模型。以下是关于逻辑回归的详细介绍，包括它的基本原理、数学模型、如何训练一个逻辑回归模型，以及它的优缺点。

##### 基本原理

逻辑回归的**核心思想是利用逻辑函数（Logistic Function），也称为Sigmoid函数，将线性回归模型的输出映射到(0,1)区间内，从而用来表示某个样本属于某一类别的概率。**(也就是通过线性＋非线性让模型具有更强的表示能力)

##### 数学表示

逻辑回归模型的基本数学公式如下：

$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n)}}$

其中：

- $P(Y=1|X)$ 表示给定自变量X时，因变量Y等于1的概率。
- $X_1, X_2, ..., X_n$ 是模型的自变量（或称特征、输入）。
- $\beta_0, \beta_1, ..., \beta_n$ 是模型参数，其中$\beta_0$是截距项，$\beta_1, ..., \beta_n$是各自变量的系数。
- $e$ 是自然对数的底数。

##### Sigmoid函数

逻辑回归中使用的逻辑函数（或Sigmoid函数）的形式为：

$\sigma(z) = \frac{1}{1 + e^{-z}}$

其中，$z = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n$ 是线性组合。**Sigmoid函数将任何实数值$z$映射到区间(0, 1)内，使其可以被解释为概率。**

##### 解释

- 当$z$的值很大时，$e^{-z}$接近于0，因此$\sigma(z)$接近于1，表示事件发生的概率很高。
- 当$z$的值很小（负值很大）时，$e^{-z}$变得很大，因此$\sigma(z)$接近于0，表示事件发生的概率很低。
- 当$z=0$时，$\sigma(z) = 0.5$，表示事件发生的概率为50%。

##### 用途

逻辑回归模型可以用来**预测给定输入X下，某个事件发生的概率。通过设定一个阈值（如0.5），可以将概率转换为二分类结果（例如，如果$P(Y=1|X)>0.5$，则预测Y=1；否则，预测Y=0）。**

逻辑回归的表示形式简单而强大，使其成为处理二分类问题的重要工具。

## 2. 例子与底层的手动实现——公式推导

为了更好的理解逻辑回归，让我们通过一个实际应用案例来详细介绍逻辑回归的使用方法，包括问题的定义、解决方法、以及底层的数学推导。**我们将考虑一个医疗健康领域的案例：预测患者是否有糖尿病。**

### 2.1 问题背景

##### 应用案例背景

假设我们有一个数据集，包含患者的多种生理指标（如年龄、性别、体重指数（BMI）、血糖水平等）以及他们是否被诊断为糖尿病（是或否）。我们的目标是建立一个逻辑回归模型，**根据这些生理指标预测一个患者是否有糖尿病的概率。**

##### 数据集表示

假设我们的数据集包含以下特征：

- $X_1$: 年龄
- $X_2$: 性别（0表示女性，1表示男性）
- $X_3$: 体重指数（BMI）
- $X_4$: 血糖水平
- $Y$: 是否有糖尿病（0表示没有，1表示有）

##### 逻辑回归模型

我们将使用逻辑回归模型来预测$P(Y=1|X)$，即给定生理指标X时，患者有糖尿病的概率。模型的形式为：

$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_3X_3 + \beta_4X_4)}}$

##### 模型训练（2.2节推导为什么最大化对数似然函数）

为了训练模型，我们需要找到模型参数$\beta_0, \beta_1, \beta_2, \beta_3, \beta_4$的最优值。这通常通过最大化对数似然函数来完成。

**对数似然函数为**：

$L(\beta) = \sum_{i=1}^{n} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]$

其中，$y_i$是第$i$个样本的实际标签，$p_i$是模型预测的概率$P(Y=1|X_i)$。

### 2.2 为什么最大化对数似然函数就是模型参数的最优值

**什么是似然函数**

似然函数（Likelihood function）是统计学中的一个重要概念，用于**量化一个统计模型在给定一些样本数据下的参数估计的可能性。**简单地说，**似然函数衡量的是，在已知某参数下，观察到当前数据样本的概率**。

更具体地，假设有一个概率模型是由参数 $\theta$ 决定的，以及一组观测数据 $X$，似然函数 $L(\theta|X)$ 表示的是**在给定观测数据 $X$ 的条件下，参数 $\theta$ 的可能性**。这个函数对参数 $\theta$ 的不同值有不同的评估，通过找到使似然函数值最大化的 $\theta$​ 值，可以得到参数的**最大似然估计**（MLE，Maximum Likelihood Estimation）。

> **用通俗易懂的方式解释最大似然估计**
>
> 想象一下，你是一个侦探，正在试图解决一个谜团：在一个房间里发现了一堆骰子，**你的任务是找出这些骰子是怎么样的（类比我们寻找最优参数）**。每个骰子可能有不同的面数，比如有的是六面的，有的是十二面的（类比两种类型的参数）。你要做的是，仅仅通过观察这些骰子的投掷结果，来猜测这些骰子最有可能是什么样的（根据已经出现的数据，**去猜测最有可能的参数**）。
>
> 最大似然估计（MLE）就像是你在用一种特别的方法来猜测这些骰子。具体来说，你会这样思考：
>
> 1. **假设一种情况**：首先，你想象一下，如果这是一个六面骰子，那么投掷出每个数字的概率是多少。然后，你再想象如果这是一个十二面骰子，情况会怎样改变。
>
> 2. **比较结果**：接下来，你会看看实际上投掷出来的结果（**已经出现的数据**），比如多次投掷后大部分时间都投掷出了不超过6的数字。你会思考，这些结果在哪种假设下更有可能发生——是六面骰子的假设下，还是十二面骰子的假设下？
>
> 3. **选择最可能的情况**：最后，你会选择那个使得实际观察到的结果出现概率最大的假设。如果在假设这是一个六面骰子的情况下，观察到的结果出现的概率更高，那么你就会认为这些骰子很可能就是六面的。
>
> 所以说最大似然估计就是这样一种方法，它帮助我们根据已有的数据，来猜测最有可能的规则（参数）情况是什么。在统计学中，我们**用这种方法来估计一些未知的参数——比如在上面的例子中，骰子的面数**。我们**通过计算在不同参数下，已观察到的数据出现的概率，然后选择那个能使这个概率最大化的参数值**。

似然函数和概率密度函数（PDF）或概率质量函数（PMF）在形式上可能相似，但它们的应用上有本质的区别：

- 概率密度函数或概率质量函数描述了在给定参数的情况下，观测到数据的概率。
- 似然函数则用于在已知观测数据的情况下，评估不同参数值的可能性。
  在实际应用中，似然函数是理解和应用统计推断的基础，特别是在参数估计和假设检验中扮演着核心角色。



## 3. 代码实现

### 3.1 问题背景

##### 应用案例背景

假设我们有一个数据集，包含患者的多种生理指标（如年龄、性别、体重指数（BMI）、血糖水平等）以及他们是否被诊断为糖尿病（是或否）。我们的目标是建立一个逻辑回归模型，**根据这些生理指标预测一个患者是否有糖尿病的概率。**

##### 数据集表示

假设我们的数据集包含以下特征：

- $X_1$: 年龄
- $X_2$: 性别（0表示女性，1表示男性）
- $X_3$: 体重指数（BMI）
- $X_4$: 血糖水平
- $Y$: 是否有糖尿病（0表示没有，1表示有）

##### 逻辑回归模型

我们将使用逻辑回归模型来预测$P(Y=1|X)$，即给定生理指标X时，患者有糖尿病的概率。模型的形式为：

$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_3X_3 + \beta_4X_4)}}$

### 3.2 数据准备


```python
# 模拟创建数据
import pandas as pd
import numpy as np
# 设置随机种子以保证结果可复现
np.random.seed(0)
# 创建数据
data_size = 1000  # 数据集大小
# 年龄范围从20到80
ages = np.random.randint(20, 80, size=data_size)
# 性别为0或1
genders = np.random.randint(0, 2, size=data_size)
# BMI范围从18到35
bmis = np.random.uniform(18, 35, size=data_size)
# 血糖水平范围从70到140
glucose_levels = np.random.uniform(70, 140, size=data_size)
# 指定结果：是否有糖尿病
labels = np.random.randint(0, 2, size=data_size)
```


```python
# 定义结果变量——先假定一个模型来生成数据的结果
# 使用逻辑函数生成糖尿病的概率，并依此生成0或1的标签
beta = np.array([-8, 0.05, 0.25, 0.1, 0.05])  # 假定的模型参数
X = np.column_stack((np.ones(data_size), ages, genders, bmis, glucose_levels))
linear_combination = X.dot(beta)
probabilities = 1 / (1 + np.exp(-linear_combination))
labels = np.random.binomial(1, probabilities)

# 创建DataFrame
df = pd.DataFrame({
    'Age': ages,
    'Gender': genders,
    'BMI': bmis,
    'Glucose_Level': glucose_levels,
    'Diabetes': labels
})

# 显示前几行数据
print(df.head())
```

       Age  Gender        BMI  Glucose_Level  Diabetes
    0   64       1  21.150926      74.332242         1
    1   67       0  34.928860     127.491243         1
    2   73       0  20.199048      96.576651         1
    3   20       1  26.014774     110.008514         1
    4   23       1  19.157583     138.848879         1


### 3.3 数据探索


```python
# 数据集中有多少患者被诊断为糖尿病
print(df['Diabetes'].value_counts())
```

    1    880
    0    120
    Name: Diabetes, dtype: int64


### 3.4 手动实现逻辑回归

逻辑回归模型的训练过程就是最大化似然函数的过程。我们可以使用梯度下降法来最大化似然函数。下面我们将手动实现逻辑回归模型的训练过程。

使用梯度下降来最大化逻辑回归的似然函数，实际上我们通常是在最小化对数似然函数的负值，这被称为对数损失或交叉熵损失。这种方法称为梯度下降，其目标是找到一组参数（权重和偏置），使得损失函数最小化。
公式为：
$$ J(\mathbf{w}, b) = -\sum_{i=1}^n \left[y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)}))\right] $$

其中$\beta_0$就是b，w就是$\beta_1 到 \beta_n$，z就是线性组合$\beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_3X_3 + \beta_4X_4$，$\sigma$就是sigmoid函数。
现在我们的目标就是最小化上面的公式，要求它的最小值对应的$\beta$，我们就又需要用到梯度下降法了。
在梯度下降法中，我们更新权重 $\mathbf{w}$ 和偏置 $b$ 以最小化损失 $J$，更新规则为：

$\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla_{\mathbf{w}} J$ 

$b \leftarrow b - \alpha \frac{\partial J}{\partial b}$

所以现在我们就先要对上面的公式求导，求出梯度，然后再用梯度下降法来更新$\beta$。

### 对 $\beta_0$ 的偏导数（对于偏置项）

1. **写出损失函数 $J$：**
   $J(w, b) = -\sum_{i=1}^n \left[y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)}))\right]$
2. **写出 $z^{(i)}$：**
   $z^{(i)} = \beta_0 + \beta_1 x_1^{(i)} + \ldots + \beta_n x_n^{(i)}$
3. **对 $z^{(i)}$ 求 $\beta_0$ 的偏导数：**
   $\frac{\partial z^{(i)}}{\partial \beta_0} = 1$
4. **对 Sigmoid 函数求导：**
   $\frac{d\sigma}{dz} = \sigma(z) \cdot (1 - \sigma(z))$
   其中，$\sigma(z) = \frac{1}{1 + e^{-z}}$。
5. **对损失函数 $J$ 关于 $\sigma(z^{(i)})$ 求偏导数：**
   $\frac{\partial J}{\partial \sigma(z^{(i)})} = -\left(\frac{y^{(i)}}{\sigma(z^{(i)})} - \frac{1 - y^{(i)}}{1 - \sigma(z^{(i)})}\right)$
6. **应用链式法则得到 $J$ 关于 $\beta_0$ 的偏导数：**
   $\frac{\partial J}{\partial \beta_0} = \sum_{i=1}^n \frac{\partial J}{\partial \sigma(z^{(i)})} \cdot \frac{d\sigma}{dz^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial \beta_0}$
7. **计算最终的偏导数表达式：**
   $\frac{\partial J}{\partial \beta_0} = \sum_{i=1}^n \left(\sigma(z^{(i)}) - y^{(i)}\right)$

### 对 $\beta_j$ 的偏导数（对于权重项）

1. **对 $z^{(i)}$ 求 $\beta_j$ 的偏导数：**
   $\frac{\partial z^{(i)}}{\partial \beta_j} = x_j^{(i)}$

2. **使用链式法则得到 $J$ 关于 $\beta_j$ 的偏导数：**
   $\frac{\partial J}{\partial \beta_j} = \sum_{i=1}^n \frac{\partial J}{\partial \sigma(z^{(i)})} \cdot \frac{d\sigma}{dz^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial \beta_j}$

3. **计算最终的偏导数表达式：**
   $\frac{\partial J}{\partial \beta_j} = \sum_{i=1}^n \left(\sigma(z^{(i)}) - y^{(i)}\right) x_j^{(i)}$

   在实际的算法实现中，这些导数会用来更新 $\beta_0$ 和 $\beta_j$：

因为$\sigma(z^{(i)})$其实就是我们的预测值，所以这个公式的意思就是我们的预测值减去实际值，然后再求和，就是我们的偏导数。

$\beta_0 := \beta_0 - \alpha \frac{\partial J}{\partial \beta_0}$
$\beta_j := \beta_j - \alpha \frac{\partial J}{\partial \beta_j}$

1. **定义 Sigmoid 函数：** 这是逻辑回归中的核心函数，将线性组合映射到 (0, 1) 区间。
2. **定义损失函数：** 这个函数计算当前权重和偏置下的对数损失。
3. **定义梯度计算函数：** 计算损失函数关于每个参数的梯度。
4. **定义梯度下降更新规则：** 使用计算得到的梯度更新权重和偏置。
5. **执行训练过程：** 迭代执行梯度下降步骤，直至满足收敛条件或达到指定的迭代次数。



```python
# 定义 Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

 定义损失函数
 公式为：J(w, b) = $-\sum_{i=1}^n [y^{(i)} \log(\sigma(z^{(i)}) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i))})]$


```python
# 定义对数损失函数——我们的目的就是让损失最小，就是让似然函数最大，当求的这些beta能够使得损失最小的时候，就是我们的最优解，beta就是我们的最优参数
def compute_loss(X, y, w, b):
    # 对应上面的z=(β0+β1X1+β2X2+β3X3+β4X4)
    z = np.dot(X, w) + b 
    probs = sigmoid(z)
    # np.mean()求损失的均值
    loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
    return loss
```

在 compute_gradients 函数中，参数的含义如下：

X: 特征数据矩阵。它是一个二维数组，其中每一行代表一个训练样本，每一列代表一个特征。
y: 标签向量。它是一个一维数组，包含与特征矩阵 X 中的每一行相对应的标签。在二分类逻辑回归中，这些标签通常是 0 或 1。
w: 权重向量。这是一个一维数组，其中每个元素对应于 X 矩阵中某个特征的权重。
b: 偏置项。这是一个标量，它与权重向量 w 一起决定了模型的预测。


```python
# 定义梯度计算函数——这个函数就是对上面的损失函数求导，求出梯度
# w对应beta1到beta4，b对应beta0，X对应X1到X4，y对应Y（实际的值）
def compute_gradients(X, y, w, b):
    z = np.dot(X, w) + b
    probs = sigmoid(z)
    error = probs - y # 对应sigmoid(z)-y
    # 梯度计算，就是前面推导的公式
    grad_w = np.dot(X.T, error) / len(y) # 对应(1/n)*sum((sigmoid(z)-y)*X)
    grad_b = np.mean(error) # 对应(1/n)*sum(sigmoid(z)-y)
    return grad_w, grad_b
```


```python
# 定义梯度下降函数
def gradient_descent(X, y, w, b, alpha, iterations):
    '''
    :param X: 表示特征数据矩阵
    :param y: 表示标签向量（实际值）
    :param w: 权重向量
    :param b: 偏置项
    :param alpha: 学习率——就是我们的步长，每一次更新多少，是一个自己定义的参数
    :param iterations: 迭代次数——就是我们的梯度下降要迭代多少次
    :return: 返回最终的权重和偏置
    '''
    for i in range(iterations):
        grad_w, grad_b = compute_gradients(X, y, w, b)
        # 更新权重和偏置
        w -= alpha * grad_w
        b -= alpha * grad_b
        # 每迭代一定次数打印损失
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {compute_loss(X, y, w, b)}")
    return w, b
```

### 准备数据

- $X_1$: 年龄
- $X_2$: 性别（0表示女性，1表示男性）
- $X_3$: 体重指数（BMI）
- $X_4$: 血糖水平
- $Y$: 是否有糖尿病（0表示没有，1表示有）


```python
# 准备数据
# x = df的前四列，y = df的最后一列
X = df.iloc[:, :-1].values # iloc是通过行号来取行数据的，这里取所有行，去掉最后一列
y = df.iloc[:, -1].values # 取所有行，取最后一列
X.shape, y.shape
```


    ((1000, 4), (1000,))


```python
# 将X和y分为训练集和测试集
from sklearn.model_selection import train_test_split
# 这里就是把X和y分为训练集和测试集，test_size=0.2表示测试集占20%，random_state=0表示随机种子，保证每次运行结果一样
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```


```python
# 特征标准化——目的是让数据的均值为0，方差为1，这样可以加快梯度下降的收敛速度
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```


```python
# 初始化权重和偏置
# 这里的w是一个一维数组，长度为X_train的列数，也就是特征的个数，这里是4
w = np.zeros(X_train.shape[1])
b = 0
```


```python
# 训练模型
# 这里的alpha是学习率，iterations是迭代次数
w, b = gradient_descent(X_train, y_train, w, b, alpha=0.1, iterations=10000)
```

    Iteration 0: Loss = 0.67694848293987
    Iteration 100: Loss = 0.31170512579698983
    Iteration 200: Loss = 0.28006450564964497
    Iteration 300: Loss = 0.26967717122142904
    Iteration 400: Loss = 0.2649489155820069
    Iteration 500: Loss = 0.26246507021700166
    Iteration 600: Loss = 0.2610482148241369
    Iteration 700: Loss = 0.26019578953398675
    .........................................
    Iteration 8300: Loss = 0.2586331617938211
    Iteration 8400: Loss = 0.25863316179381984
    Iteration 8500: Loss = 0.2586331617938189
    Iteration 8600: Loss = 0.25863316179381834
    Iteration 8700: Loss = 0.25863316179381785
    Iteration 8800: Loss = 0.2586331617938175
    Iteration 8900: Loss = 0.2586331617938173
    Iteration 9000: Loss = 0.2586331617938171
    Iteration 9100: Loss = 0.25863316179381696
    Iteration 9200: Loss = 0.25863316179381696
    Iteration 9300: Loss = 0.2586331617938169
    Iteration 9400: Loss = 0.25863316179381685
    Iteration 9500: Loss = 0.25863316179381685
    Iteration 9600: Loss = 0.2586331617938168
    Iteration 9700: Loss = 0.2586331617938168
    Iteration 9800: Loss = 0.2586331617938168
    Iteration 9900: Loss = 0.2586331617938168

```python
# 训练好模型后，我们可以使用它来进行预测
# 预测概率
test_probs = sigmoid(np.dot(X_test, w) + b)
# 预测类别
test_preds = np.where(test_probs > 0.5, 1, 0)
test_preds
```


    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
           1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1])


```python
# 计算准确率
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, test_preds)
accuracy
```


    0.885


```python
# 比较真实值和预测值
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': test_preds,
    'Probability': test_probs
})
results.head(100)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.974574</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0.961203</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0.874061</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0.996859</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0.832528</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0</td>
      <td>1</td>
      <td>0.873257</td>
    </tr>
    <tr>
      <th>96</th>
      <td>1</td>
      <td>1</td>
      <td>0.944671</td>
    </tr>
    <tr>
      <th>97</th>
      <td>1</td>
      <td>1</td>
      <td>0.903943</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1</td>
      <td>1</td>
      <td>0.979292</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1</td>
      <td>1</td>
      <td>0.998257</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>


```python
# 最终的权重和偏置
print(f"Weight: {w}")
print(f"Bias: {b}")
# 最终准确度
print(f"Final Accuracy: {accuracy}")
```

    Weight: [1.41949725 0.42282911 0.70729532 0.90150334]
    Bias: 3.0447132565927673
    Final Accuracy: 0.885


## 总结

在这个项目中，我们手动实现了逻辑回归模型的训练过程。我们首先定义了 Sigmoid 函数和对数损失函数，然后使用梯度下降法最小化损失函数。我们将这个模型应用于一个模拟数据集，其中包含患者的生理指标和是否患有糖尿病的标签。最终，我们评估了模型的准确性，并展示了预测结果。
通过最终的结果，我们可以看到我们的模型在测试集上的准确率为 0.885，说明我们的逻辑回归模型在这个数据集上表现还是不错的。

#### 补充——采用sklearn库实现逻辑回归


```python
# 使用sklearn库实现逻辑回归
from sklearn.linear_model import LogisticRegression
# 创建逻辑回归模型
model = LogisticRegression()
# 训练模型
model.fit(X_train, y_train)
# 预测
y_pred = model.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
accuracy
```


    0.885


```python
# 比较调库实现的逻辑回归和手动实现的逻辑回归的权重和偏置
print("Manual Model:")
print(f"Weight: {w}")
print(f"Bias: {b}")
print("\nSklearn Model:")
print(f"Weight: {model.coef_}")
print(f"Bias: {model.intercept_}")

```

    Manual Model:
    Weight: [1.41949725 0.42282911 0.70729532 0.90150334]
    Bias: 3.0447132565927673
    
    Sklearn Model:
    Weight: [[1.37051336 0.40978953 0.68512706 0.87338347]]
    Bias: [2.988773]


#### 可以发现，两者的权重和偏置非常接近，这说明我们手动实现的逻辑回归模型是正确的。同时，我们也可以看到，使用sklearn库实现的逻辑回归模型的准确率与我们手动实现的模型准确率相同，这说明我们的模型在这个数据集上表现良好。



