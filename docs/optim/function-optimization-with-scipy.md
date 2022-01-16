# 使用 SciPy 的函数优化

> 原文：<https://machinelearningmastery.com/function-optimization-with-scipy/>

最后更新于 2021 年 10 月 12 日

优化包括找到目标函数的输入，从而得到函数的最小或最大输出。

用于科学计算的开源 Python 库 SciPy 提供了一套优化算法。许多算法被用作其他算法的构件，最著名的是 Sklearn 库中的机器学习算法。

这些**优化算法**可以以独立的方式直接用于优化函数。最值得注意的是，局部搜索算法和全局搜索算法，这是你在机器学习项目中可能遇到的两种主要优化类型。

在本教程中，您将发现 SciPy 库提供的优化算法。

完成本教程后，您将知道:

*   SciPy 库为不同的目的提供了一套不同的优化算法。
*   SciPy 中可用的局部搜索优化算法。
*   SciPy 中可用的全局搜索优化算法。

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![Function Optimization With SciPy](img/6865a7189337ca1fce2f4e298ac3c5de.png)

函数优化与 SciPy
照片由[马诺埃尔莱默斯](https://www.flickr.com/photos/mlemos/3125484412/)，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  使用 SciPy 进行优化
2.  使用 SciPy 进行本地搜索
3.  使用 SciPy 进行全局搜索

## 使用 SciPy 进行优化

用于科学计算的 Python SciPy 开源库提供了一套优化技术。

许多算法被用作 SciPy 库中其他算法的构建模块，以及像 Sklearn 这样的机器学习库。

在我们回顾具体的技术之前，让我们来看看库提供的算法类型。

它们是:

*   **标量优化**:凸单变量函数的优化。
*   **局部搜索**:单峰多变量函数的优化。
*   **全局搜索**:多模态多变量函数的优化。
*   **最小二乘**:求解线性和非线性最小二乘问题。
*   **曲线拟合**:将曲线拟合到数据样本。
*   **求根**:求一个函数的根(给出零输出的输入)。
*   **线性规划**:受约束的线性优化。

所有算法都假设正在优化的目标函数是一个最小化函数。如果你的函数是最大化的，可以通过给目标函数返回的值加上一个负号来转换成最小化。

除了上面的列表，该库还提供了一些算法使用的实用函数，以及 Rosenbrock 测试问题。

有关 SciPy 库优化功能的概述，请参见:

*   [优化和寻根(scipy.optimize) API](https://docs.scipy.org/doc/scipy/reference/optimize.html) 。

现在，我们已经对该库支持的优化技术类型有了一个高层次的了解，让我们更仔细地看看我们更有可能在应用机器学习中使用的两组算法。它们是局部搜索和全局搜索。

## 使用 SciPy 进行本地搜索

[局部搜索](https://en.wikipedia.org/wiki/Local_search_(optimization))，或局部函数优化，指的是寻找函数输入的算法，该算法产生最小或最大输出，其中被搜索的函数或约束区域被假设为具有单一最优值，例如单峰。

正在优化的函数可能是凸的，也可能不是凸的，并且可能有一个或多个输入变量。

如果函数被认为或已知是单峰的，则可以直接应用局部搜索优化来优化函数；否则，可以应用局部搜索算法来微调全局搜索算法的结果。

SciPy 库通过[最小化()功能](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)提供本地搜索。

*最小化()*函数将被最小化的目标函数的名称和开始搜索的起始点作为输入，并返回一个[优化结果](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html)，该结果总结了搜索的成功或失败以及解决方案的细节(如果找到的话)。

```py
...
# minimize an objective function
result = minimize(objective, point)
```

如果已知，可以提供关于目标函数的附加信息，例如输入变量的界限、用于计算函数一阶导数的函数(梯度或[雅可比矩阵](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant))、用于计算函数二阶导数的函数([黑森矩阵](https://en.wikipedia.org/wiki/Hessian_matrix))以及对输入的任何约束。

重要的是，该函数提供了“*方法*”参数，该参数允许指定在本地搜索中使用的特定优化。

有一套流行的本地搜索算法，例如:

*   [NelderMead 算法](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)(方法= 'NelderMead')。
*   [牛顿法](https://en.wikipedia.org/wiki/Newton%27s_method)(方法= '牛顿-重心')。
*   [鲍威尔法](https://en.wikipedia.org/wiki/Powell%27s_method)(法= '鲍威尔')。
*   [BFGS 算法和扩展](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)(方法='BFGS ')。

下面的例子演示了如何使用 L-BFGS-B 局部搜索算法求解二维凸函数。

```py
# l-bfgs-b algorithm local optimization of a convex function
from scipy.optimize import minimize
from numpy.random import rand

# objective function
def objective(x):
	return x[0]**2.0 + x[1]**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# define the starting point as a random sample from the domain
pt = r_min + rand(2) * (r_max - r_min)
# perform the l-bfgs-b algorithm search
result = minimize(objective, pt, method='L-BFGS-B')
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

运行该示例将执行优化，并报告搜索的成功或失败、执行的函数评估次数以及导致函数最优的输入。

```py
Status : b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
Total Evaluations: 9
Solution: f([3.38059583e-07 3.70089258e-07]) = 0.00000
```

现在我们已经熟悉了如何使用 SciPy 的本地搜索算法，让我们来看看全局搜索。

## 使用 SciPy 进行全局搜索

[全局搜索](https://en.wikipedia.org/wiki/Global_optimization)或全局函数优化指的是这样的算法，该算法寻找导致最小或最大输出的函数的输入，其中假设被搜索的函数或约束区域具有多个局部最优值，例如多模态。

正在优化的函数通常是非线性的、非凸的，并且可能有一个或多个输入变量。

全局搜索算法通常是随机的，这意味着它们在搜索过程中利用了随机性，并且可能会也可能不会在搜索过程中管理一批候选解决方案。

SciPy 库提供了许多随机全局优化算法，每个算法都通过不同的函数实现。它们是:

*   [通过](https://en.wikipedia.org/wiki/Basin-hopping)[盆地跳跃()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html)功能进行盆地跳跃优化。
*   [通过](https://en.wikipedia.org/wiki/Differential_evolution)[差分进化()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)功能进行差分进化优化。
*   [通过](https://en.wikipedia.org/wiki/Simulated_annealing)[双重退火()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html)功能模拟退火。

该库还提供了用于序列优化的 [shgo()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html) 函数和用于网格搜索优化的[蛮力()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html)。

每个算法返回一个[optimizer result](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html)对象，该对象总结了搜索的成功或失败以及解决方案的细节(如果找到的话)。

下面的例子演示了如何使用模拟退火来求解二维多峰函数。

```py
# simulated annealing global optimization for a multimodal objective function
from scipy.optimize import dual_annealing

# objective function
def objective(v):
	x, y = v
	return (x**2 + y - 11)**2 + (x + y**2 -7)**2

# define range for input
r_min, r_max = -5.0, 5.0
# define the bounds on the search
bounds = [[r_min, r_max], [r_min, r_max]]
# perform the simulated annealing search
result = dual_annealing(objective, bounds)
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

运行该示例将执行优化，并报告搜索的成功或失败、执行的函数评估次数以及导致函数最优的输入。

```py
Status : ['Maximum number of iteration reached']
Total Evaluations: 4028
Solution: f([-3.77931027 -3.283186 ]) = 0.00000
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 蜜蜂

*   [优化(scipy.optimize) API](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html) 。
*   [优化和寻根(scipy.optimize) API](https://docs.scipy.org/doc/scipy/reference/optimize.html) 。

### 文章

*   [本地搜索(优化)，维基百科](https://en.wikipedia.org/wiki/Local_search_(optimization))。
*   [全局优化，维基百科](https://en.wikipedia.org/wiki/Global_optimization)。

## 摘要

在本教程中，您发现了 SciPy 库提供的优化算法。

具体来说，您了解到:

*   SciPy 库为不同的目的提供了一套不同的优化算法。
*   SciPy 中可用的局部搜索优化算法。
*   SciPy 中可用的全局搜索优化算法。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。