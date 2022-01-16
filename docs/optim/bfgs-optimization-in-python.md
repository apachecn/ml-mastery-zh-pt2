# BFGS 优化算法的温和介绍

> 原文：<https://machinelearningmastery.com/bfgs-optimization-in-python/>

最后更新于 2021 年 10 月 12 日

布赖登算法、弗莱彻算法、戈德法布算法和夏诺算法，或称 **BFGS 算法**，是一种局部搜索优化算法。

它是一种二阶优化算法，这意味着它利用了目标函数的二阶导数，属于一类被称为拟牛顿法的算法，这些算法对不能计算二阶导数的优化问题近似二阶导数(称为黑森法)。

BFGS 算法可能是数值优化中最广泛使用的二阶算法之一，通常用于拟合机器学习算法，如逻辑回归算法。

在本教程中，您将发现 BFGS 二阶优化算法。

完成本教程后，您将知道:

*   二阶优化算法是利用二阶导数的算法，二阶导数被称为多元目标函数的 Hessian 矩阵。
*   BFGS 算法可能是最受欢迎的数值优化二阶算法，属于一组所谓的拟牛顿方法。
*   如何使用 Python 中的 BFGS 和 L-BFGS-B 算法最小化目标函数。

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![A Gentle Introduction to the BFGS Optimization Algorithm](img/b68fe5b71c9cf49753b234a2b56227ac.png)

温和介绍 BFGS 优化算法
图片由[提莫牛顿-西蒙斯](https://www.flickr.com/photos/timo_w2s/5902273583/)，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  二阶优化算法
2.  BFGS 优化算法
3.  BFGS 的成功范例

## 二阶优化算法

优化包括找到最大化或最小化目标函数的输入参数值。

牛顿法优化算法是那些利用目标函数二阶导数的算法。

你可能从微积分中回想起，函数的[一阶导数](https://en.wikipedia.org/wiki/Derivative)是函数在特定点的变化率或曲率。导数可以通过优化算法朝着函数的最小值(导致目标函数最小输出的输入值)沿着下坡(或上坡)行进。

利用一阶导数的算法称为一阶优化算法。一阶算法的一个例子是梯度下降优化算法。

*   **一阶方法**:利用一阶导数寻找目标函数最优值的优化算法。

[二阶导数](https://en.wikipedia.org/wiki/Second_derivative)是导数的导数，或变化率的变化率。

二阶导数可以用来更有效地定位目标函数的最优值。这在一般情况下更有意义，因为我们关于目标函数的信息越多，优化它就越容易。

二阶导数使我们既能知道向哪个方向移动(像一阶导数一样)，又能估计向那个方向移动多远，称为步长。

> 另一方面，二阶信息允许我们对目标函数进行二次近似，并近似合适的步长以达到局部最小值…

—第 87 页，[优化算法](https://amzn.to/3i56bQZ)，2019。

利用二阶导数的算法被称为二阶优化算法。

*   **二阶方法**:利用二阶导数寻找目标函数最优值的优化算法。

二阶优化算法的一个例子是牛顿法。

当一个目标函数有一个以上的输入变量时，这些输入变量可以一起被认为是一个向量，这可能是线性代数所熟悉的。

> 梯度是多元函数导数的推广。它捕捉了函数的局部斜率，使我们能够预测从一个点向任何方向迈出一小步的效果。

—第 21 页，[优化算法](https://amzn.to/3i56bQZ)，2019。

类似地，多个输入变量的一阶导数也可以是向量，其中每个元素称为偏导数。这个偏导数向量被称为梯度。

*   **梯度**:目标函数多个输入变量的偏一阶导数向量。

这一思想推广到多元输入的二阶导数，这是一个包含二阶导数的矩阵，称为 Hessian 矩阵。

*   **黑森**:目标函数多个输入变量的偏二阶导数矩阵。

如果在我们计算导数的点上，二阶导数都是连续的，那么黑森矩阵是正方形和对称的。这通常是解决实值优化问题时的情况，也是使用许多二阶方法时的预期。

> 多元函数的 Hessian 是包含关于输入的所有二阶导数的矩阵。二阶导数捕捉关于函数局部曲率的信息。

—第 21 页，[优化算法](https://amzn.to/3i56bQZ)，2019。

因此，通常描述二阶优化算法利用或遵循目标函数的黑森最优。

现在我们对二阶优化算法有了一个高层次的理解，让我们仔细看看 BFGS 算法。

## BFGS 优化算法

**BFGS** 是二阶优化算法。

这是一个首字母缩略词，以该算法的四个共同发现命名:布赖登、弗莱彻、戈德法布和尚诺。

这是一种局部搜索算法，旨在解决具有单一最优解的凸优化问题。

BFGS 算法可能最好理解为属于一组算法，这些算法是牛顿方法优化算法的扩展，被称为准牛顿方法。

牛顿法是一种利用黑森矩阵的二阶优化算法。

牛顿法的一个局限性是它需要计算黑森矩阵的逆。这是一个计算量很大的操作，根据目标函数的性质，可能不稳定。

拟牛顿法是二阶优化算法，它利用梯度近似黑森矩阵的逆，这意味着黑森矩阵及其逆不需要在算法的每一步都可用或精确计算。

> 拟牛顿法是非线性优化中应用最广泛的方法之一。它们被集成到许多软件库中，在解决各种中小型问题时非常有效，尤其是在黑森难以计算的情况下。

—第 411 页，[线性和非线性优化](https://amzn.to/39fWKtS)，2009。

不同拟牛顿优化算法之间的主要区别在于计算逆黑森近似的具体方式。

BFGS 算法是更新逆 Hessian 计算的一种特定方式，而不是每次迭代都重新计算它。它，或者它的扩展，可能是最流行的拟牛顿甚至二阶优化算法之一，用于数值优化。

> 最流行的拟牛顿算法是 BFGS 方法，以它的发现者布赖登、弗莱彻、戈德法布和夏诺命名。

—第 136 页，[数值优化](https://amzn.to/3sbjF2t)，2006。

使用黑森函数(如果可用)的一个好处是，它可以用来确定移动的方向和步长，以改变输入参数，从而最小化(或最大化)目标函数。

像 BFGS 这样的拟牛顿方法近似逆黑森，然后可以用来确定移动的方向，但我们不再有步长。

BFGS 算法通过在选定的方向上使用直线性搜索来确定在该方向上移动多远来解决这个问题。

对于 BFGS 算法使用的推导和计算，我推荐本教程末尾的进一步阅读部分中的资源。

Hessian 及其反函数的大小与目标函数的输入参数数量成正比。因此，对于数百、数千或数百万个参数，矩阵的大小会变得非常大。

> ……BFGS 算法必须存储需要 O(n2)内存的逆 Hessian 矩阵 M，这使得 BFGS 算法对于大多数现代深度学习模型来说不切实际，这些模型通常有数百万个参数。

—第 317 页，[深度学习](https://amzn.to/3oEyDeU)，2016。

[有限内存 BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) (或 L-BFGS)是 BFGS 算法的扩展，解决了拥有大量参数的成本。它通过不要求存储逆矩阵的整个近似，通过假设在算法的前一次迭代中(在近似中使用)对逆矩阵 Hessian 进行简化来实现这一点。

现在，我们已经从高层次上熟悉了 BFGS 算法，让我们看看如何利用它。

## BFGS 的成功范例

在本节中，我们将看一些使用 BFGS 优化算法的例子。

我们可以使用[最小化()SciPy 函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)在 Python 中实现优化任意函数的 BFGS 算法。

该函数需要多个参数，但最重要的是，我们可以将目标函数的名称指定为第一个参数，将搜索的起点指定为第二个参数，并将“*方法*参数指定为“ *BFGS* ”。用于计算目标函数导数的函数的名称可以通过“ *jac* ”参数指定。

```py
...
# perform the bfgs algorithm search
result = minimize(objective, pt, method='BFGS', jac=derivative)
```

我们来看一个例子。

首先，我们可以定义一个简单的二维目标函数，一个碗函数，例如 x^2.简单来说，就是输入变量的平方和，在 f(0，0) = 0.0 时取一个最优值。

```py
# objective function
def objective(x):
	return x[0]**2.0 + x[1]**2.0
```

接下来，让我们为函数的导数定义一个函数，它是[x*2，y*2]。

```py
# derivative of the objective function
def derivative(x):
	return [x[0] * 2, x[1] * 2]
```

我们将把函数的边界定义为一个盒子，每个维度的范围是-5 和 5。

```py
...
# define range for input
r_min, r_max = -5.0, 5.0
```

搜索的起点将是搜索域中随机生成的位置。

```py
...
# define the starting point as a random sample from the domain
pt = r_min + rand(2) * (r_max - r_min)
```

然后，我们可以通过指定目标函数的名称、起始点、我们想要使用的方法(BFGS)和导数函数的名称，应用 BFGS 算法来找到目标函数的最小值。

```py
...
# perform the bfgs algorithm search
result = minimize(objective, pt, method='BFGS', jac=derivative)
```

然后，我们可以查看结果，报告一条消息，说明算法是否成功完成，以及已执行的目标函数的评估总数。

```py
...
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
```

最后，我们可以报告找到的输入变量及其对目标函数的评估。

```py
...
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

将这些联系在一起，完整的示例如下所示。

```py
# bfgs algorithm local optimization of a convex function
from scipy.optimize import minimize
from numpy.random import rand

# objective function
def objective(x):
	return x[0]**2.0 + x[1]**2.0

# derivative of the objective function
def derivative(x):
	return [x[0] * 2, x[1] * 2]

# define range for input
r_min, r_max = -5.0, 5.0
# define the starting point as a random sample from the domain
pt = r_min + rand(2) * (r_max - r_min)
# perform the bfgs algorithm search
result = minimize(objective, pt, method='BFGS', jac=derivative)
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

运行该示例将 BFGS 算法应用于我们的目标函数，并报告结果。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到算法执行了四次迭代，发现了非常接近 optima f(0.0，0.0) = 0.0 的解，至少达到了有用的准确率水平。

```py
Status: Optimization terminated successfully.
Total Evaluations: 4
Solution: f([0.00000000e+00 1.11022302e-16]) = 0.00000
```

*最小化()*功能还支持 L-BFGS 算法，该算法的内存需求低于 BFGS。

具体来说，算法的 L-BFGS-B 版本，其中-B 后缀表示算法的“*盒装*”版本，其中可以指定域的边界。

这可以通过将“*方法*参数指定为“ *L-BFGS-B* 来实现。

```py
...
# perform the l-bfgs-b algorithm search
result = minimize(objective, pt, method='L-BFGS-B', jac=derivative)
```

下面列出了此更新的完整示例。

```py
# l-bfgs-b algorithm local optimization of a convex function
from scipy.optimize import minimize
from numpy.random import rand

# objective function
def objective(x):
	return x[0]**2.0 + x[1]**2.0

# derivative of the objective function
def derivative(x):
	return [x[0] * 2, x[1] * 2]

# define range for input
r_min, r_max = -5.0, 5.0
# define the starting point as a random sample from the domain
pt = r_min + rand(2) * (r_max - r_min)
# perform the l-bfgs-b algorithm search
result = minimize(objective, pt, method='L-BFGS-B', jac=derivative)
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

运行示例应用程序将 L-BFGS-B 算法应用于我们的目标函数并报告结果。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

同样，我们可以看到函数的最小值是在很少的评估中找到的。

```py
Status : b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
Total Evaluations: 3
Solution: f([-1.33226763e-15 1.33226763e-15]) = 0.00000
```

将测试问题的维度增加到数百万个参数，并比较两种算法的内存使用和运行时间，这可能是一个有趣的练习。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [优化算法](https://amzn.to/3i56bQZ)，2019。
*   [深度学习](https://amzn.to/3oEyDeU)，2016 年。
*   [数值优化](https://amzn.to/3sbjF2t)，2006。
*   [线性与非线性优化](https://amzn.to/39fWKtS)，2009。

### 蜜蜂

*   [scipy . optimize . minimum API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)。

### 文章

*   [Broyden–Fletcher–Goldfarb–Shanno 算法，维基百科](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)。
*   [有限记忆 BFGS，维基百科](https://en.wikipedia.org/wiki/Limited-memory_BFGS)。

## 摘要

在本教程中，您发现了 BFGS 二阶优化算法。

具体来说，您了解到:

*   二阶优化算法是利用二阶导数的算法，二阶导数被称为多元目标函数的 Hessian 矩阵。
*   BFGS 算法可能是最受欢迎的数值优化二阶算法，属于一组所谓的拟牛顿方法。
*   如何使用 Python 中的 BFGS 和 L-BFGS-B 算法最小化目标函数。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。