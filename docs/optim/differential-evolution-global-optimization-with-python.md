# Python 差分进化的全局优化

> 原文：<https://machinelearningmastery.com/differential-evolution-global-optimization-with-python/>

最后更新于 2021 年 10 月 12 日

**差分进化**是一种全局优化算法。

它是一种进化算法，与遗传算法等其他进化算法相关。

与遗传算法不同，它是专门设计来处理实数向量而不是位串的。与遗传算法不同的是，它使用向量减法和加法等向量运算来导航搜索空间，而不是遗传学启发的变换。

在本教程中，您将发现差分进化全局优化算法。

完成本教程后，您将知道:

*   差分进化优化是一种进化算法，设计用于处理实值候选解。
*   如何使用 python 中的差分进化优化算法 API？
*   用差分进化算法解决多目标全局优化问题的例子。

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![Differential Evolution Global Optimization With Python](img/878b5e4a0f34438a3deb4dfefb7b9e88.png)

用 Python 进行差分进化全局优化
图片由 [Gergely Csatari](https://www.flickr.com/photos/macskapocs/44808125301/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  差分进化
2.  差分进化应用编程接口
3.  差分进化成功的例子

## 差分进化

[差分进化](https://en.wikipedia.org/wiki/Differential_evolution)，简称 DE，是一种随机全局搜索优化算法。

它是一种进化算法，与遗传算法等其他进化算法相关。与使用比特序列表示候选解的遗传算法不同，差分进化被设计为处理连续目标函数的多维实值候选解。

> 差分进化算法可以说是目前使用的最强大的随机实参数优化算法之一。

——[差分进化:最先进水平的调查](https://ieeexplore.ieee.org/abstract/document/5601760/)，2011 年。

该算法在搜索中不使用梯度信息，因此非常适合于非微分非线性目标函数。

该算法的工作原理是保持由实值向量表示的候选解的总体。新的候选解决方案是通过对现有解决方案进行修改来创建的，然后在算法的每次迭代中替换大部分人口。

使用“*策略*创建新的候选解，该策略包括选择添加了突变的基础解，以及从群体中计算突变量和类型的其他候选解，称为差异向量。例如，策略可以选择最佳候选解作为突变中差异向量的基础和随机解。

> DE 通过将两个群体向量之间的加权差加到第三个向量来生成新的参数向量。

——[差分进化:连续空间上全局优化的一种简单有效的启发式算法](https://link.springer.com/article/10.1023/A:1008202821328)，1997。

如果孩子有更好的目标函数评估，基础解将被他们的孩子替换。

> 最后，在我们建立了一组新的子代之后，我们将每个子代与创建它的父代进行比较(每个父代创建一个子代)。如果子代比父代更好，它将替换原始群体中的父代。

—第 51 页，[元试探法精要](https://amzn.to/2HxZVn4)，2011。

突变被计算为候选解对之间的差，该差产生差向量，该差向量然后被添加到基础解，由范围[0，2]中的突变因子超参数集加权。

不是基础溶液的所有元素都发生了突变。这是通过重组超参数控制的，通常设置为一个较大的值，例如 80%，这意味着基本解中的大部分但不是所有变量都被替换。通过对概率分布(如二项式或指数分布)进行采样，为每个位置分别确定保留或替换基础解中的值的决定。

标准术语用于描述差异化策略，形式如下:

*   DE/x/y/z

其中 DE 代表“*差分进化*”，“ *x* 定义待变异的基础解，如“*兰德*”代表随机或“*最佳*”代表种群中的最佳解。 *y* 代表添加到基础解的差向量的数量，如 1， *z* 定义了确定每个解在群体中是否被保留或替换的概率分布，如二项式的 *bin* 或指数式的 *exp* 。

> 上面使用的一般惯例是 DE/x/y/z，其中 DE 代表“差分进化”，x 代表表示要扰动的基向量的字符串，y 是考虑扰动 x 的差向量的数量，z 代表使用的交叉类型(exp:指数；bin:二项式)。

——[差分进化:最先进水平的调查](https://ieeexplore.ieee.org/abstract/document/5601760/)，2011 年。

配置 DE/best/1/bin 和 DE/best/2/bin 是受欢迎的配置，因为它们对许多目标功能表现良好。

> Mezura-Montes 等人进行的实验表明，无论要解决的问题的特征如何，基于结果的最终准确性和鲁棒性，DE/best/1/bin(总是使用最佳解来寻找搜索方向以及二项式交叉)仍然是最具竞争力的方案。

——[差分进化:最先进水平的调查](https://ieeexplore.ieee.org/abstract/document/5601760/)，2011 年。

现在我们已经熟悉了差分进化算法，让我们看看如何使用 SciPy API 实现。

## 差分进化应用编程接口

差分进化全局优化算法在 Python 中通过[差分进化()SciPy 函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)提供。

该函数将目标函数的名称和每个输入变量的边界作为搜索的最小参数。

```py
...
# perform the differential evolution search
result = differential_evolution(objective, bounds)
```

有许多附加的搜索超参数具有默认值，尽管您可以配置它们来自定义搜索。

一个关键的超参数是“*策略*”参数，它控制执行的差分进化搜索的类型。默认情况下，此设置为“*best 1 bin*”(DE/best/1/bin)，对于大多数问题来说，这是一个不错的配置。它通过从总体中选择随机解，从另一个中减去一个，并将差值的比例版本添加到总体中的最佳候选解来创建新的候选解。

*   new = best +(突变*(rand 1–rand 2))

“ *popsize* ”参数控制群体中保留的候选解决方案的大小或数量。它是候选解决方案中维度数量的一个因素，默认情况下，它被设置为 15。这意味着对于 2D 目标函数，将保持(2 * 15)或 30 个候选解的人口规模。

算法的总迭代次数由“ *maxiter* 参数维护，默认为 1，000。

“*突变*”参数控制每次迭代对候选解所做的更改数量。默认情况下，该值设置为 0.5。重组量通过“*重组*参数控制，默认情况下，该参数设置为 0.7(给定候选溶液的 70%)。

最后，对搜索结束时找到的最佳候选解进行局部搜索。这是通过“*抛光*”参数控制的，该参数默认设置为真。

搜索的结果是一个[optimizer result](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html)对象，在该对象中可以像字典一样访问属性。搜索成功与否可通过“*成功*或“*消息*键进入。

可通过“ *nfev* 访问功能评估的总数，可通过“ *x* 键访问为搜索找到的最佳输入。

现在我们已经熟悉了 Python 中的差分进化 API，让我们来看看一些工作过的例子。

## 差分进化成功的例子

在本节中，我们将看一个在具有挑战性的目标函数上使用差分进化算法的例子。

[阿克利函数](https://en.wikipedia.org/wiki/Ackley_function)是一个目标函数的例子，它有一个全局最优解和多个局部最优解，局部搜索可能会陷入其中。

因此，需要一种全局优化技术。它是一个二维目标函数，其全局最优值为[0，0]，计算结果为 0.0。

下面的示例实现了 Ackley，并创建了一个显示全局最优值和多个局部最优值的三维曲面图。

```py
# ackley multimodal function
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# objective function
def objective(x, y):
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a surface plot with the jet color scheme
figure = pyplot.figure()
axis = figure.gca(projection='3d')
axis.plot_surface(x, y, results, cmap='jet')
# show the plot
pyplot.show()
```

运行该示例会创建阿克利函数的曲面图，显示大量的局部最优值。

![3D Surface Plot of the Ackley Multimodal Function](img/b9be2d2844dbd19c272458ee43960f1c.png)

阿克利多峰函数的三维表面图

我们可以将差分进化算法应用于阿克利目标函数。

首先，我们可以将搜索空间的边界定义为函数在每个维度上的极限。

```py
...
# define the bounds on the search
bounds = [[r_min, r_max], [r_min, r_max]]
```

然后，我们可以通过指定目标函数的名称和搜索范围来应用搜索。在这种情况下，我们将使用默认的超参数。

```py
...
# perform the differential evolution search
result = differential_evolution(objective, bounds)
```

搜索完成后，它将报告搜索状态和执行的迭代次数，以及通过评估找到的最佳结果。

```py
...
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

将这些联系在一起，下面列出了将差分进化应用于阿克利目标函数的完整示例。

```py
# differential evolution global optimization for the ackley multimodal objective function
from scipy.optimize import differential_evolution
from numpy.random import rand
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi

# objective function
def objective(v):
	x, y = v
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# define range for input
r_min, r_max = -5.0, 5.0
# define the bounds on the search
bounds = [[r_min, r_max], [r_min, r_max]]
# perform the differential evolution search
result = differential_evolution(objective, bounds)
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

运行该示例会执行优化，然后报告结果。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，该算法定位了输入等于零且目标函数评估等于零的最优值。

我们可以看到总共进行了 3063 次功能评估。

```py
Status: Optimization terminated successfully.
Total Evaluations: 3063
Solution: f([0\. 0.]) = 0.00000
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [差分进化——连续空间上全局优化的简单有效的启发式算法](https://link.springer.com/article/10.1023/A:1008202821328)，1997。
*   [差分进化:最先进水平的调查](https://ieeexplore.ieee.org/abstract/document/5601760/)，2011 年。

### 书

*   [优化算法](https://amzn.to/2Traqek)，2019。
*   [元试探法精要](https://amzn.to/2HxZVn4)，2011。
*   [计算智能:导论](https://amzn.to/2HzjbjV)，2007。

### 蜜蜂

*   [scipy . optimize . differential _ evolution API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)。
*   [scipy . optimize . optimizer result API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html)。

### 文章

*   [差分进化，维基百科](https://en.wikipedia.org/wiki/Differential_evolution)。

## 摘要

在本教程中，您发现了差分进化全局优化算法。

具体来说，您了解到:

*   差分进化优化是一种进化算法，设计用于处理实值候选解。
*   如何使用 python 中的差分进化优化算法 API？
*   用差分进化算法解决多目标全局优化问题的例子。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。