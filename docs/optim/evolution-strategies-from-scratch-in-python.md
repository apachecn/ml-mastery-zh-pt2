# Python 中从零开始的进化策略

> 原文：<https://machinelearningmastery.com/evolution-strategies-from-scratch-in-python/>

最后更新于 2021 年 10 月 12 日

**进化策略**是一种随机全局优化算法。

它是一种与其他算法相关的进化算法，例如遗传算法，尽管它是专门为连续函数优化而设计的。

在本教程中，您将发现如何实现进化策略优化算法。

完成本教程后，您将知道:

*   进化策略是一种随机全局优化算法，其灵感来自于自然选择进化的生物学理论。
*   进化策略有一个标准术语，该算法的两个常见版本被称为(μ，λ)-ES 和(μ+λ)-ES。
*   如何将进化策略算法应用于连续目标函数？

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![Evolution Strategies From Scratch in Python](img/ebbd2ddcd6ea8951b03beb0cd2e3e2ab.png)

Python 从零开始的进化策略
图片由[亚历克西斯·a·贝穆德斯](https://www.flickr.com/photos/northn/25266142837/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  进化策略
2.  开发一个(μ，λ)-专家系统
3.  开发一个(μ+λ)-专家系统

## 进化策略

[进化策略](https://en.wikipedia.org/wiki/Evolution_strategy)，有时也称为进化策略(单数)或 es，是一种随机全局优化算法。

这项技术是在 20 世纪 60 年代开发的，由工程师手动实现，用于风洞中的最小阻力设计。

> 被称为进化策略(es)的算法家族是由英戈·雷亨伯格和汉斯-保罗·施韦费尔于 20 世纪 60 年代中期在柏林技术大学开发的。

—第 31 页，[元试探法精要](https://amzn.to/37mn7PN)，2011。

进化策略是一种进化算法，其灵感来自于生物进化理论中的自然选择。与其他进化算法不同，它不使用任何形式的交叉；相反，候选解的修改仅限于变异算子。这样，进化策略可以被认为是一种平行的随机爬山。

该算法涉及最初随机生成的候选解的群体。算法的每一次迭代都包括首先评估解的总体，然后删除除一个子集以外的所有最佳解，这被称为截断选择。剩余的解(父解)每个都被用作生成多个新的候选解(变异)的基础，这些候选解替换父解或与父解竞争群体中的位置，以便在算法的下一次迭代(生成)中考虑。

这个过程有许多变体和一个标准术语来概括算法。群体的大小被称为*λ*，每次迭代选择的亲本数量被称为*μ*。

从每个父代创建的子代的数量计算为(*λ*/*μ*)并且应该选择参数，使得除法没有余数。

*   *亩*:每次迭代选择的亲本数量。
*   *λ*:人口规模。
*   *λ/mu*:从每个选定的父代生成的子代数量。

括号符号用于描述算法配置，例如*(μ，λ)-ES*。比如*μ= 5**λ= 20*，那么就概括为 *(5，20)-ES* 。将*μ*和*λ*参数分开的逗号(，)表示在算法的每次迭代中，孩子直接替换父母。

*   **(mu，lambda)-ES** :一个孩子取代父母的进化策略版本。

mu 和 lambda 参数的加号(+)分隔表示子代和父代将一起定义下一次迭代的群体。

*   **(mu + lambda)-ES** :进化策略的一个版本，将孩子和父母加入到种群中。

随机爬山算法可以作为进化策略来实现，符号为 *(1 + 1)-ES* 。

这是明喻或规范的 ES 算法，文献中描述了许多扩展和变体。

现在我们已经熟悉了进化策略，我们可以探索如何实现该算法。

## 开发一个(μ，λ)-专家系统

在这一节中，我们将开发一个*(μ，λ)-ES*，也就是孩子代替父母的算法版本。

首先，让我们定义一个具有挑战性的优化问题作为实现算法的基础。

[阿克利函数](https://en.wikipedia.org/wiki/Ackley_function)是多模态目标函数的一个例子，它有一个全局最优解和多个局部最优解，局部搜索可能会陷入其中。

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

我们将生成随机候选解决方案以及现有候选解决方案的修改版本。重要的是，所有候选解都在搜索问题的范围内。

为了实现这一点，我们将开发一个函数来检查候选解决方案是否在搜索范围内，然后丢弃它，如果不在，则生成另一个解决方案。

下面的 *in_bounds()* 函数将获取一个候选解(点)和搜索空间边界的定义(边界)，如果解在搜索的边界内，则返回真，否则返回假。

```py
# check if a point is within the bounds of the search
def in_bounds(point, bounds):
	# enumerate all dimensions of the point
	for d in range(len(bounds)):
		# check if out of bounds for this dimension
		if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
			return False
	return True
```

然后我们可以在生成“ *lam* ”(例如*λ*)随机候选解的初始种群时使用这个函数。

例如:

```py
...
# initial population
population = list()
for _ in range(lam):
	candidate = None
	while candidate is None or not in_bounds(candidate, bounds):
		candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	population.append(candidate)
```

接下来，我们可以迭代算法的固定迭代次数。每次迭代首先涉及评估群体中的每个候选解。

我们将计算分数并将它们存储在单独的平行列表中。

```py
...
# evaluate fitness for the population
scores = [objective(c) for c in population]
```

接下来，我们需要选择分数最好的“ *mu* ”父母，在这种情况下分数最低，因为我们正在最小化目标函数。

我们将分两步进行。首先，我们将根据候选解决方案的得分按升序对其进行排名，这样得分最低的解决方案的排名为 0，下一个解决方案的排名为 1，依此类推。我们可以使用 [argsort 函数](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html)的双重调用来实现这一点。

然后我们将使用等级并选择那些等级低于值“ *mu* ”的父母这意味着，如果将 mu 设置为 5 以选择 5 个父代，则只会选择等级在 0 到 4 之间的那些父代。

```py
...
# rank scores in ascending order
ranks = argsort(argsort(scores))
# select the indexes for the top mu ranked solutions
selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
```

然后，我们可以为每个选定的父代创建子代。

首先，我们必须计算每个父代要创建的子代总数。

```py
...
# calculate the number of children per parent
n_children = int(lam / mu)
```

然后，我们可以迭代每个父类，并创建每个父类的修改版本。

我们将使用随机爬山中使用的类似技术来创造孩子。具体来说，将使用高斯分布对每个变量进行采样，当前值作为平均值，标准偏差作为“*步长*”超参数。

```py
...
# create children for parent
for _ in range(n_children):
	child = None
	while child is None or not in_bounds(child, bounds):
		child = population[i] + randn(len(bounds)) * step_size
```

我们还可以检查每个选定的父项是否优于迄今为止看到的最佳解决方案，以便我们可以在搜索结束时返回最佳解决方案。

```py
...
# check if this parent is the best solution ever seen
if scores[i] < best_eval:
	best, best_eval = population[i], scores[i]
	print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
```

可以将创建的子代添加到列表中，我们可以在算法迭代结束时用子代列表替换人口。

```py
...
# replace population with children
population = children
```

我们可以将所有这些联系到一个名为*es _ 逗号()*的函数中，该函数执行进化策略算法的逗号版本。

该函数采用目标函数的名称、搜索空间的边界、迭代次数、步长以及 mu 和 lambda 超参数，并返回在搜索和评估过程中找到的最佳解。

```py
# evolution strategy (mu, lambda) algorithm
def es_comma(objective, bounds, n_iter, step_size, mu, lam):
	best, best_eval = None, 1e+10
	# calculate the number of children per parent
	n_children = int(lam / mu)
	# initial population
	population = list()
	for _ in range(lam):
		candidate = None
		while candidate is None or not in_bounds(candidate, bounds):
			candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
		population.append(candidate)
	# perform the search
	for epoch in range(n_iter):
		# evaluate fitness for the population
		scores = [objective(c) for c in population]
		# rank scores in ascending order
		ranks = argsort(argsort(scores))
		# select the indexes for the top mu ranked solutions
		selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
		# create children from parents
		children = list()
		for i in selected:
			# check if this parent is the best solution ever seen
			if scores[i] < best_eval:
				best, best_eval = population[i], scores[i]
				print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
			# create children for parent
			for _ in range(n_children):
				child = None
				while child is None or not in_bounds(child, bounds):
					child = population[i] + randn(len(bounds)) * step_size
				children.append(child)
		# replace population with children
		population = children
	return [best, best_eval]
```

接下来，我们可以将该算法应用于我们的阿克利目标函数。

我们将运行该算法 5000 次迭代，并在搜索空间中使用 0.15 的步长。我们将使用 100 的种群大小(*λ*)选择 20 个亲本(*亩*)。这些超参数是经过一点点反复试验后选择的。

在搜索结束时，我们将报告在搜索过程中找到的最佳候选解决方案。

```py
...
# seed the pseudorandom number generator
seed(1)
# define range for input
bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
# define the total iterations
n_iter = 5000
# define the maximum step size
step_size = 0.15
# number of parents selected
mu = 20
# the number of children generated by parents
lam = 100
# perform the evolution strategy (mu, lambda) search
best, score = es_comma(objective, bounds, n_iter, step_size, mu, lam)
print('Done!')
print('f(%s) = %f' % (best, score))
```

下面列出了将进化策略算法的逗号版本应用于阿克利目标函数的完整示例。

```py
# evolution strategy (mu, lambda) of the ackley objective function
from numpy import asarray
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import argsort
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed

# objective function
def objective(v):
	x, y = v
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# check if a point is within the bounds of the search
def in_bounds(point, bounds):
	# enumerate all dimensions of the point
	for d in range(len(bounds)):
		# check if out of bounds for this dimension
		if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
			return False
	return True

# evolution strategy (mu, lambda) algorithm
def es_comma(objective, bounds, n_iter, step_size, mu, lam):
	best, best_eval = None, 1e+10
	# calculate the number of children per parent
	n_children = int(lam / mu)
	# initial population
	population = list()
	for _ in range(lam):
		candidate = None
		while candidate is None or not in_bounds(candidate, bounds):
			candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
		population.append(candidate)
	# perform the search
	for epoch in range(n_iter):
		# evaluate fitness for the population
		scores = [objective(c) for c in population]
		# rank scores in ascending order
		ranks = argsort(argsort(scores))
		# select the indexes for the top mu ranked solutions
		selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
		# create children from parents
		children = list()
		for i in selected:
			# check if this parent is the best solution ever seen
			if scores[i] < best_eval:
				best, best_eval = population[i], scores[i]
				print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
			# create children for parent
			for _ in range(n_children):
				child = None
				while child is None or not in_bounds(child, bounds):
					child = population[i] + randn(len(bounds)) * step_size
				children.append(child)
		# replace population with children
		population = children
	return [best, best_eval]

# seed the pseudorandom number generator
seed(1)
# define range for input
bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
# define the total iterations
n_iter = 5000
# define the maximum step size
step_size = 0.15
# number of parents selected
mu = 20
# the number of children generated by parents
lam = 100
# perform the evolution strategy (mu, lambda) search
best, score = es_comma(objective, bounds, n_iter, step_size, mu, lam)
print('Done!')
print('f(%s) = %f' % (best, score))
```

运行该示例会报告候选解决方案，并在每次找到更好的解决方案时进行评分，然后报告在搜索结束时找到的最佳解决方案。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到在搜索过程中大约有 22 项表现改进，最佳解决方案接近 optima。

毫无疑问，这个解决方案可以作为一个起点提供给一个局部搜索算法来进一步细化，这是使用像 ES 这样的全局优化算法时的一个常见做法。

```py
0, Best: f([-0.82977995 2.20324493]) = 6.91249
0, Best: f([-1.03232526 0.38816734]) = 4.49240
1, Best: f([-1.02971385 0.21986453]) = 3.68954
2, Best: f([-0.98361735 0.19391181]) = 3.40796
2, Best: f([-0.98189724 0.17665892]) = 3.29747
2, Best: f([-0.07254927 0.67931431]) = 3.29641
3, Best: f([-0.78716147 0.02066442]) = 2.98279
3, Best: f([-1.01026218 -0.03265665]) = 2.69516
3, Best: f([-0.08851828 0.26066485]) = 2.00325
4, Best: f([-0.23270782 0.04191618]) = 1.66518
4, Best: f([-0.01436704 0.03653578]) = 0.15161
7, Best: f([0.01247004 0.01582657]) = 0.06777
9, Best: f([0.00368129 0.00889718]) = 0.02970
25, Best: f([ 0.00666975 -0.0045051 ]) = 0.02449
33, Best: f([-0.00072633 -0.00169092]) = 0.00530
211, Best: f([2.05200123e-05 1.51343187e-03]) = 0.00434
315, Best: f([ 0.00113528 -0.00096415]) = 0.00427
418, Best: f([ 0.00113735 -0.00030554]) = 0.00337
491, Best: f([ 0.00048582 -0.00059587]) = 0.00219
704, Best: f([-6.91643854e-04 -4.51583644e-05]) = 0.00197
1504, Best: f([ 2.83063223e-05 -4.60893027e-04]) = 0.00131
3725, Best: f([ 0.00032757 -0.00023643]) = 0.00115
Done!
f([ 0.00032757 -0.00023643]) = 0.001147
```

现在我们已经熟悉了如何实现逗号版本的进化策略，让我们看看如何实现 plus 版本。

## 开发一个(μ+λ)-专家系统

进化策略算法的 plus 版本与逗号版本非常相似。

主要的区别是孩子和父母构成了最终的人口，而不仅仅是孩子。这允许父母在算法的下一次迭代中与孩子竞争选择。

这可能导致搜索算法更加贪婪，并可能过早收敛到局部最优(次优解)。好处是，该算法能够利用发现的好的候选解决方案，并专注于该地区的候选解决方案，有可能找到进一步的改进。

我们可以通过修改函数来实现算法的 plus 版本，以便在创建子代时将父代添加到群体中。

```py
...
# keep the parent
children.append(population[i])
```

添加了该功能的更新版本以及新名称 *es_plus()* ，如下所示。

```py
# evolution strategy (mu + lambda) algorithm
def es_plus(objective, bounds, n_iter, step_size, mu, lam):
	best, best_eval = None, 1e+10
	# calculate the number of children per parent
	n_children = int(lam / mu)
	# initial population
	population = list()
	for _ in range(lam):
		candidate = None
		while candidate is None or not in_bounds(candidate, bounds):
			candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
		population.append(candidate)
	# perform the search
	for epoch in range(n_iter):
		# evaluate fitness for the population
		scores = [objective(c) for c in population]
		# rank scores in ascending order
		ranks = argsort(argsort(scores))
		# select the indexes for the top mu ranked solutions
		selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
		# create children from parents
		children = list()
		for i in selected:
			# check if this parent is the best solution ever seen
			if scores[i] < best_eval:
				best, best_eval = population[i], scores[i]
				print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
			# keep the parent
			children.append(population[i])
			# create children for parent
			for _ in range(n_children):
				child = None
				while child is None or not in_bounds(child, bounds):
					child = population[i] + randn(len(bounds)) * step_size
				children.append(child)
		# replace population with children
		population = children
	return [best, best_eval]
```

我们可以将这个版本的算法应用于阿克利目标函数，其超参数与上一节中使用的相同。

下面列出了完整的示例。

```py
# evolution strategy (mu + lambda) of the ackley objective function
from numpy import asarray
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import argsort
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed

# objective function
def objective(v):
	x, y = v
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# check if a point is within the bounds of the search
def in_bounds(point, bounds):
	# enumerate all dimensions of the point
	for d in range(len(bounds)):
		# check if out of bounds for this dimension
		if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
			return False
	return True

# evolution strategy (mu + lambda) algorithm
def es_plus(objective, bounds, n_iter, step_size, mu, lam):
	best, best_eval = None, 1e+10
	# calculate the number of children per parent
	n_children = int(lam / mu)
	# initial population
	population = list()
	for _ in range(lam):
		candidate = None
		while candidate is None or not in_bounds(candidate, bounds):
			candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
		population.append(candidate)
	# perform the search
	for epoch in range(n_iter):
		# evaluate fitness for the population
		scores = [objective(c) for c in population]
		# rank scores in ascending order
		ranks = argsort(argsort(scores))
		# select the indexes for the top mu ranked solutions
		selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
		# create children from parents
		children = list()
		for i in selected:
			# check if this parent is the best solution ever seen
			if scores[i] < best_eval:
				best, best_eval = population[i], scores[i]
				print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
			# keep the parent
			children.append(population[i])
			# create children for parent
			for _ in range(n_children):
				child = None
				while child is None or not in_bounds(child, bounds):
					child = population[i] + randn(len(bounds)) * step_size
				children.append(child)
		# replace population with children
		population = children
	return [best, best_eval]

# seed the pseudorandom number generator
seed(1)
# define range for input
bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
# define the total iterations
n_iter = 5000
# define the maximum step size
step_size = 0.15
# number of parents selected
mu = 20
# the number of children generated by parents
lam = 100
# perform the evolution strategy (mu + lambda) search
best, score = es_plus(objective, bounds, n_iter, step_size, mu, lam)
print('Done!')
print('f(%s) = %f' % (best, score))
```

运行该示例会报告候选解决方案，并在每次找到更好的解决方案时进行评分，然后报告在搜索结束时找到的最佳解决方案。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到在搜索过程中大约有 24 项表现改进。我们还可以看到，在评估值为 0.000532 时，找到了更好的最终解决方案，而在此目标函数上，逗号版本的评估值为 0.001147。

```py
0, Best: f([-0.82977995 2.20324493]) = 6.91249
0, Best: f([-1.03232526 0.38816734]) = 4.49240
1, Best: f([-1.02971385 0.21986453]) = 3.68954
2, Best: f([-0.96315064 0.21176994]) = 3.48942
2, Best: f([-0.9524528 -0.19751564]) = 3.39266
2, Best: f([-1.02643442 0.14956346]) = 3.24784
2, Best: f([-0.90172166 0.15791013]) = 3.17090
2, Best: f([-0.15198636 0.42080645]) = 3.08431
3, Best: f([-0.76669476 0.03852254]) = 3.06365
3, Best: f([-0.98979547 -0.01479852]) = 2.62138
3, Best: f([-0.10194792 0.33439734]) = 2.52353
3, Best: f([0.12633886 0.27504489]) = 2.24344
4, Best: f([-0.01096566 0.22380389]) = 1.55476
4, Best: f([0.16241469 0.12513091]) = 1.44068
5, Best: f([-0.0047592 0.13164993]) = 0.77511
5, Best: f([ 0.07285478 -0.0019298 ]) = 0.34156
6, Best: f([-0.0323925 -0.06303525]) = 0.32951
6, Best: f([0.00901941 0.0031937 ]) = 0.02950
32, Best: f([ 0.00275795 -0.00201658]) = 0.00997
109, Best: f([-0.00204732 0.00059337]) = 0.00615
195, Best: f([-0.00101671 0.00112202]) = 0.00434
555, Best: f([ 0.00020392 -0.00044394]) = 0.00139
2804, Best: f([3.86555110e-04 6.42776651e-05]) = 0.00111
4357, Best: f([ 0.00013889 -0.0001261 ]) = 0.00053
Done!
f([ 0.00013889 -0.0001261 ]) = 0.000532
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [进化策略:综合介绍](https://link.springer.com/article/10.1023/A:1015059928466)，2002。

### 书

*   [元试探法精要](https://amzn.to/37mn7PN)，2011。
*   [优化算法](https://amzn.to/3je8O1J)，2019。
*   [计算智能:导论](https://amzn.to/3ob61KA)，2007。

### 文章

*   [进化策略，维基百科](https://en.wikipedia.org/wiki/Evolution_strategy)。
*   [进化策略，Scholarpedia](http://www.scholarpedia.org/article/Evolution_strategies) 。

## 摘要

在本教程中，您发现了如何实现进化策略优化算法。

具体来说，您了解到:

*   进化策略是一种随机全局优化算法，其灵感来自于自然选择进化的生物学理论。
*   进化策略有一个标准术语，该算法的两个常见版本被称为(μ，λ)-ES 和(μ+λ)-ES。
*   如何将进化策略算法应用于连续目标函数？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。