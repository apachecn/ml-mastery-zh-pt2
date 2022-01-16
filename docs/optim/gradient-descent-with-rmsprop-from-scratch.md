# 从零开始的 RMSProp 梯度下降

> 原文：<https://machinelearningmastery.com/gradient-descent-with-rmsprop-from-scratch/>

最后更新于 2021 年 10 月 12 日

梯度下降是一种优化算法，它遵循目标函数的负梯度来定位函数的最小值。

梯度下降的一个限制是，它对每个输入变量使用相同的步长(学习率)。简而言之，AdaGrad 是梯度下降优化算法的扩展，它允许优化算法使用的每个维度中的步长根据搜索过程中变量(偏导数)的梯度自动调整。

AdaGrad 的一个限制是，在搜索结束时，它会导致每个参数的步长非常小，这可能会大大减慢搜索进度，并可能意味着找不到最优解。

**均方根传播**或**均方根传播**是梯度下降的扩展，是梯度下降的 AdaGrad 版本，它使用部分梯度的衰减平均值来适应每个参数的步长。衰减移动平均线的使用允许算法忘记早期的梯度，并专注于搜索过程中最近观察到的部分梯度，克服了 AdaGrad 的限制。

在本教程中，您将发现如何使用 RMSProp 优化算法从零开始开发梯度下降。

完成本教程后，您将知道:

*   梯度下降是一种优化算法，它使用目标函数的梯度来导航搜索空间。
*   梯度下降可以更新为使用一个自动自适应的步长为每个输入变量使用一个衰减的移动平均偏导数，称为 RMSProp。
*   如何从零开始实现 RMSProp 优化算法，并将其应用于目标函数并评估结果。

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![Gradient Descent With RMSProp from Scratch](img/f9d7ae2e938fe388f0965a5658dac49a.png)

带 RMSProp 的梯度下降从零开始
照片由[帕维尔·艾哈迈德](https://www.flickr.com/photos/pavelahmed/6568904661/)拍摄，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  梯度下降
2.  均方根传播
3.  带 RMSProp 的梯度下降
    1.  二维测试问题
    2.  基于 RMSProp 的梯度下降优化
    3.  RMSProp 的可视化

## 梯度下降

[梯度下降](https://en.wikipedia.org/wiki/Gradient_descent)是一种优化算法。

它在技术上被称为一阶优化算法，因为它明确地利用了目标函数的一阶导数。

> 一阶方法依赖于梯度信息来帮助指导搜索最小值…

—第 69 页，[优化算法](https://amzn.to/39KZSQn)，2019。

[一阶导数](https://en.wikipedia.org/wiki/Derivative)，或简称为“导数”，是目标函数在特定点(例如特定输入)的变化率或斜率。

如果目标函数有多个输入变量，它被称为多元函数，输入变量可以被认为是一个向量。反过来，多元目标函数的导数也可以作为向量，并且通常被称为[梯度](https://en.wikipedia.org/wiki/Gradient)。

*   **梯度**:多元目标函数的一阶导数。

对于特定输入，导数或梯度指向目标函数最陡上升的方向。

梯度下降是指一种最小化优化算法，它遵循目标函数梯度下降的负值来定位函数的最小值。

梯度下降算法需要一个正在优化的目标函数和目标函数的导数函数。目标函数 *f()* 返回给定输入集的得分，导数函数 *f'()* 给出给定输入集的目标函数的导数。

梯度下降算法要求问题中有一个起点( *x* )，比如输入空间中随机选择的一个点。

然后计算导数，并在输入空间中采取一个步骤，该步骤预计会导致目标函数的下坡运动，假设我们正在最小化目标函数。

下坡移动是通过首先计算在输入空间中移动多远来实现的，计算方法是步长(称为 alpha 或学习率)乘以梯度。然后从当前点减去这一点，确保我们逆着梯度或目标函数向下移动。

*   x = x–步长* f'(x)

给定点处的目标函数越陡，梯度的幅度就越大，反过来，搜索空间中的步长就越大。使用步长超参数来缩放所采取的步长。

*   **步长**(*α*):超参数，控制算法每次迭代在搜索空间中逆着梯度移动多远。

如果步长太小，搜索空间中的移动将会很小，并且搜索将会花费很长时间。如果步长过大，搜索可能会绕过搜索空间并跳过 optima。

现在我们已经熟悉了梯度下降优化算法，让我们来看看 RMSProp。

## 均方根传播

均方根传播，简称 RMSProp，是梯度下降优化算法的扩展。

这是一个未发表的扩展，首先在杰弗里·辛顿的神经网络课程讲义中描述，特别是第 6e 讲，题目是“ [rmsprop:将梯度除以其最近幅度的运行平均值](http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf)”

RMSProp 旨在加速优化过程，例如减少达到最优所需的函数求值次数，或者提高优化算法的能力，例如产生更好的最终结果。

它与梯度下降的另一个扩展有关，称为自适应梯度，或 AdaGrad。

AdaGrad 旨在专门探索为搜索空间中的每个参数自动定制步长(学习率)的想法。这是通过首先计算给定维度的步长，然后使用计算出的步长在该维度上使用偏导数进行移动来实现的。然后对搜索空间中的每个维度重复这个过程。

Adagrad 计算每个参数的步长，首先对搜索过程中所见参数的偏导数求和，然后将初始步长超参数除以偏导数平方和的平方根。

一个参数的自定义步长计算如下:

*   cust _ step _ size = step _ size/(1e-8+sqrt)

其中 *cust_step_size* 是搜索过程中给定点输入变量的计算步长， *step_size* 是初始步长， *sqrt()* 是平方根运算， *s* 是迄今为止搜索过程中看到的输入变量的平方偏导数之和。

这具有平滑搜索空间中曲率很大的优化问题的振荡的效果。

> AdaGrad 根据平方梯度的整个历史来缩小学习率，在达到这样的凸结构之前，可能已经使学习率变得太小了。

—第 307-308 页，[深度学习](https://amzn.to/3qSk3C2)，2016。

AdaGrad 的一个问题是，它会使搜索速度减慢太多，导致运行结束时搜索的每个参数或维度的学习率非常小。这具有在找到最小值之前过早停止搜索的效果。

> RMSProp 扩展了 Adagrad，以避免单调递减的学习率的影响。

—第 78 页，[优化算法](https://amzn.to/39KZSQn)，2019。

RMSProp 可以被认为是 AdaGrad 的扩展，因为它在计算每个参数的学习率时使用偏导数的衰减平均值或移动平均值，而不是总和。

这是通过增加一个新的超参数来实现的，我们称之为*ρ*，它的作用类似于偏导数的动量。

> RMSProp 保持平方梯度的衰减平均值。

—第 78 页，[优化算法](https://amzn.to/39KZSQn)，2019。

使用偏导数的衰减移动平均允许搜索忘记早期的偏导数值，并且集中于搜索空间的最近看到的形状。

> RMSProp 使用指数衰减的平均值来丢弃极端过去的历史，这样它就可以在找到一个凸碗后快速收敛，就好像它是在那个碗内初始化的 AdaGrad 算法的一个实例。

—第 308 页，[深度学习](https://amzn.to/3qSk3C2)，2016。

一个参数的均方偏导数计算如下:

*   s(t+1)=(s(t)*ρ)+(f'(x(t))^2 *(1.0-ρ))

其中 s(t+1)是算法当前迭代的一个参数的平方偏导数的衰减移动平均，s(t)是前一次迭代的衰减移动平均平方偏导数，f'(x(t))^2 是当前参数的平方偏导数，ρ是超参数，通常具有类似动量的 0.9 的值。

假设我们使用的是偏导数的衰减平均值，并计算该平均值的平方根，这种技术就有了它的名字，例如，偏导数的均方根或均方根(RMS)。例如，参数的自定义步长可以写成:

*   cust _ step _ size(t+1)= step _ size/(1e-8+RMS(s(t+1)))

一旦我们有了参数的自定义步长，我们就可以使用自定义步长和偏导数*f’(x(t))*来更新参数。

*   x(t+1)= x(t)–cust _ step _ size(t+1)* f '(x(t))

然后对每个输入变量重复这个过程，直到在搜索空间中创建一个新的点并可以对其进行评估。

RMSProp 是梯度下降的非常有效的扩展，并且是通常用于拟合深度学习神经网络的优选方法之一。

> 经验证明，RMSProp 是一种有效而实用的深度神经网络优化算法。它是深度学习实践者经常使用的优化方法之一。

—第 308 页，[深度学习](https://amzn.to/3qSk3C2)，2016。

现在我们已经熟悉了 RMSprop 算法，让我们探索如何实现它并评估它的表现。

## 带 RMSProp 的梯度下降

在本节中，我们将探索如何使用 RMSProp 算法实现具有自适应梯度的梯度下降优化算法。

### 二维测试问题

首先，让我们定义一个优化函数。

我们将使用一个简单的二维函数，它对每个维度的输入进行平方，并定义从-1.0 到 1.0 的有效输入范围。

下面的*目标()*函数实现该功能

```py
# objective function
def objective(x, y):
	return x**2.0 + y**2.0
```

我们可以创建数据集的三维图，以获得对响应表面曲率的感觉。

下面列出了绘制目标函数的完整示例。

```py
# 3d plot of the test function
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# define range for input
r_min, r_max = -1.0, 1.0
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

运行该示例会创建目标函数的三维表面图。

我们可以看到熟悉的碗形，全局最小值在 f(0，0) = 0。

![Three-Dimensional Plot of the Test Objective Function](img/a7e7279e861cce6af4f38fc500f83405.png)

测试目标函数的三维图

我们还可以创建函数的二维图。这将有助于我们以后绘制搜索进度。

以下示例创建了目标函数的等高线图。

```py
# contour plot of the test function
from numpy import asarray
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# show the plot
pyplot.show()
```

运行该示例会创建目标函数的二维等高线图。

我们可以看到碗的形状被压缩成带有颜色梯度的轮廓。我们将使用此图来绘制搜索过程中探索的具体点。

![Two-Dimensional Contour Plot of the Test Objective Function](img/6d45ed0d01550999d609f23fb77ce685.png)

测试目标函数的二维等高线图

现在我们有了一个测试目标函数，让我们看看如何实现 RMSProp 优化算法。

### 基于 RMSProp 的梯度下降优化

我们可以将带有 RMSProp 的梯度下降应用于测试问题。

首先，我们需要一个函数来计算这个函数的导数。

*   f(x) = x^2
*   f'(x) = x * 2

x^2 的导数在每个维度上都是 x * 2。*导数()*函数实现如下。

```py
# derivative of objective function
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])
```

接下来，我们可以实现梯度下降优化。

首先，我们可以在问题的边界中选择一个随机点作为搜索的起点。

这假设我们有一个定义搜索范围的数组，每个维度有一行，第一列定义维度的最小值，第二列定义维度的最大值。

```py
...
# generate an initial point
solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
```

接下来，我们需要将每个维度的平方偏导数的衰减平均值初始化为 0.0。

```py
...
# list of the average square gradients for each variable
sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
```

然后，我们可以枚举由“ *n_iter* ”超参数定义的搜索优化算法的固定迭代次数。

```py
...
# run the gradient descent
for it in range(n_iter):
	...
```

第一步是使用*导数()*函数计算当前解的梯度。

```py
...
# calculate gradient
gradient = derivative(solution[0], solution[1])
```

然后我们需要计算偏导数的平方，并用“*ρ*”超参数更新平方偏导数的衰减平均值。

```py
...
# update the average of the squared partial derivatives
for i in range(gradient.shape[0]):
	# calculate the squared gradient
	sg = gradient[i]**2.0
	# update the moving average of the squared gradient
	sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0-rho))
```

然后，我们可以使用平方偏导数和梯度的移动平均值来计算下一个点的步长。

我们将一次处理一个变量，首先计算变量的步长，然后计算变量的新值。这些值在一个数组中建立，直到我们有一个全新的解决方案，该解决方案使用自定义步长从当前点开始以最陡的下降方向下降。

```py
...
# build a solution one variable at a time
new_solution = list()
for i in range(solution.shape[0]):
	# calculate the step size for this variable
	alpha = step_size / (1e-8 + sqrt(sq_grad_avg[i]))
	# calculate the new position in this variable
	value = solution[i] - alpha * gradient[i]
	# store this variable
	new_solution.append(value)
```

然后，可以使用*客观()*函数来评估这个新的解决方案，并且可以报告搜索的表现。

```py
...
# evaluate candidate point
solution = asarray(new_solution)
solution_eval = objective(solution[0], solution[1])
# report progress
print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
```

就这样。

我们可以将所有这些联系到一个名为 *rmsprop()* 的函数中，该函数采用目标函数和导数函数的名称，一个数组，该数组具有算法迭代总数和初始学习率的域和超参数值的边界，并返回最终解及其评估。

下面列出了完整的功能。

```py
# gradient descent algorithm with rmsprop
def rmsprop(objective, derivative, bounds, n_iter, step_size, rho):
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# list of the average square gradients for each variable
	sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent
	for it in range(n_iter):
		# calculate gradient
		gradient = derivative(solution[0], solution[1])
		# update the average of the squared partial derivatives
		for i in range(gradient.shape[0]):
			# calculate the squared gradient
			sg = gradient[i]**2.0
			# update the moving average of the squared gradient
			sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0-rho))
		# build a solution one variable at a time
		new_solution = list()
		for i in range(solution.shape[0]):
			# calculate the step size for this variable
			alpha = step_size / (1e-8 + sqrt(sq_grad_avg[i]))
			# calculate the new position in this variable
			value = solution[i] - alpha * gradient[i]
			# store this variable
			new_solution.append(value)
		# evaluate candidate point
		solution = asarray(new_solution)
		solution_eval = objective(solution[0], solution[1])
		# report progress
		print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
	return [solution, solution_eval]
```

**注**:为了可读性，我们特意使用了列表和命令式编码风格，而不是矢量化操作。请随意将该实现调整为使用 NumPy 数组的矢量化实现，以获得更好的表现。

然后，我们可以定义我们的超参数，并调用 *rmsprop()* 函数来优化我们的测试目标函数。

在这种情况下，我们将使用算法的 50 次迭代，初始学习率为 0.01，rho 超参数的值为 0.99，所有这些都是在一点点尝试和错误之后选择的。

```py
...
# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# define the step size
step_size = 0.01
# momentum for rmsprop
rho = 0.99
# perform the gradient descent search with rmsprop
best, score = rmsprop(objective, derivative, bounds, n_iter, step_size, rho)
print('Done!')
print('f(%s) = %f' % (best, score))
```

将所有这些结合在一起，下面列出了使用 RMSProp 进行梯度下降优化的完整示例。

```py
# gradient descent optimization with rmsprop for a two-dimensional test function
from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# derivative of objective function
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with rmsprop
def rmsprop(objective, derivative, bounds, n_iter, step_size, rho):
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# list of the average square gradients for each variable
	sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent
	for it in range(n_iter):
		# calculate gradient
		gradient = derivative(solution[0], solution[1])
		# update the average of the squared partial derivatives
		for i in range(gradient.shape[0]):
			# calculate the squared gradient
			sg = gradient[i]**2.0
			# update the moving average of the squared gradient
			sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0-rho))
		# build a solution one variable at a time
		new_solution = list()
		for i in range(solution.shape[0]):
			# calculate the step size for this variable
			alpha = step_size / (1e-8 + sqrt(sq_grad_avg[i]))
			# calculate the new position in this variable
			value = solution[i] - alpha * gradient[i]
			# store this variable
			new_solution.append(value)
		# evaluate candidate point
		solution = asarray(new_solution)
		solution_eval = objective(solution[0], solution[1])
		# report progress
		print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
	return [solution, solution_eval]

# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# define the step size
step_size = 0.01
# momentum for rmsprop
rho = 0.99
# perform the gradient descent search with rmsprop
best, score = rmsprop(objective, derivative, bounds, n_iter, step_size, rho)
print('Done!')
print('f(%s) = %f' % (best, score))
```

运行该示例将 RMSProp 优化算法应用于我们的测试问题，并报告算法每次迭代的搜索表现。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，在大约 33 次搜索迭代后，找到了接近最优的解，输入值接近 0.0 和 0.0，评估为 0.0。

```py
...
>30 f([-9.61030898e-14 3.19352553e-03]) = 0.00001
>31 f([-3.42767893e-14 2.71513758e-03]) = 0.00001
>32 f([-1.21143047e-14 2.30636623e-03]) = 0.00001
>33 f([-4.24204875e-15 1.95738936e-03]) = 0.00000
>34 f([-1.47154482e-15 1.65972553e-03]) = 0.00000
>35 f([-5.05629595e-16 1.40605727e-03]) = 0.00000
>36 f([-1.72064649e-16 1.19007691e-03]) = 0.00000
>37 f([-5.79813754e-17 1.00635204e-03]) = 0.00000
>38 f([-1.93445677e-17 8.50208253e-04]) = 0.00000
>39 f([-6.38906842e-18 7.17626999e-04]) = 0.00000
>40 f([-2.08860690e-18 6.05156738e-04]) = 0.00000
>41 f([-6.75689941e-19 5.09835645e-04]) = 0.00000
>42 f([-2.16291217e-19 4.29124484e-04]) = 0.00000
>43 f([-6.84948980e-20 3.60848338e-04]) = 0.00000
>44 f([-2.14551097e-20 3.03146089e-04]) = 0.00000
>45 f([-6.64629576e-21 2.54426642e-04]) = 0.00000
>46 f([-2.03575780e-21 2.13331041e-04]) = 0.00000
>47 f([-6.16437387e-22 1.78699710e-04]) = 0.00000
>48 f([-1.84495110e-22 1.49544152e-04]) = 0.00000
>49 f([-5.45667355e-23 1.25022522e-04]) = 0.00000
Done!
f([-5.45667355e-23 1.25022522e-04]) = 0.000000
```

### RMSProp 的可视化

我们可以在域的等高线图上绘制搜索进度。

这可以为算法迭代过程中的搜索进度提供直觉。

我们必须更新 rmsprop()函数，以维护搜索过程中找到的所有解决方案的列表，然后在搜索结束时返回该列表。

下面列出了带有这些更改的功能的更新版本。

```py
# gradient descent algorithm with rmsprop
def rmsprop(objective, derivative, bounds, n_iter, step_size, rho):
	# track all solutions
	solutions = list()
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# list of the average square gradients for each variable
	sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent
	for it in range(n_iter):
		# calculate gradient
		gradient = derivative(solution[0], solution[1])
		# update the average of the squared partial derivatives
		for i in range(gradient.shape[0]):
			# calculate the squared gradient
			sg = gradient[i]**2.0
			# update the moving average of the squared gradient
			sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0-rho))
		# build solution
		new_solution = list()
		for i in range(solution.shape[0]):
			# calculate the learning rate for this variable
			alpha = step_size / (1e-8 + sqrt(sq_grad_avg[i]))
			# calculate the new position in this variable
			value = solution[i] - alpha * gradient[i]
			new_solution.append(value)
		# store the new solution
		solution = asarray(new_solution)
		solutions.append(solution)
		# evaluate candidate point
		solution_eval = objective(solution[0], solution[1])
		# report progress
		print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
	return solutions
```

然后，我们可以像以前一样执行搜索，这次检索解决方案列表，而不是最佳最终解决方案。

```py
...
# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# define the step size
step_size = 0.01
# momentum for rmsprop
rho = 0.99
# perform the gradient descent search with rmsprop
solutions = rmsprop(objective, derivative, bounds, n_iter, step_size, rho)
```

然后，我们可以像以前一样创建目标函数的等高线图。

```py
...
# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
```

最后，我们可以将搜索过程中找到的每个解决方案绘制成由一条线连接的白点。

```py
...
# plot the sample as black circles
solutions = asarray(solutions)
pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
```

将所有这些结合起来，下面列出了对测试问题执行 RMSProp 优化并将结果绘制在等高线图上的完整示例。

```py
# example of plotting the rmsprop search on a contour plot of the test function
from math import sqrt
from numpy import asarray
from numpy import arange
from numpy.random import rand
from numpy.random import seed
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# derivative of objective function
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with rmsprop
def rmsprop(objective, derivative, bounds, n_iter, step_size, rho):
	# track all solutions
	solutions = list()
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# list of the average square gradients for each variable
	sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent
	for it in range(n_iter):
		# calculate gradient
		gradient = derivative(solution[0], solution[1])
		# update the average of the squared partial derivatives
		for i in range(gradient.shape[0]):
			# calculate the squared gradient
			sg = gradient[i]**2.0
			# update the moving average of the squared gradient
			sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0-rho))
		# build solution
		new_solution = list()
		for i in range(solution.shape[0]):
			# calculate the learning rate for this variable
			alpha = step_size / (1e-8 + sqrt(sq_grad_avg[i]))
			# calculate the new position in this variable
			value = solution[i] - alpha * gradient[i]
			new_solution.append(value)
		# store the new solution
		solution = asarray(new_solution)
		solutions.append(solution)
		# evaluate candidate point
		solution_eval = objective(solution[0], solution[1])
		# report progress
		print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
	return solutions

# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# define the step size
step_size = 0.01
# momentum for rmsprop
rho = 0.99
# perform the gradient descent search with rmsprop
solutions = rmsprop(objective, derivative, bounds, n_iter, step_size, rho)
# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# plot the sample as black circles
solutions = asarray(solutions)
pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
# show the plot
pyplot.show()
```

运行该示例会像以前一样执行搜索，只是在这种情况下，会创建目标函数的等高线图。

在这种情况下，我们可以看到，搜索过程中找到的每个解决方案都显示一个白点，从 optima 上方开始，逐渐靠近图中心的 optima。

![Contour Plot of the Test Objective Function With RMSProp Search Results Shown](img/be52d091a37897a7f655afca17b27ee0.png)

显示 RMSProp 搜索结果的测试目标函数等高线图

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   第 6e 讲， [rmsprop:用梯度最近大小的运行平均值来划分梯度](http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf)，机器学习的神经网络，杰弗里·辛顿。

### 书

*   [优化算法](https://amzn.to/39KZSQn)，2019。
*   [深度学习](https://amzn.to/3qSk3C2)，2016 年。

### 蜜蜂

*   num py . random . rand API。
*   num py . asar ray API。
*   [Matplotlib API](https://matplotlib.org/api/pyplot_api.html) 。

### 文章

*   [梯度下降，维基百科](https://en.wikipedia.org/wiki/Gradient_descent)。
*   [随机梯度下降，维基百科](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)。
*   [梯度下降优化算法概述](https://ruder.io/optimizing-gradient-descent/index.html)，2016。

## 摘要

在本教程中，您发现了如何使用 RMSProp 优化算法从零开始开发梯度下降。

具体来说，您了解到:

*   梯度下降是一种优化算法，它使用目标函数的梯度来导航搜索空间。
*   梯度下降可以更新，以使用一个衰减的偏导数平均值(称为 RMSProp)为每个输入变量使用自动自适应步长。
*   如何从零开始实现 RMSProp 优化算法，并将其应用于目标函数并评估结果。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。