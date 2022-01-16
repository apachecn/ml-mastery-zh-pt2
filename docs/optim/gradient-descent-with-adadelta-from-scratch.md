# 从零开始的 Adadelta 梯度下降

> 原文：<https://machinelearningmastery.com/gradient-descent-with-adadelta-from-scratch/>

最后更新于 2021 年 10 月 12 日

梯度下降是一种优化算法，它遵循目标函数的负梯度来定位函数的最小值。

梯度下降的一个限制是，它对每个输入变量使用相同的步长(学习率)。AdaGradn 和 RMSProp 是梯度下降的扩展，为目标函数的每个参数增加了自适应学习率。

**Adadelta** 可以被认为是梯度下降的进一步扩展，它建立在 AdaGrad 和 RMSProp 的基础上，并改变了自定义步长的计算，使得单位一致，进而不再需要初始学习率超参数。

在本教程中，您将发现如何从零开始使用阿达塔优化算法开发梯度下降。

完成本教程后，您将知道:

*   梯度下降是一种优化算法，它使用目标函数的梯度来导航搜索空间。
*   梯度下降可以更新，以使用一个衰减的偏导数平均值(称为 Adadelta)为每个输入变量使用自动自适应步长。
*   如何从零开始实现 Adadelta 优化算法，并将其应用于目标函数并评估结果。

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![Gradient Descent With Adadelta from Scratch](img/25e2e3f430bcfbe31f1135c4cc754c7c.png)

罗伯特·明克勒拍摄的《从零开始的梯度下降》照片，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  梯度下降
2.  Adadelta 算法
3.  Adadelta 梯度下降
    1.  二维测试问题
    2.  带自适应增量的梯度下降优化
    3.  阿达塔的可视化

## 梯度下降

[梯度下降](https://en.wikipedia.org/wiki/Gradient_descent)是一种优化算法。

它在技术上被称为一阶优化算法，因为它明确地利用了目标函数的一阶导数。

> 一阶方法依赖于梯度信息来帮助指导搜索最小值…

—第 69 页，[优化算法](https://amzn.to/39KZSQn)，2019。

[一阶导数](https://en.wikipedia.org/wiki/Derivative)，或简称为“*导数*，是目标函数在特定点的变化率或斜率，例如对于特定输入。

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

现在我们已经熟悉了梯度下降优化算法，让我们来看看 Adadelta。

## Adadelta 算法

Adadelta(或“ADADELTA”)是梯度下降优化算法的扩展。

该算法由[马修·泽勒](https://www.linkedin.com/in/mattzeiler/)在 2012 年发表的题为“ [ADADELTA:一种自适应学习率方法](https://arxiv.org/abs/1212.5701)的论文中进行了描述

Adadelta 旨在加速优化过程，例如减少达到最优所需的函数求值次数，或者提高优化算法的能力，例如产生更好的最终结果。

最好将其理解为 AdaGrad 和 RMSProp 算法的扩展。

AdaGrad 是梯度下降的扩展，它在每次更新时为目标函数的每个参数计算步长(学习率)。步长的计算方法是，首先对搜索过程中所见参数的偏导数求和，然后将初始步长超参数除以偏导数平方和的平方根。

AdaGrad 中一个参数的自定义步长计算如下:

*   cust _ step _ size(t+1)= step _ size/(1e-8+sqrt(s(t)))

其中 *cust_step_size(t+1)* 是搜索过程中给定点输入变量的计算步长， *step_size* 是初始步长， *sqrt()* 是平方根运算， *s(t)* 是迄今为止(包括当前迭代)搜索过程中看到的输入变量的平方偏导数之和。

RMSProp 可以被认为是 AdaGrad 的扩展，因为它在计算每个参数的步长时使用偏导数的衰减平均值或移动平均值，而不是总和。这是通过增加一个新的超参数“*ρ*”来实现的，它就像偏导数的动量。

一个参数的衰减移动平均平方偏导数的计算如下:

*   s(t+1)=(s(t)*ρ)+(f'(x(t))^2 *(1.0-ρ))

其中 *s(t+1)* 是算法当前迭代的一个参数的均方偏导数， *s(t)* 是上一次迭代的衰减移动平均平方偏导数， *f'(x(t))^2* 是当前参数的平方偏导数，rho 是一个超参数，通常具有类似动量的 0.9 的值。

Adadelta 是 RMSProp 的进一步扩展，旨在提高算法的收敛性，并消除对手动指定初始学习率的需求。

> 本文提出的想法源自 ADAGRAD，目的是改进该方法的两个主要缺点:1)在整个训练过程中学习率不断衰减，以及 2)需要手动选择全局学习率。

——[ADADELTA:一种自适应学习率方法](https://arxiv.org/abs/1212.5701)，2012。

平方偏导数的衰减移动平均值是为每个参数计算的，就像 RMSProp 一样。关键的区别在于计算参数的步长时，使用的是增量的衰减平均值或参数的变化。

选择分子是为了确保计算的两个部分具有相同的单位。

> 在独立导出 RMSProp 更新后，作者注意到更新方程中梯度下降、动量和阿达格勒的单位不匹配。为了解决这个问题，他们使用平方更新的指数衰减平均值

—第 78-79 页，[优化算法](https://amzn.to/39KZSQn)，2019。

首先，自定义步长计算为δ变化的衰减移动平均值的平方根除以平方偏导数的衰减移动平均值的平方根。

*   cust _ step _ size(t+1)=(EP+sqrt(δ(t)))/(EP+sqrt(s(t)))

其中 *cust_step_size(t+1)* 是给定更新的参数的自定义步长， *ep* 是添加到分子和分母以避免被零除误差的超参数， *delta(t)* 是参数平方变化的衰减移动平均值(在上一次迭代中计算)，而 *s(t)* 是平方偏导数的衰减移动平均值(在当前迭代中计算)。

*ep* 超参数设置为小值，如 1e-3 或 1e-8。除了避免被零除的误差，当衰减的移动平均平方变化和衰减的移动平均平方梯度为零时，它也有助于算法的第一步。

接下来，参数的变化被计算为自定义步长乘以偏导数

*   change(t+1)= cust _ step _ size(t+1)* f '(x(t))

接下来，更新参数平方变化的衰减平均值。

*   δ(t+1)=(δ(t)*ρ)+(change(t+1)^2 *(1.0-ρ))

其中*δ(t+1)*是要在下一次迭代中使用的变量变化的衰减平均值，*变化(t+1)* 是在之前的步骤中计算的，*ρ*是一个类似动量的超参数，其值类似于 0.9。

最后，使用该变化计算变量的新值。

*   x(t+1)= x(t)–变化(t+1)

然后对目标函数的每个变量重复该过程，然后重复整个过程以在搜索空间中导航固定次数的算法迭代。

现在我们已经熟悉了 Adadelta 算法，让我们探索如何实现它并评估它的表现。

## Adadelta 梯度下降

在本节中，我们将探索如何用 Adadelta 实现梯度下降优化算法。

### 二维测试问题

首先，让我们定义一个优化函数。

我们将使用一个简单的二维函数，它对每个维度的输入进行平方，并定义从-1.0 到 1.0 的有效输入范围。

下面的 objective()函数实现了这个功能

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

现在我们有了一个测试目标函数，让我们看看如何实现 Adadelta 优化算法。

### 带自适应增量的梯度下降优化

我们可以将带有 Adadelta 的梯度下降应用于测试问题。

首先，我们需要一个函数来计算这个函数的导数。

*   f(x) = x^2
*   f'(x) = x * 2

x^2 的导数在每个维度上都是 x * 2。导数()函数实现如下。

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

接下来，我们需要将每个维度的平方偏导数和平方变化的衰减平均值初始化为 0.0。

```py
...
# list of the average square gradients for each variable
sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
# list of the average parameter updates
sq_para_avg = [0.0 for _ in range(bounds.shape[0])]
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

然后我们需要计算偏导数的平方，并用“*ρ*”超参数更新平方偏导数的衰减移动平均值。

```py
...
# update the average of the squared partial derivatives
for i in range(gradient.shape[0]):
	# calculate the squared gradient
	sg = gradient[i]**2.0
	# update the moving average of the squared gradient
	sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0-rho))
```

然后，我们可以使用平方偏导数和梯度的衰减移动平均值来计算下一个点的步长。我们将一次做一个变量。

```py
...
# build solution
new_solution = list()
for i in range(solution.shape[0]):
	...
```

首先，我们将使用平方变化和平方偏导数的衰减移动平均以及“ep”超参数来计算这个变量在这次迭代中的自定义步长。

```py
...
# calculate the step size for this variable
alpha = (ep + sqrt(sq_para_avg[i])) / (ep + sqrt(sq_grad_avg[i]))
```

接下来，我们可以使用自定义步长和偏导数来计算变量的变化。

```py
...
# calculate the change
change = alpha * gradient[i]
```

然后，我们可以使用变化来更新平方变化的衰减移动平均，使用“*ρ*”超参数。

```py
...
# update the moving average of squared parameter changes
sq_para_avg[i] = (sq_para_avg[i] * rho) + (change**2.0 * (1.0-rho))
```

最后，我们可以在继续下一个变量之前更改变量并存储结果。

```py
...
# calculate the new position in this variable
value = solution[i] - change
# store this variable
new_solution.append(value)
```

然后可以使用 objective()函数评估这个新的解决方案，并报告搜索的表现。

```py
...
# evaluate candidate point
solution = asarray(new_solution)
solution_eval = objective(solution[0], solution[1])
# report progress
print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
```

就这样。

我们可以将所有这些联系到一个名为 *adadelta()* 的函数中，该函数采用目标函数和导数函数的名称，一个数组，该数组具有算法迭代总数和*ρ*的定义域和超参数值的边界，并返回最终解及其求值结果。

*ep* 超参数也可以作为一个参数，尽管它有一个合理的默认值 1e-3。

下面列出了完整的功能。

```py
# gradient descent algorithm with adadelta
def adadelta(objective, derivative, bounds, n_iter, rho, ep=1e-3):
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# list of the average square gradients for each variable
	sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
	# list of the average parameter updates
	sq_para_avg = [0.0 for _ in range(bounds.shape[0])]
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
			alpha = (ep + sqrt(sq_para_avg[i])) / (ep + sqrt(sq_grad_avg[i]))
			# calculate the change
			change = alpha * gradient[i]
			# update the moving average of squared parameter changes
			sq_para_avg[i] = (sq_para_avg[i] * rho) + (change**2.0 * (1.0-rho))
			# calculate the new position in this variable
			value = solution[i] - change
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

然后我们可以定义我们的超参数，并调用 *adadelta()* 函数来优化我们的测试目标函数。

在这种情况下，我们将使用该算法的 120 次迭代和 0.99 的 rho 超参数值，该值是在经过一点反复试验后选择的。

```py
...
# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 120
# momentum for adadelta
rho = 0.99
# perform the gradient descent search with adadelta
best, score = adadelta(objective, derivative, bounds, n_iter, rho)
print('Done!')
print('f(%s) = %f' % (best, score))
```

将所有这些联系在一起，下面列出了使用 Adadelta 进行梯度下降优化的完整示例。

```py
# gradient descent optimization with adadelta for a two-dimensional test function
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

# gradient descent algorithm with adadelta
def adadelta(objective, derivative, bounds, n_iter, rho, ep=1e-3):
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# list of the average square gradients for each variable
	sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
	# list of the average parameter updates
	sq_para_avg = [0.0 for _ in range(bounds.shape[0])]
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
			alpha = (ep + sqrt(sq_para_avg[i])) / (ep + sqrt(sq_grad_avg[i]))
			# calculate the change
			change = alpha * gradient[i]
			# update the moving average of squared parameter changes
			sq_para_avg[i] = (sq_para_avg[i] * rho) + (change**2.0 * (1.0-rho))
			# calculate the new position in this variable
			value = solution[i] - change
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
n_iter = 120
# momentum for adadelta
rho = 0.99
# perform the gradient descent search with adadelta
best, score = adadelta(objective, derivative, bounds, n_iter, rho)
print('Done!')
print('f(%s) = %f' % (best, score))
```

运行该示例将 Adadelta 优化算法应用于我们的测试问题，并报告算法每次迭代的搜索表现。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，在大约 105 次搜索迭代后，找到了接近最优的解，输入值接近 0.0 和 0.0，评估为 0.0。

```py
...
>100 f([-1.45142626e-07 2.71163181e-03]) = 0.00001
>101 f([-1.24898699e-07 2.56875692e-03]) = 0.00001
>102 f([-1.07454197e-07 2.43328237e-03]) = 0.00001
>103 f([-9.24253035e-08 2.30483111e-03]) = 0.00001
>104 f([-7.94803792e-08 2.18304501e-03]) = 0.00000
>105 f([-6.83329263e-08 2.06758392e-03]) = 0.00000
>106 f([-5.87354975e-08 1.95812477e-03]) = 0.00000
>107 f([-5.04744185e-08 1.85436071e-03]) = 0.00000
>108 f([-4.33652179e-08 1.75600036e-03]) = 0.00000
>109 f([-3.72486699e-08 1.66276699e-03]) = 0.00000
>110 f([-3.19873691e-08 1.57439783e-03]) = 0.00000
>111 f([-2.74627662e-08 1.49064334e-03]) = 0.00000
>112 f([-2.3572602e-08 1.4112666e-03]) = 0.00000
>113 f([-2.02286891e-08 1.33604264e-03]) = 0.00000
>114 f([-1.73549914e-08 1.26475787e-03]) = 0.00000
>115 f([-1.48859650e-08 1.19720951e-03]) = 0.00000
>116 f([-1.27651224e-08 1.13320504e-03]) = 0.00000
>117 f([-1.09437923e-08 1.07256172e-03]) = 0.00000
>118 f([-9.38004754e-09 1.01510604e-03]) = 0.00000
>119 f([-8.03777865e-09 9.60673346e-04]) = 0.00000
Done!
f([-8.03777865e-09 9.60673346e-04]) = 0.000001
```

### 阿达塔的可视化

我们可以在域的等高线图上绘制阿达塔搜索的进度。

这可以为算法迭代过程中的搜索进度提供直觉。

我们必须更新 *adadelta()* 函数，以维护搜索过程中找到的所有解决方案的列表，然后在搜索结束时返回该列表。

下面列出了带有这些更改的功能的更新版本。

```py
# gradient descent algorithm with adadelta
def adadelta(objective, derivative, bounds, n_iter, rho, ep=1e-3):
	# track all solutions
	solutions = list()
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# list of the average square gradients for each variable
	sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
	# list of the average parameter updates
	sq_para_avg = [0.0 for _ in range(bounds.shape[0])]
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
			# calculate the step size for this variable
			alpha = (ep + sqrt(sq_para_avg[i])) / (ep + sqrt(sq_grad_avg[i]))
			# calculate the change
			change = alpha * gradient[i]
			# update the moving average of squared parameter changes
			sq_para_avg[i] = (sq_para_avg[i] * rho) + (change**2.0 * (1.0-rho))
			# calculate the new position in this variable
			value = solution[i] - change
			# store this variable
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
n_iter = 120
# rho for adadelta
rho = 0.99
# perform the gradient descent search with adadelta
solutions = adadelta(objective, derivative, bounds, n_iter, rho)
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

将所有这些结合起来，下面列出了对测试问题执行 Adadelta 优化并将结果绘制在等高线图上的完整示例。

```py
# example of plotting the adadelta search on a contour plot of the test function
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

# gradient descent algorithm with adadelta
def adadelta(objective, derivative, bounds, n_iter, rho, ep=1e-3):
	# track all solutions
	solutions = list()
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# list of the average square gradients for each variable
	sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
	# list of the average parameter updates
	sq_para_avg = [0.0 for _ in range(bounds.shape[0])]
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
			# calculate the step size for this variable
			alpha = (ep + sqrt(sq_para_avg[i])) / (ep + sqrt(sq_grad_avg[i]))
			# calculate the change
			change = alpha * gradient[i]
			# update the moving average of squared parameter changes
			sq_para_avg[i] = (sq_para_avg[i] * rho) + (change**2.0 * (1.0-rho))
			# calculate the new position in this variable
			value = solution[i] - change
			# store this variable
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
n_iter = 120
# rho for adadelta
rho = 0.99
# perform the gradient descent search with adadelta
solutions = adadelta(objective, derivative, bounds, n_iter, rho)
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

![Contour Plot of the Test Objective Function With Adadelta Search Results Shown](img/21a1add8870b8aec4a5466adc67dca3e.png)

显示 Adadelta 搜索结果的测试目标函数等高线图

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [ADADELTA:一种自适应学习率方法](https://arxiv.org/abs/1212.5701)，2012。

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

在本教程中，您发现了如何从零开始使用 Adadelta 优化算法开发梯度下降。

具体来说，您了解到:

*   梯度下降是一种优化算法，它使用目标函数的梯度来导航搜索空间。
*   梯度下降可以更新，以使用一个衰减的偏导数平均值(称为 Adadelta)为每个输入变量使用自动自适应步长。
*   如何从零开始实现 Adadelta 优化算法，并将其应用于目标函数并评估结果。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。