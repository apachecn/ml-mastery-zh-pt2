# 如何手动优化神经网络模型

> 原文：<https://machinelearningmastery.com/manually-optimize-neural-networks/>

最后更新于 2021 年 10 月 12 日

**深度学习神经网络**模型使用随机梯度下降优化算法拟合训练数据。

使用误差算法的反向传播来更新模型的权重。优化和权重更新算法的组合是精心选择的，并且是已知的最有效的拟合神经网络的方法。

然而，可以使用替代优化算法将神经网络模型拟合到训练数据集。这可能是一个有用的练习，以了解更多关于神经网络如何工作以及优化在应用机器学习中的核心性质。对于具有非常规模型结构和不可微传递函数的神经网络，也可能需要它。

在本教程中，您将发现如何手动优化神经网络模型的权重。

完成本教程后，您将知道:

*   如何从零开始开发神经网络模型的正向推理通路？
*   如何为二进制分类优化感知器模型的权重。
*   如何使用随机爬山优化多层感知器模型的权重。

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![How to Manually Optimize Neural Network Models](img/b8af554e30c694dde22676dd077b7ce8.png)

如何手动优化神经网络模型
图片由[土地管理局](https://www.flickr.com/photos/mypubliclands/26153922644/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  优化神经网络
2.  优化感知器模型
3.  优化多层感知器

## 优化神经网络

[深度学习](https://machinelearningmastery.com/what-is-deep-learning/)或神经网络是机器学习的一种灵活类型。

它们是由节点和层组成的模型，灵感来自大脑的结构和功能。神经网络模型的工作原理是通过一个或多个层传播给定的输入向量，以产生可以解释为分类或回归预测建模的数字输出。

通过反复将模型暴露于输入和输出的示例中，并调整权重以最小化模型输出与预期输出相比的误差，来训练模型。这被称为随机梯度下降优化算法。使用微积分中的特定规则来调整模型的权重，该规则将误差按比例分配给网络中的每个权重。这被称为[反向传播算法](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)。

使用反向传播进行权重更新的随机梯度下降优化算法是训练神经网络模型的最佳方法。然而，这不是训练神经网络的唯一方法。

可以使用任意优化算法来训练神经网络模型。

也就是说，我们可以定义一个神经网络模型架构，并使用给定的优化算法来为模型找到一组权重，从而使预测误差最小或分类准确率最大。

平均而言，使用交替优化算法的效率低于使用带反向传播的随机梯度下降算法。然而，在某些特定情况下，例如非标准网络架构或非差分传输功能，它可能更有效。

它也可以是一个有趣的练习，展示优化在训练机器学习算法，特别是神经网络中的核心性质。

接下来，让我们探索如何使用随机爬山训练一个简单的单节点神经网络，称为感知器模型。

## 优化感知器模型

[感知器算法](https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/)是最简单的人工神经网络。

它是单个神经元的模型，可用于两类分类问题，并为以后开发更大的网络奠定了基础。

在本节中，我们将优化感知器神经网络模型的权重。

首先，让我们定义一个合成的二分类问题，我们可以将其作为优化模型的重点。

我们可以使用 [make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)定义一个包含 1000 行和 5 个输入变量的二分类问题。

下面的示例创建数据集并总结数据的形状。

```py
# define a binary classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# summarize the shape of the dataset
print(X.shape, y.shape)
```

运行该示例会打印出创建的数据集的形状，这证实了我们的预期。

```py
(1000, 5) (1000,)
```

接下来，我们需要定义一个感知器模型。

感知器模型有一个节点，该节点对数据集中的每一列都有一个输入权重。

每个输入乘以相应的权重，得到加权和，然后加上偏差权重，就像回归模型中的截距系数一样。这个加权和称为激活。最后，激活被解释并用于预测类别标签，1 表示正激活，0 表示负激活。

在优化模型权重之前，我们必须开发模型，并对其工作方式充满信心。

让我们从定义一个解释模型激活的函数开始。

这被称为激活函数，或传递函数；后一个名字更传统，也是我的偏好。

下面的 *transfer()* 函数获取模型的激活，并返回一个类标签，类=1 表示正激活或零激活，类=0 表示负激活。这被称为阶跃传递函数。

```py
# transfer function
def transfer(activation):
	if activation >= 0.0:
		return 1
	return 0
```

接下来，我们可以开发一个函数，为数据集的给定输入数据行计算模型的激活。

该函数将获取该行数据和模型的权重，并计算输入加上偏差权重的加权和。下面的*激活()*功能实现了这一点。

**注**:我们有意使用简单的 Python 列表和命令式编程风格，而不是 NumPy 数组或列表压缩，以使代码对 Python 初学者来说更具可读性。请随意优化它，并在下面的评论中发布您的代码。

```py
# activation function
def activate(row, weights):
	# add the bias, the last weight
	activation = weights[-1]
	# add the weighted input
	for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation
```

接下来，我们可以一起使用 *activate()* 和 *transfer()* 函数来为给定的数据行生成预测。下面的 *predict_row()* 函数实现了这一点。

```py
# use model weights to predict 0 or 1 for a given row of data
def predict_row(row, weights):
	# activate for input
	activation = activate(row, weights)
	# transfer for activation
	return transfer(activation)
```

接下来，我们可以为给定数据集中的每一行调用 *predict_row()* 函数。下面的*预测 _ 数据集()*函数实现了这一点。

同样，为了可读性，我们有意使用简单的命令式编码风格，而不是列表压缩。

```py
# use model weights to generate predictions for a dataset of rows
def predict_dataset(X, weights):
	yhats = list()
	for row in X:
		yhat = predict_row(row, weights)
		yhats.append(yhat)
	return yhats
```

最后，我们可以使用该模型对我们的合成数据集进行预测，以确认它都工作正常。

我们可以使用 [rand()函数](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html)生成一组随机的模型权重。

回想一下，我们需要为每个输入(这个数据集中的五个输入)加上一个额外的偏置权重。

```py
...
# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# determine the number of weights
n_weights = X.shape[1] + 1
# generate random weights
weights = rand(n_weights)
```

然后，我们可以使用数据集的这些权重来进行预测。

```py
...
# generate predictions for dataset
yhat = predict_dataset(X, weights)
```

我们可以评估这些预测的分类准确性。

```py
...
# calculate accuracy
score = accuracy_score(y, yhat)
print(score)
```

就这样。

我们可以将所有这些联系在一起，并演示我们简单的用于分类的感知器模型。下面列出了完整的示例。

```py
# simple perceptron model for binary classification
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# transfer function
def transfer(activation):
	if activation >= 0.0:
		return 1
	return 0

# activation function
def activate(row, weights):
	# add the bias, the last weight
	activation = weights[-1]
	# add the weighted input
	for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation

# use model weights to predict 0 or 1 for a given row of data
def predict_row(row, weights):
	# activate for input
	activation = activate(row, weights)
	# transfer for activation
	return transfer(activation)

# use model weights to generate predictions for a dataset of rows
def predict_dataset(X, weights):
	yhats = list()
	for row in X:
		yhat = predict_row(row, weights)
		yhats.append(yhat)
	return yhats

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# determine the number of weights
n_weights = X.shape[1] + 1
# generate random weights
weights = rand(n_weights)
# generate predictions for dataset
yhat = predict_dataset(X, weights)
# calculate accuracy
score = accuracy_score(y, yhat)
print(score)
```

运行示例会为训练数据集中的每个示例生成预测，然后打印预测的分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在给定一组随机权重和每个类中具有相同数量示例的数据集的情况下，我们期望大约 50%的准确性，这大约是我们在本例中看到的。

```py
0.548
```

我们现在可以优化数据集的权重，以在该数据集上获得良好的准确率。

首先，我们需要将数据集拆分成[训练和测试集](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/)。重要的是保留一些未用于优化模型的数据，以便我们可以在用于对新数据进行预测时对模型的表现进行合理的估计。

我们将使用 67%的数据进行训练，剩下的 33%作为测试集来评估模型的表现。

```py
...
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

接下来，我们可以开发一个随机爬山算法。

优化算法需要一个目标函数来优化。它必须采用一组权重，并返回一个对应于更好模型的最小化或最大化分数。

在这种情况下，我们将使用给定的一组权重来评估模型的准确性，并返回分类准确性，分类准确性必须最大化。

给定数据集和一组权重，下面的 *objective()* 函数实现了这一点，并返回模型的准确率

```py
# objective function
def objective(X, y, weights):
	# generate predictions for dataset
	yhat = predict_dataset(X, weights)
	# calculate accuracy
	score = accuracy_score(y, yhat)
	return score
```

接下来，我们可以定义[随机爬山算法](https://machinelearningmastery.com/stochastic-hill-climbing-in-python-from-scratch/)。

该算法将需要一个初始解(例如，随机权重)，并将反复不断地对解进行小的改变，并检查它是否会产生一个表现更好的模型。当前解的变化量由*步长*超参数控制。该过程将持续固定次数的迭代，也作为超参数提供。

下面的*爬山()*函数实现了这一点，将数据集、目标函数、初始解和超参数作为参数，并返回找到的最佳权重集和估计的表现。

```py
# hill climbing local search algorithm
def hillclimbing(X, y, objective, solution, n_iter, step_size):
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = solution + randn(len(solution)) * step_size
		# evaluate candidate point
		candidte_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidte_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
			print('>%d %.5f' % (i, solution_eval))
	return [solution, solution_eval]
```

然后我们可以调用这个函数，传入一组权重作为初始解，将训练数据集作为数据集来优化模型。

```py
...
# define the total iterations
n_iter = 1000
# define the maximum step size
step_size = 0.05
# determine the number of weights
n_weights = X.shape[1] + 1
# define the initial solution
solution = rand(n_weights)
# perform the hill climbing search
weights, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
print('Done!')
print('f(%s) = %f' % (weights, score))
```

最后，我们可以在测试数据集上评估最佳模型并报告表现。

```py
...
# generate predictions for the test dataset
yhat = predict_dataset(X_test, weights)
# calculate accuracy
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %.5f' % (score * 100))
```

将这些联系在一起，下面列出了在合成二进制优化数据集上优化感知器模型权重的完整示例。

```py
# hill climbing to optimize weights of a perceptron model for classification
from numpy import asarray
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# transfer function
def transfer(activation):
	if activation >= 0.0:
		return 1
	return 0

# activation function
def activate(row, weights):
	# add the bias, the last weight
	activation = weights[-1]
	# add the weighted input
	for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation

# # use model weights to predict 0 or 1 for a given row of data
def predict_row(row, weights):
	# activate for input
	activation = activate(row, weights)
	# transfer for activation
	return transfer(activation)

# use model weights to generate predictions for a dataset of rows
def predict_dataset(X, weights):
	yhats = list()
	for row in X:
		yhat = predict_row(row, weights)
		yhats.append(yhat)
	return yhats

# objective function
def objective(X, y, weights):
	# generate predictions for dataset
	yhat = predict_dataset(X, weights)
	# calculate accuracy
	score = accuracy_score(y, yhat)
	return score

# hill climbing local search algorithm
def hillclimbing(X, y, objective, solution, n_iter, step_size):
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = solution + randn(len(solution)) * step_size
		# evaluate candidate point
		candidte_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidte_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
			print('>%d %.5f' % (i, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# define the total iterations
n_iter = 1000
# define the maximum step size
step_size = 0.05
# determine the number of weights
n_weights = X.shape[1] + 1
# define the initial solution
solution = rand(n_weights)
# perform the hill climbing search
weights, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
print('Done!')
print('f(%s) = %f' % (weights, score))
# generate predictions for the test dataset
yhat = predict_dataset(X_test, weights)
# calculate accuracy
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %.5f' % (score * 100))
```

每次对模型进行改进时，运行该示例都会报告迭代次数和分类准确率。

在搜索结束时，报告最佳权重集在训练数据集上的表现，并计算和报告相同模型在测试数据集上的表现。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到优化算法找到了一组权重，在训练数据集上获得了大约 88.5%的准确率，在测试数据集上获得了大约 81.8%的准确率。

```py
...
>111 0.88060
>119 0.88060
>126 0.88209
>134 0.88209
>205 0.88209
>262 0.88209
>280 0.88209
>293 0.88209
>297 0.88209
>336 0.88209
>373 0.88209
>437 0.88358
>463 0.88507
>630 0.88507
>701 0.88507
Done!
f([ 0.0097317 0.13818088 1.17634326 -0.04296336 0.00485813 -0.14767616]) = 0.885075
Test Accuracy: 81.81818
```

现在我们已经熟悉了如何手动优化感知器模型的权重，让我们看看如何扩展示例来优化多层感知器(MLP)模型的权重。

## 优化多层感知器

多层感知器(MLP)模型是具有一层或多层的神经网络，其中每层具有一个或多个节点。

它是感知器模型的扩展，可能是应用最广泛的神经网络(深度学习)模型。

在本节中，我们将在上一节中所学的基础上优化 MLP 模型的权重，每层具有任意数量的层和节点。

首先，我们将开发模型并用随机权重进行测试，然后使用随机爬山来优化模型权重。

当使用 MLPs 进行二进制分类时，通常使用 sigmoid 传递函数(也称为逻辑函数)来代替感知器中使用的阶跃传递函数。

该函数输出 0-1 之间的实数值，表示[二项式概率分布](https://machinelearningmastery.com/discrete-probability-distributions-for-machine-learning/)，例如一个例子属于类=1 的概率。下面的*转移()*功能实现了这一点。

```py
# transfer function
def transfer(activation):
	# sigmoid transfer function
	return 1.0 / (1.0 + exp(-activation))
```

我们可以使用上一节中相同的*激活()*功能。这里，我们将使用它来计算给定层中每个节点的激活。

*predict_row()* 函数必须用更复杂的版本替换。

该函数获取一行数据和网络，并返回网络的输出。

我们将把我们的网络定义为一系列列表。每一层都是一个节点列表，每个节点都是一个权重列表或数组。

为了计算网络的预测，我们简单地枚举层，然后枚举节点，然后计算每个节点的激活和传输输出。在这种情况下，我们将对网络中的所有节点使用相同的传递函数，尽管并非必须如此。

对于具有多个层的网络，前一层的输出被用作下一层中每个节点的输入。然后返回网络中最后一层的输出。

下面的 *predict_row()* 函数实现了这一点。

```py
# activation function for a network
def predict_row(row, network):
	inputs = row
	# enumerate the layers in the network from input to output
	for layer in network:
		new_inputs = list()
		# enumerate nodes in the layer
		for node in layer:
			# activate the node
			activation = activate(inputs, node)
			# transfer activation
			output = transfer(activation)
			# store output
			new_inputs.append(output)
		# output from this layer is input to the next layer
		inputs = new_inputs
	return inputs[0]
```

差不多了。

最后，我们需要定义一个要使用的网络。

例如，我们可以定义一个具有单个隐藏层和单个节点的 MLP，如下所示:

```py
...
# create a one node network
node = rand(n_inputs + 1)
layer = [node]
network = [layer]
```

这实际上是一个感知器，虽然有一个乙状元传递函数。相当无聊。

让我们定义一个具有一个隐藏层和一个输出层的 MLP。第一个隐藏层将有 10 个节点，每个节点将从数据集获取输入模式(例如 5 个输入)。输出层将具有单个节点，该节点从第一隐藏层的输出中获取输入，然后输出预测。

```py
...
# one hidden layer and an output layer
n_hidden = 10
hidden1 = [rand(n_inputs + 1) for _ in range(n_hidden)]
output1 = [rand(n_hidden + 1)]
network = [hidden1, output1]
```

然后，我们可以使用该模型对数据集进行预测。

```py
...
# generate predictions for dataset
yhat = predict_dataset(X, network)
```

在计算分类准确率之前，我们必须将预测四舍五入到类别标签 0 和 1。

```py
...
# round the predictions
yhat = [round(y) for y in yhat]
# calculate accuracy
score = accuracy_score(y, yhat)
print(score)
```

将所有这些联系在一起，下面列出了在我们的合成二进制类别数据集上用随机初始权重评估 MLP 的完整示例。

```py
# develop an mlp model for classification
from math import exp
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# transfer function
def transfer(activation):
	# sigmoid transfer function
	return 1.0 / (1.0 + exp(-activation))

# activation function
def activate(row, weights):
	# add the bias, the last weight
	activation = weights[-1]
	# add the weighted input
	for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation

# activation function for a network
def predict_row(row, network):
	inputs = row
	# enumerate the layers in the network from input to output
	for layer in network:
		new_inputs = list()
		# enumerate nodes in the layer
		for node in layer:
			# activate the node
			activation = activate(inputs, node)
			# transfer activation
			output = transfer(activation)
			# store output
			new_inputs.append(output)
		# output from this layer is input to the next layer
		inputs = new_inputs
	return inputs[0]

# use model weights to generate predictions for a dataset of rows
def predict_dataset(X, network):
	yhats = list()
	for row in X:
		yhat = predict_row(row, network)
		yhats.append(yhat)
	return yhats

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# determine the number of inputs
n_inputs = X.shape[1]
# one hidden layer and an output layer
n_hidden = 10
hidden1 = [rand(n_inputs + 1) for _ in range(n_hidden)]
output1 = [rand(n_hidden + 1)]
network = [hidden1, output1]
# generate predictions for dataset
yhat = predict_dataset(X, network)
# round the predictions
yhat = [round(y) for y in yhat]
# calculate accuracy
score = accuracy_score(y, yhat)
print(score)
```

运行示例会为训练数据集中的每个示例生成一个预测，然后打印预测的分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

同样，在给定一组随机权重和每个类中具有相同数量示例的数据集的情况下，我们期望大约 50%的准确性，这大约是我们在本例中看到的。

```py
0.499
```

接下来，我们可以将[随机爬山算法](https://machinelearningmastery.com/stochastic-hill-climbing-in-python-from-scratch/)应用于数据集。

这与将爬山应用于感知器模型非常相似，只是在这种情况下，一个步骤需要修改网络中的所有权重。

为此，我们将开发一个新的函数，创建一个网络副本，并在创建副本时变异网络中的每个权重。

下面的*步骤()*功能实现了这一点。

```py
# take a step in the search space
def step(network, step_size):
	new_net = list()
	# enumerate layers in the network
	for layer in network:
		new_layer = list()
		# enumerate nodes in this layer
		for node in layer:
			# mutate the node
			new_node = node.copy() + randn(len(node)) * step_size
			# store node in layer
			new_layer.append(new_node)
		# store layer in network
		new_net.append(new_layer)
	return new_net
```

修改网络中的所有权重是激进的。

搜索空间中一个不太激进的步骤可能是对模型中的权重子集做一个小的改变，也许由一个超参数控制。这是作为扩展留下的。

然后我们可以从爬山()函数中调用这个新的 *step()* 函数。

```py
# hill climbing local search algorithm
def hillclimbing(X, y, objective, solution, n_iter, step_size):
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = step(solution, step_size)
		# evaluate candidate point
		candidte_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidte_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
			print('>%d %f' % (i, solution_eval))
	return [solution, solution_eval]
```

将这些联系在一起，下面列出了应用随机爬山来优化二分类的 MLP 模型权重的完整示例。

```py
# stochastic hill climbing to optimize a multilayer perceptron for classification
from math import exp
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# transfer function
def transfer(activation):
	# sigmoid transfer function
	return 1.0 / (1.0 + exp(-activation))

# activation function
def activate(row, weights):
	# add the bias, the last weight
	activation = weights[-1]
	# add the weighted input
	for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation

# activation function for a network
def predict_row(row, network):
	inputs = row
	# enumerate the layers in the network from input to output
	for layer in network:
		new_inputs = list()
		# enumerate nodes in the layer
		for node in layer:
			# activate the node
			activation = activate(inputs, node)
			# transfer activation
			output = transfer(activation)
			# store output
			new_inputs.append(output)
		# output from this layer is input to the next layer
		inputs = new_inputs
	return inputs[0]

# use model weights to generate predictions for a dataset of rows
def predict_dataset(X, network):
	yhats = list()
	for row in X:
		yhat = predict_row(row, network)
		yhats.append(yhat)
	return yhats

# objective function
def objective(X, y, network):
	# generate predictions for dataset
	yhat = predict_dataset(X, network)
	# round the predictions
	yhat = [round(y) for y in yhat]
	# calculate accuracy
	score = accuracy_score(y, yhat)
	return score

# take a step in the search space
def step(network, step_size):
	new_net = list()
	# enumerate layers in the network
	for layer in network:
		new_layer = list()
		# enumerate nodes in this layer
		for node in layer:
			# mutate the node
			new_node = node.copy() + randn(len(node)) * step_size
			# store node in layer
			new_layer.append(new_node)
		# store layer in network
		new_net.append(new_layer)
	return new_net

# hill climbing local search algorithm
def hillclimbing(X, y, objective, solution, n_iter, step_size):
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = step(solution, step_size)
		# evaluate candidate point
		candidte_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidte_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
			print('>%d %f' % (i, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# define the total iterations
n_iter = 1000
# define the maximum step size
step_size = 0.1
# determine the number of inputs
n_inputs = X.shape[1]
# one hidden layer and an output layer
n_hidden = 10
hidden1 = [rand(n_inputs + 1) for _ in range(n_hidden)]
output1 = [rand(n_hidden + 1)]
network = [hidden1, output1]
# perform the hill climbing search
network, score = hillclimbing(X_train, y_train, objective, network, n_iter, step_size)
print('Done!')
print('Best: %f' % (score))
# generate predictions for the test dataset
yhat = predict_dataset(X_test, network)
# round the predictions
yhat = [round(y) for y in yhat]
# calculate accuracy
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %.5f' % (score * 100))
```

每次对模型进行改进时，运行该示例都会报告迭代次数和分类准确率。

在搜索结束时，报告最佳权重集在训练数据集上的表现，并计算和报告相同模型在测试数据集上的表现。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到优化算法找到了一组权重，在训练数据集上获得了大约 87.3%的准确率，在测试数据集上获得了大约 85.1%的准确率。

```py
...
>55 0.755224
>56 0.765672
>59 0.794030
>66 0.805970
>77 0.835821
>120 0.838806
>165 0.840299
>188 0.841791
>218 0.846269
>232 0.852239
>237 0.852239
>239 0.855224
>292 0.867164
>368 0.868657
>823 0.868657
>852 0.871642
>889 0.871642
>892 0.871642
>992 0.873134
Done!
Best: 0.873134
Test Accuracy: 85.15152
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [用于评估机器学习算法的训练-测试分割](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/)
*   [如何在 Python 中从零开始实现感知器算法](https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/)
*   [如何用 Python 反向传播来编码神经网络(从零开始)](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)

### 蜜蜂

*   [sklearn . datasets . make _ classification APIS](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)。
*   [sklearn . metrics . accuracy _ score APIS](https://Sklearn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)。
*   num py . random . rand API。

## 摘要

在本教程中，您发现了如何手动优化神经网络模型的权重。

具体来说，您了解到:

*   如何从零开始开发神经网络模型的正向推理通路？
*   如何为二进制分类优化感知器模型的权重。
*   如何使用随机爬山优化多层感知器模型的权重。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。