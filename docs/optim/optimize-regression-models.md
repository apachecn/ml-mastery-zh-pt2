# 如何使用优化算法手动拟合回归模型

> 原文：<https://machinelearningmastery.com/optimize-regression-models/>

最后更新于 2021 年 10 月 12 日

使用线性回归和局部搜索优化算法对训练数据拟合回归模型。

像线性回归和逻辑回归这样的模型是通过最小二乘优化来训练的，这是找到使这些模型的误差最小化的系数的最有效的方法。

然而，可以使用替代的**优化算法将回归模型**拟合到训练数据集。这可能是一个有用的练习，以了解更多关于回归函数和优化在应用机器学习中的核心性质。对于数据不符合最小二乘优化程序要求的回归，也可能需要它。

在本教程中，您将发现如何手动优化回归模型的系数。

完成本教程后，您将知道:

*   如何从零开始开发回归的推理模型？
*   如何优化预测数值的线性回归模型的系数？
*   如何用随机爬山法优化逻辑回归模型的系数？

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![How to Use Optimization Algorithms to Manually Fit Regression Models](img/ad98febd62a7321d9fa60db6a67425e4.png)

如何使用优化算法手动拟合回归模型
图片由[克里斯蒂安·科林斯](https://www.flickr.com/photos/collins_family/31023265312/)提供，版权所有。

## 教程概述

本教程分为三个部分；它们是:

1.  优化回归模型
2.  优化线性回归模型
3.  优化逻辑回归模型

## 优化回归模型

回归模型，像线性回归和逻辑回归，是统计学领域中众所周知的算法。

两种算法都是线性的，这意味着模型的输出是输入的加权和。[线性回归](https://en.wikipedia.org/wiki/Linear_regression)是针对需要预测一个数的“*回归*”问题设计的，[逻辑回归](https://en.wikipedia.org/wiki/Logistic_regression)是针对需要预测一个类标签的“*分类*”问题设计的。

这些回归模型包括使用优化算法来为模型的每个输入找到一组系数，从而最小化预测误差。因为模型是线性的并且被很好地理解，所以可以使用有效的优化算法。

在线性回归的情况下，系数可以通过最小二乘优化找到，可以使用[线性代数](https://machinelearningmastery.com/solve-linear-regression-using-linear-algebra/)求解。在逻辑回归的情况下，通常使用局部搜索优化算法。

可以使用任意优化算法来训练线性和逻辑回归模型。

也就是说，我们可以定义一个回归模型，并使用给定的优化算法为该模型找到一组系数，从而使预测误差最小或分类准确率最大。

平均而言，使用替代优化算法的效率低于使用推荐的优化算法。尽管如此，在某些特定情况下，例如如果输入数据不符合模型的期望(如高斯分布)并且与外部输入不相关，这可能更有效。

演示优化在训练机器学习算法，特别是回归模型中的核心性质也是一个有趣的练习。

接下来，让我们探索如何使用随机爬山训练线性回归模型。

## 优化线性回归模型

[线性回归](https://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/)模型可能是从数据中学习的最简单的预测模型。

该模型对每个输入都有一个系数，预测输出只是一些输入和系数的权重。

在本节中，我们将优化线性回归模型的系数。

首先，让我们定义一个合成回归问题，我们可以将其作为优化模型的重点。

我们可以使用[make _ revolution()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_regression.html)定义一个有 1000 行和 10 个输入变量的回归问题。

下面的示例创建数据集并总结数据的形状。

```py
# define a regression dataset
from sklearn.datasets import make_regression
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, noise=0.2, random_state=1)
# summarize the shape of the dataset
print(X.shape, y.shape)
```

运行该示例会打印出创建的数据集的形状，这证实了我们的预期。

```py
(1000, 10) (1000,)
```

接下来，我们需要定义一个线性回归模型。

在我们优化模型系数之前，我们必须发展模型和我们对它如何工作的信心。

让我们从开发一个函数开始，该函数为来自数据集的给定输入数据行计算模型的激活。

该函数将获取模型的数据行和系数，并计算输入的加权和，加上一个额外的 y 截距(也称为偏移或偏差)系数。下面的 *predict_row()* 函数实现了这一点。

我们正在使用简单的 Python 列表和命令式编程风格，而不是故意使用 [NumPy 数组](https://machinelearningmastery.com/gentle-introduction-n-dimensional-arrays-python-numpy/)或列表压缩，以使代码对 Python 初学者来说更易读。请随意优化它，并在下面的评论中发布您的代码。

```py
# linear regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
	for i in range(len(row)):
		result += coefficients[i] * row[i]
	return result
```

接下来，我们可以为给定数据集中的每一行调用 *predict_row()* 函数。下面的*预测 _ 数据集()*函数实现了这一点。

同样，为了可读性，我们有意使用简单的命令式编码风格，而不是列表压缩。

```py
# use model coefficients to generate predictions for a dataset of rows
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
		# make a prediction
		yhat = predict_row(row, coefficients)
		# store the prediction
		yhats.append(yhat)
	return yhats
```

最后，我们可以使用该模型对我们的合成数据集进行预测，以确认它都工作正常。

我们可以使用 [rand()函数](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html)生成一组随机的模型系数。

回想一下，我们每个输入需要一个系数(本数据集中有 10 个输入)加上 y 截距系数的额外权重。

```py
...
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, noise=0.2, random_state=1)
# determine the number of coefficients
n_coeff = X.shape[1] + 1
# generate random coefficients
coefficients = rand(n_coeff)
```

然后，我们可以将这些系数与数据集一起使用来进行预测。

```py
...
# generate predictions for dataset
yhat = predict_dataset(X, coefficients)
```

我们可以评估这些预测的均方误差。

```py
...
# calculate model prediction error
score = mean_squared_error(y, yhat)
print('MSE: %f' % score)
```

就这样。

我们可以将所有这些联系在一起，并演示我们的线性回归模型用于回归预测建模。下面列出了完整的示例。

```py
# linear regression model
from numpy.random import rand
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# linear regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
	for i in range(len(row)):
		result += coefficients[i] * row[i]
	return result

# use model coefficients to generate predictions for a dataset of rows
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
		# make a prediction
		yhat = predict_row(row, coefficients)
		# store the prediction
		yhats.append(yhat)
	return yhats

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, noise=0.2, random_state=1)
# determine the number of coefficients
n_coeff = X.shape[1] + 1
# generate random coefficients
coefficients = rand(n_coeff)
# generate predictions for dataset
yhat = predict_dataset(X, coefficients)
# calculate model prediction error
score = mean_squared_error(y, yhat)
print('MSE: %f' % score)
```

运行示例会为训练数据集中的每个示例生成预测，然后打印预测的均方误差。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在给定一组随机权重的情况下，我们预计会有很大的误差，这就是我们在这种情况下看到的，误差值约为 7，307 个单位。

```py
MSE: 7307.756740
```

我们现在可以优化数据集的系数，以实现该数据集的低误差。

首先，我们需要将数据集分成训练集和测试集。重要的是保留一些未用于优化模型的数据，以便我们可以在用于对新数据进行预测时对模型的表现进行合理的估计。

我们将使用 67%的数据进行训练，剩下的 33%作为测试集来评估模型的表现。

```py
...
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

接下来，我们可以开发一个[随机爬山算法](https://machinelearningmastery.com/stochastic-hill-climbing-in-python-from-scratch/)。

优化算法需要一个目标函数来优化。它必须采用一组系数，并返回一个对应于更好模型的最小化或最大化分数。

在这种情况下，我们将使用给定的一组系数来评估模型的均方误差，并返回误差分数，该分数必须最小化。

给定数据集和一组系数，下面的 *objective()* 函数实现了这一点，并返回模型的误差。

```py
# objective function
def objective(X, y, coefficients):
	# generate predictions for dataset
	yhat = predict_dataset(X, coefficients)
	# calculate accuracy
	score = mean_squared_error(y, yhat)
	return score
```

接下来，我们可以定义随机爬山算法。

该算法将需要一个初始解(例如，随机系数)，并将反复不断地对解进行小的改变，并检查它是否产生一个表现更好的模型。对当前解决方案的更改量由步长超参数控制。该过程将持续固定次数的迭代，也作为超参数提供。

下面的*爬山()*函数实现了这一点，将数据集、目标函数、初始解和超参数作为参数，并返回找到的最佳系数集和估计的表现。

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
		if candidte_eval <= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
			print('>%d %.5f' % (i, solution_eval))
	return [solution, solution_eval]
```

然后我们可以调用这个函数，传入一组初始系数作为初始解，并将训练数据集作为数据集来优化模型。

```py
...
# define the total iterations
n_iter = 2000
# define the maximum step size
step_size = 0.15
# determine the number of coefficients
n_coef = X.shape[1] + 1
# define the initial solution
solution = rand(n_coef)
# perform the hill climbing search
coefficients, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
print('Done!')
print('Coefficients: %s' % coefficients)
print('Train MSE: %f' % (score))
```

最后，我们可以在测试数据集上评估最佳模型并报告表现。

```py
...
# generate predictions for the test dataset
yhat = predict_dataset(X_test, coefficients)
# calculate accuracy
score = mean_squared_error(y_test, yhat)
print('Test MSE: %f' % (score))
```

将这些联系在一起，下面列出了在综合回归数据集上优化线性回归模型系数的完整示例。

```py
# optimize linear regression coefficients for regression dataset
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# linear regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
	for i in range(len(row)):
		result += coefficients[i] * row[i]
	return result

# use model coefficients to generate predictions for a dataset of rows
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
		# make a prediction
		yhat = predict_row(row, coefficients)
		# store the prediction
		yhats.append(yhat)
	return yhats

# objective function
def objective(X, y, coefficients):
	# generate predictions for dataset
	yhat = predict_dataset(X, coefficients)
	# calculate accuracy
	score = mean_squared_error(y, yhat)
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
		if candidte_eval <= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
			print('>%d %.5f' % (i, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, noise=0.2, random_state=1)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# define the total iterations
n_iter = 2000
# define the maximum step size
step_size = 0.15
# determine the number of coefficients
n_coef = X.shape[1] + 1
# define the initial solution
solution = rand(n_coef)
# perform the hill climbing search
coefficients, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
print('Done!')
print('Coefficients: %s' % coefficients)
print('Train MSE: %f' % (score))
# generate predictions for the test dataset
yhat = predict_dataset(X_test, coefficients)
# calculate accuracy
score = mean_squared_error(y_test, yhat)
print('Test MSE: %f' % (score))
```

每次对模型进行改进时，运行该示例都会报告迭代次数和均方误差。

在搜索结束时，报告最佳系数集在训练数据集上的表现，并计算和报告相同模型在测试数据集上的表现。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到优化算法在训练和测试数据集上都找到了一组误差约为 0.08 的系数。

该算法在训练数据集和测试数据集上找到了表现非常相似的模型，这是一个好的迹象，表明该模型没有过拟合(过度优化)训练数据集。这意味着该模型可以很好地推广到新数据。

```py
...
>1546 0.35426
>1567 0.32863
>1572 0.32322
>1619 0.24890
>1665 0.24800
>1691 0.24162
>1715 0.15893
>1809 0.15337
>1892 0.14656
>1956 0.08042
Done!
Coefficients: [ 1.30559829e-02 -2.58299382e-04  3.33118191e+00  3.20418534e-02
  1.36497902e-01  8.65445367e+01  2.78356715e-02 -8.50901499e-02
  8.90078243e-02  6.15779867e-02 -3.85657793e-02]
Train MSE: 0.080415
Test MSE: 0.080779
```

现在我们已经熟悉了如何手动优化线性回归模型的系数，让我们看看如何扩展示例来优化用于分类的逻辑回归模型的系数。

## 优化逻辑回归模型

逻辑回归模型是线性回归的扩展，用于分类预测建模。

[Logistic 回归](https://machinelearningmastery.com/logistic-regression-with-maximum-likelihood-estimation/)是针对二分类任务的，意思是数据集有两个类标签，class=0 和 class=1。

输出首先包括计算输入的加权和，然后将这个加权和传递给一个逻辑函数，也称为 sigmoid 函数。对于属于类=1 的例子，结果是 0 和 1 之间的[二项式概率](https://machinelearningmastery.com/discrete-probability-distributions-for-machine-learning/)。

在本节中，我们将在上一节所学的基础上优化回归模型的系数以进行分类。我们将开发模型并用随机系数进行测试，然后使用随机爬山来优化模型系数。

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

接下来，我们需要定义一个逻辑回归模型。

让我们从更新 *predict_row()* 函数开始，通过逻辑函数传递输入和系数的加权和。

逻辑函数定义为:

*   logistic = 1.0/(1.0+exp(-结果))

其中，结果是输入和系数的加权和，exp()是 *e* ( [欧拉数](https://en.wikipedia.org/wiki/E_(mathematical_constant)))乘以提供值的幂，通过 [exp()函数](https://docs.python.org/3/library/math.html#math.exp)实现。

更新后的 *predict_row()* 功能如下。

```py
# logistic regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
	for i in range(len(row)):
		result += coefficients[i] * row[i]
	# logistic function
	logistic = 1.0 / (1.0 + exp(-result))
	return logistic
```

这就是线性回归到逻辑回归的变化。

与线性回归一样，我们可以用一组随机模型系数来测试模型。

```py
...
# determine the number of coefficients
n_coeff = X.shape[1] + 1
# generate random coefficients
coefficients = rand(n_coeff)
# generate predictions for dataset
yhat = predict_dataset(X, coefficients)
```

该模型做出的预测是属于类=1 的例子的概率。

对于预期的类标签，我们可以将预测舍入为整数值 0 和 1。

```py
...
# round predictions to labels
yhat = [round(y) for y in yhat]
```

我们可以评估这些预测的分类准确性。

```py
...
# calculate accuracy
score = accuracy_score(y, yhat)
print('Accuracy: %f' % score)
```

就这样。

我们可以将所有这些联系在一起，并演示我们用于二分类的简单逻辑回归模型。下面列出了完整的示例。

```py
# logistic regression function for binary classification
from math import exp
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# logistic regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
	for i in range(len(row)):
		result += coefficients[i] * row[i]
	# logistic function
	logistic = 1.0 / (1.0 + exp(-result))
	return logistic

# use model coefficients to generate predictions for a dataset of rows
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
		# make a prediction
		yhat = predict_row(row, coefficients)
		# store the prediction
		yhats.append(yhat)
	return yhats

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# determine the number of coefficients
n_coeff = X.shape[1] + 1
# generate random coefficients
coefficients = rand(n_coeff)
# generate predictions for dataset
yhat = predict_dataset(X, coefficients)
# round predictions to labels
yhat = [round(y) for y in yhat]
# calculate accuracy
score = accuracy_score(y, yhat)
print('Accuracy: %f' % score)
```

运行示例会为训练数据集中的每个示例生成预测，然后打印预测的分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在给定一组随机权重和每个类中具有相同数量示例的数据集的情况下，我们期望大约 50%的准确性，这大约是我们在本例中看到的。

```py
Accuracy: 0.540000
```

我们现在可以优化数据集的权重，以在该数据集上获得良好的准确率。

用于线性回归的随机爬山算法可以再次用于逻辑回归。

重要的区别是更新了*目标()*函数，使用分类准确率而不是均方误差来舍入预测和评估模型。

```py
# objective function
def objective(X, y, coefficients):
	# generate predictions for dataset
	yhat = predict_dataset(X, coefficients)
	# round predictions to labels
	yhat = [round(y) for y in yhat]
	# calculate accuracy
	score = accuracy_score(y, yhat)
	return score
```

*爬山()*功能也必须更新，以最大化解的得分，而不是线性回归情况下的最小化。

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

最后，通过搜索找到的系数可以在运行结束时使用分类准确率进行评估。

```py
...
# generate predictions for the test dataset
yhat = predict_dataset(X_test, coefficients)
# round predictions to labels
yhat = [round(y) for y in yhat]
# calculate accuracy
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %f' % (score))
```

将所有这些联系在一起，下面列出了使用随机爬山来最大化逻辑回归模型的分类准确率的完整示例。

```py
# optimize logistic regression model with a stochastic hill climber
from math import exp
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# logistic regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
	for i in range(len(row)):
		result += coefficients[i] * row[i]
	# logistic function
	logistic = 1.0 / (1.0 + exp(-result))
	return logistic

# use model coefficients to generate predictions for a dataset of rows
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
		# make a prediction
		yhat = predict_row(row, coefficients)
		# store the prediction
		yhats.append(yhat)
	return yhats

# objective function
def objective(X, y, coefficients):
	# generate predictions for dataset
	yhat = predict_dataset(X, coefficients)
	# round predictions to labels
	yhat = [round(y) for y in yhat]
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
n_iter = 2000
# define the maximum step size
step_size = 0.1
# determine the number of coefficients
n_coef = X.shape[1] + 1
# define the initial solution
solution = rand(n_coef)
# perform the hill climbing search
coefficients, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
print('Done!')
print('Coefficients: %s' % coefficients)
print('Train Accuracy: %f' % (score))
# generate predictions for the test dataset
yhat = predict_dataset(X_test, coefficients)
# round predictions to labels
yhat = [round(y) for y in yhat]
# calculate accuracy
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %f' % (score))
```

每次对模型进行改进时，运行该示例都会报告迭代次数和分类准确率。

在搜索结束时，报告最佳系数集在训练数据集上的表现，并计算和报告相同模型在测试数据集上的表现。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到优化算法找到了一组权重，在训练数据集上获得了大约 87.3%的准确率，在测试数据集上获得了大约 83.9%的准确率。

```py
...
>200 0.85672
>225 0.85672
>230 0.85672
>245 0.86418
>281 0.86418
>285 0.86716
>294 0.86716
>306 0.86716
>316 0.86716
>317 0.86716
>320 0.86866
>348 0.86866
>362 0.87313
>784 0.87313
>1649 0.87313
Done!
Coefficients: [-0.04652756  0.23243427  2.58587637 -0.45528253 -0.4954355  -0.42658053]
Train Accuracy: 0.873134
Test Accuracy: 0.839394
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

*   [用于评估机器学习算法的训练-测试分割](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/)
*   [如何在 Python 中从零开始实现线性回归](https://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/)
*   [如何在 Python 中从零开始实现逻辑回归](https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/)

### 蜜蜂

*   [sklearn . dataset . make _ revolution APIS](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_regression.html)。
*   [sklearn . datasets . make _ classification APIS](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)。
*   [sklearn . metrics . mean _ squared _ error API](https://Sklearn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)。
*   num py . random . rand API。

### 文章

*   [线性回归，维基百科](https://en.wikipedia.org/wiki/Linear_regression)。
*   [逻辑回归，维基百科](https://en.wikipedia.org/wiki/Logistic_regression)。

## 摘要

在本教程中，您发现了如何手动优化回归模型的系数。

具体来说，您了解到:

*   如何从零开始开发回归的推理模型？
*   如何优化预测数值的线性回归模型的系数？
*   如何用随机爬山法优化逻辑回归模型的系数？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。