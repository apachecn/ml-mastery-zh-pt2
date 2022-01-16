# 使用随机优化算法的特征选择

> 原文：<https://machinelearningmastery.com/feature-selection-with-optimization/>

最后更新于 2021 年 10 月 12 日

通常，可以通过从训练数据集中移除输入特征(列)来开发更简单且表现更好的机器学习模型。

这被称为特征选择，有许多不同类型的算法可以使用。

可以将特征选择问题框架为优化问题。在输入特征很少的情况下，可以评估输入特征的所有可能组合，并最终找到最佳子集。在大量输入特征的情况下，可以使用随机优化算法来探索搜索空间并找到有效的特征子集。

在本教程中，您将发现如何在机器学习中使用优化算法进行特征选择。

完成本教程后，您将知道:

*   特征选择问题可以广义地定义为优化问题。
*   如何枚举数据集输入要素的所有可能子集？
*   如何应用随机优化选择输入特征的最优子集？

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![How to Use Optimization for Feature Selection](img/b5963ba625cea1af1231c33e5d03fc2b.png)

如何使用优化进行特征选择
图片由[格雷戈里“斯洛博丹”史密斯](https://www.flickr.com/photos/slobirdr/22613339576/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  特征选择的优化
2.  枚举所有特征子集
3.  优化特征子集

## 特征选择的优化

特征选择是在开发预测模型时减少输入变量数量的过程。

希望减少输入变量的数量，以降低建模的计算成本，并在某些情况下提高模型的表现。有许多不同类型的特征选择算法，尽管它们可以大致分为两种主要类型:包装和过滤方法。

包装器特征选择方法创建许多具有不同输入特征子集的模型，并根据表现度量选择那些导致最佳表现模型的特征。这些方法与变量类型无关，尽管它们在计算上很昂贵。RFE 是包装特征选择方法的一个很好的例子。

过滤器特征选择方法使用统计技术来评估每个输入变量和目标变量之间的关系，并且这些分数被用作选择(过滤)将在模型中使用的那些输入变量的基础。

*   **包装特征选择**:搜索表现良好的特征子集。
*   **过滤特征选择**:根据特征与目标的关系选择特征子集。

有关选择要素选择算法的更多信息，请参见教程:

*   [如何选择机器学习的特征选择方法](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/)

一种流行的包装方法是递归特征消除或 RFE 算法。

RFE 的工作原理是从训练数据集中的所有要素开始搜索要素子集，并成功移除要素，直到剩余所需数量。

这是通过拟合模型核心中使用的给定机器学习算法、按重要性排列特征、丢弃最不重要的特征以及重新拟合模型来实现的。重复此过程，直到保留指定数量的特征。

有关 RFE 的更多信息，请参见教程:

*   [Python 中特征选择的递归特征消除(RFE)](https://machinelearningmastery.com/rfe-feature-selection-in-python/)

包装器特征选择的问题可以被框定为优化问题。也就是说，找到导致最佳模型表现的输入特征子集。

RFE 是系统地解决这个问题的一种方法，尽管它可能受到大量特征的限制。

当特征数量非常大时，另一种方法是使用随机优化算法，例如随机爬山算法。当特征的数量相对较少时，可以列举所有可能的特征子集。

*   **输入变量少**:枚举所有可能的特征子集。
*   **多输入特征**:寻找好的特征子集的随机优化算法。

既然我们已经熟悉了特征选择可以作为一个优化问题来探索的想法，那么让我们来看看如何枚举所有可能的特征子集。

## 枚举所有特征子集

当输入变量的数量相对较少并且模型评估相对较快时，则有可能枚举所有可能的输入变量子集。

这意味着在给定每个可能的唯一输入变量组的情况下，使用测试工具来评估模型的表现。

我们将通过一个工作示例来探讨如何做到这一点。

首先，让我们定义一个输入特征很少的小型二进制类别数据集。我们可以使用 [make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)定义一个包含五个输入变量的数据集，其中两个是信息变量，一千行。

下面的示例定义了数据集并总结了它的形状。

```py
# define a small classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=3, random_state=1)
# summarize the shape of the dataset
print(X.shape, y.shape)
```

运行该示例将创建数据集，并确认它具有所需的形状。

```py
(1000, 5) (1000,)
```

接下来，我们可以使用在整个数据集上评估的模型来建立表现基线。

我们将使用[决策树分类器](https://Sklearn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)作为模型，因为它的表现对输入变量的选择非常敏感。

我们将使用良好的实践来评估模型，例如[重复分层的 k-fold 交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)，重复 3 次，重复 10 次。

下面列出了完整的示例。

```py
# evaluate a decision tree on the entire small dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=3, n_informative=2, n_redundant=1, random_state=1)
# define model
model = DecisionTreeClassifier()
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例会评估整个数据集上的决策树，并报告平均和标准偏差分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型达到了大约 80.5%的准确率。

```py
Mean Accuracy: 0.805 (0.030)
```

接下来，我们可以尝试通过使用输入特征的子集来提高模型表现。

首先，我们必须选择一个代表来列举。

在这种情况下，我们将枚举一个布尔值列表，每个输入特征有一个值:如果要使用该特征，则为*真*，如果不使用该特征作为输入，则为*假*。

例如，对于五个输入特征，序列[ *真、真、真、真、真* ]将使用所有输入特征，并且[ *真、假、假、假、假* ]将仅使用第一个输入特征作为输入。

我们可以使用[乘积()Python 函数](https://docs.python.org/3/library/itertools.html#itertools.product)枚举所有长度为*5*的布尔值序列。我们必须指定有效值[ *真，假* ]和序列中的步数，等于输入变量的数量。

该函数返回一个可迭代的，我们可以直接为每个序列枚举。

```py
...
# determine the number of columns
n_cols = X.shape[1]
best_subset, best_score = None, 0.0
# enumerate all combinations of input features
for subset in product([True, False], repeat=n_cols):
	...
```

对于给定的布尔值序列，我们可以枚举它，并将其转换为序列中每个 *True* 的列索引序列。

```py
...
# convert into column indexes
ix = [i for i, x in enumerate(subset) if x]
```

如果序列没有列索引(在所有*为假*值的情况下)，那么我们可以跳过该序列。

```py
# check for now column (all False)
if len(ix) == 0:
	continue
```

然后，我们可以使用列索引来选择数据集中的列。

```py
...
# select columns
X_new = X[:, ix]
```

然后数据集的这个子集可以像我们之前做的那样被评估。

```py
...
# define model
model = DecisionTreeClassifier()
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X_new, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize scores
result = mean(scores)
```

如果模型的准确率优于目前为止找到的最佳序列，我们可以存储它。

```py
...
# check if it is better than the best so far
if best_score is None or result >= best_score:
	# better result
	best_subset, best_score = ix, result
```

就这样。

将这些联系在一起，下面列出了通过列举所有可能的特征子集进行特征选择的完整示例。

```py
# feature selection by enumerating all possible subsets of features
from itertools import product
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=3, random_state=1)
# determine the number of columns
n_cols = X.shape[1]
best_subset, best_score = None, 0.0
# enumerate all combinations of input features
for subset in product([True, False], repeat=n_cols):
	# convert into column indexes
	ix = [i for i, x in enumerate(subset) if x]
	# check for now column (all False)
	if len(ix) == 0:
		continue
	# select columns
	X_new = X[:, ix]
	# define model
	model = DecisionTreeClassifier()
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X_new, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# summarize scores
	result = mean(scores)
	# report progress
	print('>f(%s) = %f ' % (ix, result))
	# check if it is better than the best so far
	if best_score is None or result >= best_score:
		# better result
		best_subset, best_score = ix, result
# report best
print('Done!')
print('f(%s) = %f' % (best_subset, best_score))
```

运行该示例会报告所考虑的每个特征子集的模型平均分类准确率。然后在运行结束时报告最佳子集。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到特征的最佳子集涉及索引[2，3，4]处的特征，这导致了大约 83.0%的平均分类准确率，这优于之前使用所有输入特征报告的结果。

```py
>f([0, 1, 2, 3, 4]) = 0.813667
>f([0, 1, 2, 3]) = 0.827667
>f([0, 1, 2, 4]) = 0.815333
>f([0, 1, 2]) = 0.824000
>f([0, 1, 3, 4]) = 0.821333
>f([0, 1, 3]) = 0.825667
>f([0, 1, 4]) = 0.807333
>f([0, 1]) = 0.817667
>f([0, 2, 3, 4]) = 0.830333
>f([0, 2, 3]) = 0.819000
>f([0, 2, 4]) = 0.828000
>f([0, 2]) = 0.818333
>f([0, 3, 4]) = 0.830333
>f([0, 3]) = 0.821333
>f([0, 4]) = 0.816000
>f([0]) = 0.639333
>f([1, 2, 3, 4]) = 0.823667
>f([1, 2, 3]) = 0.821667
>f([1, 2, 4]) = 0.823333
>f([1, 2]) = 0.818667
>f([1, 3, 4]) = 0.818000
>f([1, 3]) = 0.820667
>f([1, 4]) = 0.809000
>f([1]) = 0.797000
>f([2, 3, 4]) = 0.827667
>f([2, 3]) = 0.755000
>f([2, 4]) = 0.827000
>f([2]) = 0.516667
>f([3, 4]) = 0.824000
>f([3]) = 0.514333
>f([4]) = 0.777667
Done!
f([0, 3, 4]) = 0.830333
```

既然我们知道了如何枚举所有可能的特征子集，让我们看看如何使用随机优化算法来选择特征子集。

## 优化特征子集

我们可以将随机优化算法应用于输入特征子集的搜索空间。

首先，让我们定义一个更大的问题，它有更多的特征，使得模型评估太慢，搜索空间太大，无法枚举所有子集。

我们将定义一个具有 10，000 行和 500 个输入特征的分类问题，其中 10 个是相关的，其余 490 个是冗余的。

```py
# define a large classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=10000, n_features=500, n_informative=10, n_redundant=490, random_state=1)
# summarize the shape of the dataset
print(X.shape, y.shape)
```

运行该示例将创建数据集，并确认它具有所需的形状。

```py
(10000, 500) (10000,)
```

我们可以通过评估数据集上具有所有输入要素的模型来建立表现基线。

因为数据集很大，模型评估很慢，我们将修改模型的评估以使用 3 重交叉验证，例如更少的折叠和没有重复。

下面列出了完整的示例。

```py
# evaluate a decision tree on the entire larger dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
# define dataset
X, y = make_classification(n_samples=10000, n_features=500, n_informative=10, n_redundant=490, random_state=1)
# define model
model = DecisionTreeClassifier()
# define evaluation procedure
cv = StratifiedKFold(n_splits=3)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例会评估整个数据集上的决策树，并报告平均和标准偏差分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型达到了大约 91.3%的准确率。

这提供了一个基线，我们期望使用特征选择能够超越它。

```py
Mean Accuracy: 0.913 (0.001)
```

我们将使用简单的随机爬山算法作为优化算法。

首先，我们必须定义目标函数。它将使用数据集和要素子集作为输入，并返回从 0(最差)到 1(最佳)的估计模型准确率。这是一个最大化优化问题。

这个目标函数只是上一节中序列和模型评估步骤的解码。

下面的 *objective()* 函数实现了这一点，并返回分数和用于有用报告的解码列子集。

```py
# objective function
def objective(X, y, subset):
	# convert into column indexes
	ix = [i for i, x in enumerate(subset) if x]
	# check for now column (all False)
	if len(ix) == 0:
		return 0.0
	# select columns
	X_new = X[:, ix]
	# define model
	model = DecisionTreeClassifier()
	# evaluate model
	scores = cross_val_score(model, X_new, y, scoring='accuracy', cv=3, n_jobs=-1)
	# summarize scores
	result = mean(scores)
	return result, ix
```

我们还需要一个能在搜索空间迈出一步的函数。

给定一个现有的解决方案，它必须修改它并返回一个新的解决方案。在这种情况下，我们将通过随机翻转子序列中列的包含/排除来实现这一点。

序列中的每个位置将被独立考虑，并且将被概率翻转，其中翻转的概率是一个超参数。

下面的 *mutate()* 函数给定一个候选解(布尔序列)和一个突变超参数，创建并返回一个修改的解(搜索空间中的一个步骤)。

*p_mutate* 值越大(在 0 到 1 的范围内)，搜索空间中的步长越大。

```py
# mutation operator
def mutate(solution, p_mutate):
	# make a copy
	child = solution.copy()
	for i in range(len(child)):
		# check for a mutation
		if rand() < p_mutate:
			# flip the inclusion
			child[i] = not child[i]
	return child
```

我们现在可以实现爬山算法了。

初始解是随机生成的序列，然后对其进行评估。

```py
...
# generate an initial point
solution = choice([True, False], size=X.shape[1])
# evaluate the initial point
solution_eval, ix = objective(X, y, solution)
```

然后，我们循环进行固定次数的迭代，创建当前解决方案的变异版本，对它们进行评估，如果分数更高，则保存它们。

```py
...
# run the hill climb
for i in range(n_iter):
	# take a step
	candidate = mutate(solution, p_mutate)
	# evaluate candidate point
	candidate_eval, ix = objective(X, y, candidate)
	# check if we should keep the new point
	if candidate_eval >= solution_eval:
		# store the new point
		solution, solution_eval = candidate, candidate_eval
	# report progress
	print('>%d f(%s) = %f' % (i+1, len(ix), solution_eval))
```

下面的*爬山()*函数实现了这一点，将数据集、目标函数和超参数作为参数，返回数据集列的最佳子集和模型的估计表现。

```py
# hill climbing local search algorithm
def hillclimbing(X, y, objective, n_iter, p_mutate):
	# generate an initial point
	solution = choice([True, False], size=X.shape[1])
	# evaluate the initial point
	solution_eval, ix = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = mutate(solution, p_mutate)
		# evaluate candidate point
		candidate_eval, ix = objective(X, y, candidate)
		# check if we should keep the new point
		if candidate_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidate_eval
		# report progress
		print('>%d f(%s) = %f' % (i+1, len(ix), solution_eval))
	return solution, solution_eval
```

然后，我们可以调用这个函数，并传入我们的合成数据集来执行特征选择的优化。

在这种情况下，我们将运行算法 100 次迭代，并对给定突变的序列进行大约 5 次翻转，这相当保守。

```py
...
# define dataset
X, y = make_classification(n_samples=10000, n_features=500, n_informative=10, n_redundant=490, random_state=1)
# define the total iterations
n_iter = 100
# probability of including/excluding a column
p_mut = 10.0 / 500.0
# perform the hill climbing search
subset, score = hillclimbing(X, y, objective, n_iter, p_mut)
```

在运行结束时，我们将把布尔序列转换成列索引(这样，如果需要，我们可以拟合最终模型)，并报告最佳子序列的表现。

```py
...
# convert into column indexes
ix = [i for i, x in enumerate(subset) if x]
print('Done!')
print('Best: f(%d) = %f' % (len(ix), score))
```

将这些结合在一起，完整的示例如下所示。

```py
# stochastic optimization for feature selection
from numpy import mean
from numpy.random import rand
from numpy.random import choice
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# objective function
def objective(X, y, subset):
	# convert into column indexes
	ix = [i for i, x in enumerate(subset) if x]
	# check for now column (all False)
	if len(ix) == 0:
		return 0.0
	# select columns
	X_new = X[:, ix]
	# define model
	model = DecisionTreeClassifier()
	# evaluate model
	scores = cross_val_score(model, X_new, y, scoring='accuracy', cv=3, n_jobs=-1)
	# summarize scores
	result = mean(scores)
	return result, ix

# mutation operator
def mutate(solution, p_mutate):
	# make a copy
	child = solution.copy()
	for i in range(len(child)):
		# check for a mutation
		if rand() < p_mutate:
			# flip the inclusion
			child[i] = not child[i]
	return child

# hill climbing local search algorithm
def hillclimbing(X, y, objective, n_iter, p_mutate):
	# generate an initial point
	solution = choice([True, False], size=X.shape[1])
	# evaluate the initial point
	solution_eval, ix = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = mutate(solution, p_mutate)
		# evaluate candidate point
		candidate_eval, ix = objective(X, y, candidate)
		# check if we should keep the new point
		if candidate_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidate_eval
		# report progress
		print('>%d f(%s) = %f' % (i+1, len(ix), solution_eval))
	return solution, solution_eval

# define dataset
X, y = make_classification(n_samples=10000, n_features=500, n_informative=10, n_redundant=490, random_state=1)
# define the total iterations
n_iter = 100
# probability of including/excluding a column
p_mut = 10.0 / 500.0
# perform the hill climbing search
subset, score = hillclimbing(X, y, objective, n_iter, p_mut)
# convert into column indexes
ix = [i for i, x in enumerate(subset) if x]
print('Done!')
print('Best: f(%d) = %f' % (len(ix), score))
```

运行该示例会报告所考虑的每个特征子集的模型平均分类准确率。然后在运行结束时报告最佳子集。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到最佳表现是通过 239 个特征的子集和大约 91.8%的分类准确率实现的。

这比在所有输入特征上评估的模型要好。

虽然结果更好，但我们知道我们可以做得更好，也许通过调整优化算法的超参数，或者通过使用替代优化算法。

```py
...
>80 f(240) = 0.918099
>81 f(236) = 0.918099
>82 f(238) = 0.918099
>83 f(236) = 0.918099
>84 f(239) = 0.918099
>85 f(240) = 0.918099
>86 f(239) = 0.918099
>87 f(245) = 0.918099
>88 f(241) = 0.918099
>89 f(239) = 0.918099
>90 f(239) = 0.918099
>91 f(241) = 0.918099
>92 f(243) = 0.918099
>93 f(245) = 0.918099
>94 f(239) = 0.918099
>95 f(245) = 0.918099
>96 f(244) = 0.918099
>97 f(242) = 0.918099
>98 f(238) = 0.918099
>99 f(248) = 0.918099
>100 f(238) = 0.918099
Done!
Best: f(239) = 0.918099
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [Python 中特征选择的递归特征消除(RFE)](https://machinelearningmastery.com/rfe-feature-selection-in-python/)
*   [如何选择机器学习的特征选择方法](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/)

### 蜜蜂

*   [sklearn . datasets . make _ classification API](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)。
*   [te tools . product API](https://docs.python.org/3/library/itertools.html#itertools.product)。

## 摘要

在本教程中，您发现了如何在机器学习中使用优化算法进行特征选择。

具体来说，您了解到:

*   特征选择问题可以广义地定义为优化问题。
*   如何枚举数据集输入要素的所有可能子集？
*   如何应用随机优化选择输入特征的最优子集？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。