# 如何手动优化机器学习模型超参数

> 原文：<https://machinelearningmastery.com/manually-optimize-hyperparameters/>

最后更新于 2021 年 10 月 12 日

机器学习算法有[超参数](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)，允许算法适合特定的数据集。

虽然超参数的影响可以被普遍理解，但是它们对数据集的具体影响以及它们在学习过程中的相互作用可能是未知的。因此，作为机器学习项目的一部分，调整算法超参数的值非常重要。

通常使用简单的优化算法来调整超参数，例如网格搜索和随机搜索。另一种方法是使用随机优化算法，如随机爬山算法。

在本教程中，您将发现如何手动优化机器学习算法的超参数。

完成本教程后，您将知道:

*   随机优化算法可以代替网格和随机搜索用于超参数优化。
*   如何使用随机爬山算法来调整感知器算法的超参数。
*   如何手动优化 XGBoost 梯度提升算法的超参数？

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![How to Manually Optimize Machine Learning Model Hyperparameters](img/2c92d7d120d80086772a03918aa34e0a.png)

如何手动优化机器学习模型超参数
图片由[约翰·法雷尔·麦克唐纳](https://www.flickr.com/photos/jfmacdonald/19867924249/)提供，版权所有。

## 教程概述

本教程分为三个部分；它们是:

1.  手动超参数优化
2.  感知器超参数优化
3.  超参数优化

## 手动超参数优化

机器学习模型有您必须设置的超参数，以便根据数据集定制模型。

通常，超参数对模型的一般影响是已知的，但是如何为给定数据集最佳地设置超参数和交互超参数的组合是具有挑战性的。

一种更好的方法是客观地搜索模型超参数的不同值，并选择一个子集，该子集导致在给定数据集上获得最佳表现的模型。这被称为超参数优化，或超参数调整。

可以使用一系列不同的优化算法，尽管最简单和最常见的两种方法是随机搜索和网格搜索。

*   **随机搜索**。将搜索空间定义为超参数值的有界域，并在该域中随机采样点。
*   **网格搜索**。将搜索空间定义为超参数值网格，并计算网格中的每个位置。

网格搜索非常适合抽查那些通常表现良好的组合。随机搜索非常适合于发现和获得你凭直觉无法猜到的超参数组合，尽管它通常需要更多的时间来执行。

有关超参数优化的网格和随机搜索的更多信息，请参见教程:

*   [随机搜索和网格搜索的超参数优化](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/)

网格和随机搜索是原始的优化算法，可以使用我们喜欢的任何优化来调整机器学习算法的表现。例如，可以使用随机优化算法。当需要良好或出色的表现，并且有足够的资源来调整模型时，这可能是可取的。

接下来，让我们看看如何使用随机爬山算法来调整感知器算法的表现。

## 感知器超参数优化

[感知器算法](https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/)是最简单的人工神经网络。

它是单个神经元的模型，可用于两类分类问题，并为以后开发更大的网络奠定了基础。

在本节中，我们将探讨如何手动优化感知器模型的超参数。

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

Sklearn 通过[感知器类](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)提供感知器模型的实现。

在我们调整模型的超参数之前，我们可以使用默认超参数建立表现基线。

我们将通过[重复分层 K 折交叉验证类](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)使用[重复分层 K 折交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)的良好实践来评估模型。

下面列出了在我们的合成二进制类别数据集上使用默认超参数评估感知器模型的完整示例。

```py
# perceptron default hyperparameters for binary classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# define model
model = Perceptron()
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行示例报告评估模型，并报告分类准确率的平均值和标准偏差。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到具有默认超参数的模型实现了大约 78.5%的分类准确率。

我们希望通过优化超参数，我们可以获得比这更好的表现。

```py
Mean Accuracy: 0.786 (0.069)
```

接下来，我们可以使用随机爬山算法来优化感知器模型的超参数。

有许多超参数我们可以优化，尽管我们将重点关注两个可能对模型的学习行为影响最大的超参数；它们是:

*   学习率( *eta0* )。
*   正则化(*α*)。

[学习率](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/)控制模型基于预测误差的更新量，并控制学习速度。eta 的默认值是 1.0。合理的值大于零(例如大于 1e-8 或 1e-10)并且可能小于 1.0

默认情况下，感知器不使用任何正则化，但是我们将启用“*弹性网*”正则化，该正则化在学习过程中同时应用了 [L1 和 L2 正则化](https://machinelearningmastery.com/weight-regularization-to-reduce-overfitting-of-deep-learning-models/)。这将鼓励模型寻求更小的模型权重，并反过来通常获得更好的表现。

我们将调整控制正则化权重的“*α*”超参数，例如它影响学习的量。如果设置为 0.0，则好像没有使用正则化。合理的值介于 0.0 和 1.0 之间。

首先，我们需要为优化算法定义目标函数。我们将使用重复分层 k 倍交叉验证的平均分类准确率来评估配置。我们将寻求最大限度地提高配置的准确性。

下面的*目标()*函数实现了这一点，获取数据集和配置值列表。配置值(学习率和正则化权重)被解包，用于配置模型，然后对模型进行评估，并返回平均准确率。

```py
# objective function
def objective(X, y, cfg):
	# unpack config
	eta, alpha = cfg
	# define model
	model = Perceptron(penalty='elasticnet', alpha=alpha, eta0=eta)
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# calculate mean accuracy
	result = mean(scores)
	return result
```

接下来，我们需要一个函数在搜索空间中迈出一步。

搜索空间由两个变量定义(*η*和*α*)。搜索空间中的一个步骤必须与先前的值有某种关系，并且必须绑定到可感知的值(例如，在 0 和 1 之间)。

我们将使用一个“*步长*”超参数来控制允许算法从现有配置移动多远。将使用高斯分布概率地选择新的配置，当前值作为分布的平均值，步长作为分布的标准偏差。

我们可以使用 [randn() NumPy 函数](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html)生成高斯分布的随机数。

下面的 *step()* 函数实现了这一点，并将在搜索空间中迈出一步，并使用现有配置生成新配置。

```py
# take a step in the search space
def step(cfg, step_size):
	# unpack the configuration
	eta, alpha = cfg
	# step eta
	new_eta = eta + randn() * step_size
	# check the bounds of eta
	if new_eta <= 0.0:
		new_eta = 1e-8
	# step alpha
	new_alpha = alpha + randn() * step_size
	# check the bounds of alpha
	if new_alpha < 0.0:
		new_alpha = 0.0
	# return the new configuration
	return [new_eta, new_alpha]
```

接下来，我们需要实现[随机爬山算法](https://machinelearningmastery.com/stochastic-hill-climbing-in-python-from-scratch/)，该算法将调用我们的*目标()*函数来评估候选解，并调用我们的 *step()* 函数在搜索空间中迈出一步。

搜索首先生成一个随机的初始解，在这种情况下，eta 和 alpha 值在 0 和 1 的范围内。然后评估初始解决方案，并将其作为当前最佳工作解决方案。

```py
...
# starting point for the search
solution = [rand(), rand()]
# evaluate the initial point
solution_eval = objective(X, y, solution)
```

接下来，该算法迭代固定次数的迭代，作为搜索的超参数。每次迭代都包括采取一个步骤并评估新的候选解决方案。

```py
...
# take a step
candidate = step(solution, step_size)
# evaluate candidate point
candidate_eval = objective(X, y, candidate)
```

如果新方案优于当前工作方案，则作为新的当前工作方案。

```py
...
# check if we should keep the new point
if candidate_eval >= solution_eval:
	# store the new point
	solution, solution_eval = candidate, candidate_eval
	# report progress
	print('>%d, cfg=%s %.5f' % (i, solution, solution_eval))
```

搜索结束后，将返回最佳解决方案及其表现。

将这些联系在一起，下面的*爬山()*函数实现了随机爬山算法，用于调整感知器算法，将数据集、目标函数、迭代次数和步长作为参数。

```py
# hill climbing local search algorithm
def hillclimbing(X, y, objective, n_iter, step_size):
	# starting point for the search
	solution = [rand(), rand()]
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = step(solution, step_size)
		# evaluate candidate point
		candidate_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidate_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidate_eval
			# report progress
			print('>%d, cfg=%s %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]
```

然后我们可以调用算法并报告搜索结果。

在这种情况下，我们将运行该算法 100 次迭代，并使用 0.1 的步长，这是经过一点反复试验后选择的。

```py
...
# define the total iterations
n_iter = 100
# step size in the search space
step_size = 0.1
# perform the hill climbing search
cfg, score = hillclimbing(X, y, objective, n_iter, step_size)
print('Done!')
print('cfg=%s: Mean Accuracy: %f' % (cfg, score))
```

将这些联系在一起，手动调整感知器算法的完整示例如下所示。

```py
# manually search perceptron hyperparameters for binary classification
from numpy import mean
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron

# objective function
def objective(X, y, cfg):
	# unpack config
	eta, alpha = cfg
	# define model
	model = Perceptron(penalty='elasticnet', alpha=alpha, eta0=eta)
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# calculate mean accuracy
	result = mean(scores)
	return result

# take a step in the search space
def step(cfg, step_size):
	# unpack the configuration
	eta, alpha = cfg
	# step eta
	new_eta = eta + randn() * step_size
	# check the bounds of eta
	if new_eta <= 0.0:
		new_eta = 1e-8
	# step alpha
	new_alpha = alpha + randn() * step_size
	# check the bounds of alpha
	if new_alpha < 0.0:
		new_alpha = 0.0
	# return the new configuration
	return [new_eta, new_alpha]

# hill climbing local search algorithm
def hillclimbing(X, y, objective, n_iter, step_size):
	# starting point for the search
	solution = [rand(), rand()]
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = step(solution, step_size)
		# evaluate candidate point
		candidate_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidate_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidate_eval
			# report progress
			print('>%d, cfg=%s %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# define the total iterations
n_iter = 100
# step size in the search space
step_size = 0.1
# perform the hill climbing search
cfg, score = hillclimbing(X, y, objective, n_iter, step_size)
print('Done!')
print('cfg=%s: Mean Accuracy: %f' % (cfg, score))
```

每次在搜索过程中看到改进时，运行示例都会报告配置和结果。运行结束时，会报告最佳配置和结果。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，最佳结果涉及在 1.004 时使用略高于 1 的学习率和大约 0.002 的正则化权重，实现了大约 79.1%的平均准确率，优于实现了大约 78.5%的准确率的默认配置。

**能不能得到更好的结果？**
在下面的评论里告诉我。

```py
>0, cfg=[0.5827274503894747, 0.260872709578015] 0.70533
>4, cfg=[0.5449820307807399, 0.3017271170801444] 0.70567
>6, cfg=[0.6286475606495414, 0.17499090243915086] 0.71933
>7, cfg=[0.5956196828965779, 0.0] 0.78633
>8, cfg=[0.5878361167354715, 0.0] 0.78633
>10, cfg=[0.6353507984485595, 0.0] 0.78633
>13, cfg=[0.5690530537610675, 0.0] 0.78633
>17, cfg=[0.6650936023999641, 0.0] 0.78633
>22, cfg=[0.9070451625704087, 0.0] 0.78633
>23, cfg=[0.9253366187387938, 0.0] 0.78633
>26, cfg=[0.9966143540220266, 0.0] 0.78633
>31, cfg=[1.0048613895650054, 0.002162219228449132] 0.79133
Done!
cfg=[1.0048613895650054, 0.002162219228449132]: Mean Accuracy: 0.791333
```

现在我们已经熟悉了如何使用随机爬山算法来调整简单机器学习算法的超参数，接下来让我们看看如何调整更高级的算法，例如 XGBoost。

## 超参数优化

XGBoost 是[极限梯度提升](https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/)的简称，是随机梯度提升机学习算法的高效实现。

随机梯度提升算法，也称为梯度提升机或树增强，是一种强大的机器学习技术，在一系列具有挑战性的机器学习问题上表现良好，甚至最好。

首先，必须安装 XGBoost 库。

您可以使用 pip 安装它，如下所示:

```py
sudo pip install xgboost
```

安装完成后，您可以通过运行以下代码来确认安装成功，并且您使用的是现代版本:

```py
# xgboost
import xgboost
print("xgboost", xgboost.__version__)
```

运行代码时，您应该会看到以下版本号或更高的版本号。

```py
xgboost 1.0.1
```

虽然 XGBoost 库有自己的 Python API，但是我们可以通过 [XGBClassifier 包装类](https://xgboost.readthedocs.io/en/latest/python/python_api.html)将 XGBoost 模型与 Sklearn API 一起使用。

模型的一个实例可以像任何其他用于模型评估的 Sklearn 类一样被实例化和使用。例如:

```py
...
# define model
model = XGBClassifier()
```

在我们调整 XGBoost 的超参数之前，我们可以使用默认的超参数建立一个表现基线。

我们将使用上一节中相同的合成二进制类别数据集和重复分层 k-fold 交叉验证的相同测试工具。

下面列出了使用默认超参数评估 XGBoost 表现的完整示例。

```py
# xgboost with default hyperparameters for binary classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# define model
model = XGBClassifier()
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例评估模型，并报告分类准确率的平均值和标准偏差。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到具有默认超参数的模型实现了大约 84.9%的分类准确率。

我们希望通过优化超参数，我们可以获得比这更好的表现。

```py
Mean Accuracy: 0.849 (0.040)
```

接下来，我们可以采用随机爬山优化算法来调整 XGBoost 模型的超参数。

对于 XGBoost 模型，我们可能需要优化许多超参数。

有关如何调整 XGBoost 模型的概述，请参见教程:

*   [如何配置梯度提升算法](https://machinelearningmastery.com/configure-gradient-boosting-algorithm/)

我们将关注四个关键的超参数；它们是:

*   学习率(*学习 _ 速率*)
*   树的数量(*n _ 估计量*)
*   子采样百分比(*子采样*)
*   树深(*最大 _ 深度*)

**学习率**控制每棵树对集成的贡献。可感知值小于 1.0，略高于 0.0(例如 1e-8)。

**树的数量**控制着集成的大小，往往越多的树越好到收益递减的地步。可感知值介于 1 棵树和数百或数千棵树之间。

**子样本**百分比定义了用于训练每棵树的随机样本大小，定义为原始数据集大小的百分比。值介于略高于 0.0 的值(例如 1e-8)和 1.0 之间

**树深**是每棵树的层数。更深的树对训练数据集来说更具体，可能会过度匹配。矮一些的树通常更好地概括。合理值在 1 到 10 或 20 之间。

首先，我们必须更新 *objective()* 函数来解包 XGBoost 模型的超参数，对其进行配置，然后评估平均分类准确率。

```py
# objective function
def objective(X, y, cfg):
	# unpack config
	lrate, n_tree, subsam, depth = cfg
	# define model
	model = XGBClassifier(learning_rate=lrate, n_estimators=n_tree, subsample=subsam, max_depth=depth)
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# calculate mean accuracy
	result = mean(scores)
	return result
```

接下来，我们需要定义用于在搜索空间中迈出一步的 *step()* 函数。

每个超参数都是相当不同的范围，因此，我们将分别为每个超参数定义步长(分布的标准偏差)。为了简单起见，我们还将按行定义步长，而不是作为函数的参数。

树的数量和深度是整数，因此阶梯值是四舍五入的。

所选择的步长是任意的，是经过反复试验后选择的。

更新后的阶跃函数如下所示。

```py
# take a step in the search space
def step(cfg):
	# unpack config
	lrate, n_tree, subsam, depth = cfg
	# learning rate
	lrate = lrate + randn() * 0.01
	if lrate <= 0.0:
		lrate = 1e-8
	if lrate > 1:
		lrate = 1.0
	# number of trees
	n_tree = round(n_tree + randn() * 50)
	if n_tree <= 0.0:
		n_tree = 1
	# subsample percentage
	subsam = subsam + randn() * 0.1
	if subsam <= 0.0:
		subsam = 1e-8
	if subsam > 1:
		subsam = 1.0
	# max tree depth
	depth = round(depth + randn() * 7)
	if depth <= 1:
		depth = 1
	# return new config
	return [lrate, n_tree, subsam, depth]
```

最后，必须更新*爬山()*算法，以定义具有适当值的初始解。

在这种情况下，我们将用合理的缺省值定义初始解，匹配缺省超参数，或者接近它们。

```py
...
# starting point for the search
solution = step([0.1, 100, 1.0, 7])
```

将这些联系在一起，下面列出了使用随机爬山算法手动调整 XGBoost 算法的超参数的完整示例。

```py
# xgboost manual hyperparameter optimization for binary classification
from numpy import mean
from numpy.random import randn
from numpy.random import rand
from numpy.random import randint
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

# objective function
def objective(X, y, cfg):
	# unpack config
	lrate, n_tree, subsam, depth = cfg
	# define model
	model = XGBClassifier(learning_rate=lrate, n_estimators=n_tree, subsample=subsam, max_depth=depth)
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# calculate mean accuracy
	result = mean(scores)
	return result

# take a step in the search space
def step(cfg):
	# unpack config
	lrate, n_tree, subsam, depth = cfg
	# learning rate
	lrate = lrate + randn() * 0.01
	if lrate <= 0.0:
		lrate = 1e-8
	if lrate > 1:
		lrate = 1.0
	# number of trees
	n_tree = round(n_tree + randn() * 50)
	if n_tree <= 0.0:
		n_tree = 1
	# subsample percentage
	subsam = subsam + randn() * 0.1
	if subsam <= 0.0:
		subsam = 1e-8
	if subsam > 1:
		subsam = 1.0
	# max tree depth
	depth = round(depth + randn() * 7)
	if depth <= 1:
		depth = 1
	# return new config
	return [lrate, n_tree, subsam, depth]

# hill climbing local search algorithm
def hillclimbing(X, y, objective, n_iter):
	# starting point for the search
	solution = step([0.1, 100, 1.0, 7])
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = step(solution)
		# evaluate candidate point
		candidate_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidate_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidate_eval
			# report progress
			print('>%d, cfg=[%s] %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# define the total iterations
n_iter = 200
# perform the hill climbing search
cfg, score = hillclimbing(X, y, objective, n_iter)
print('Done!')
print('cfg=[%s]: Mean Accuracy: %f' % (cfg, score))
```

每次在搜索过程中看到改进时，运行示例都会报告配置和结果。运行结束时，会报告最佳配置和结果。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到最佳结果涉及使用大约 0.02 的学习率、52 棵树、大约 50%的子采样率和 53 个级别的大深度。

这种配置的平均准确率约为 87.3%，优于默认配置的 84.9%。

**能不能得到更好的结果？**
在下面的评论里告诉我。

```py
>0, cfg=[[0.1058242692126418, 67, 0.9228490731610172, 12]] 0.85933
>1, cfg=[[0.11060813799692253, 51, 0.859353656735739, 13]] 0.86100
>4, cfg=[[0.11890247679234153, 58, 0.7135275461723894, 12]] 0.86167
>5, cfg=[[0.10226257987735601, 61, 0.6086462443373852, 17]] 0.86400
>15, cfg=[[0.11176962034280596, 106, 0.5592742266405146, 13]] 0.86500
>19, cfg=[[0.09493587069112454, 153, 0.5049124222437619, 34]] 0.86533
>23, cfg=[[0.08516531024154426, 88, 0.5895201311518876, 31]] 0.86733
>46, cfg=[[0.10092590898175327, 32, 0.5982811365027455, 30]] 0.86867
>75, cfg=[[0.099469211050998, 20, 0.36372573610040404, 32]] 0.86900
>96, cfg=[[0.09021536590375884, 38, 0.4725379807796971, 20]] 0.86900
>100, cfg=[[0.08979482274655906, 65, 0.3697395430835758, 14]] 0.87000
>110, cfg=[[0.06792737273465625, 89, 0.33827505722318224, 17]] 0.87000
>118, cfg=[[0.05544969684589669, 72, 0.2989721608535262, 23]] 0.87200
>122, cfg=[[0.050102976159097, 128, 0.2043203965148931, 24]] 0.87200
>123, cfg=[[0.031493266763680444, 120, 0.2998819062922256, 30]] 0.87333
>128, cfg=[[0.023324201169625292, 84, 0.4017169945431015, 42]] 0.87333
>140, cfg=[[0.020224220443108752, 52, 0.5088096815056933, 53]] 0.87367
Done!
cfg=[[0.020224220443108752, 52, 0.5088096815056933, 53]]: Mean Accuracy: 0.873667
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [随机搜索和网格搜索的超参数优化](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/)
*   [如何配置梯度提升算法](https://machinelearningmastery.com/configure-gradient-boosting-algorithm/)
*   [如何在 Python 中从零开始实现感知器算法](https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/)

### 蜜蜂

*   [sklearn . datasets . make _ classification APIS](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)。
*   [sklearn . metrics . accuracy _ score APIS](https://Sklearn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)。
*   num py . random . rand API。
*   [sklearn.linear_model。感知器 API](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html) 。

### 文章

*   [感知，维基百科](https://en.wikipedia.org/wiki/Perceptron)。
*   [XGBoost，维基百科](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)。

## 摘要

在本教程中，您发现了如何手动优化机器学习算法的超参数。

具体来说，您了解到:

*   随机优化算法可以代替网格和随机搜索用于超参数优化。
*   如何使用随机爬山算法来调整感知器算法的超参数。
*   如何手动优化 XGBoost 梯度提升算法的超参数？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。