# Python 中的动态分类器选择集成

> 原文：<https://machinelearningmastery.com/dynamic-classifier-selection-in-python/>

最后更新于 2021 年 4 月 27 日

**动态分类器选择**是一种用于分类预测建模的集成学习算法。

该技术包括在训练数据集上拟合多个机器学习模型，然后基于要预测的示例的具体细节，选择在进行预测时预期表现最佳的模型。

这可以通过使用 k-最近邻模型来定位训练数据集中最接近要预测的新示例的示例，评估该邻域上池中的所有模型，并使用在该邻域上表现最好的模型来对新示例进行预测来实现。

因此，动态分类器选择通常可以比池中的任何单个模型表现得更好，并且提供了对来自多个模型的预测进行平均的替代方案，正如在其他集成算法中的情况一样。

在本教程中，您将发现如何在 Python 中开发动态分类器选择集成。

完成本教程后，您将知道:

*   动态分类器选择算法从许多模型中选择一个来为每个新的例子做出预测。
*   如何使用 Sklearn API 为分类任务开发和评估动态分类器选择模型。
*   如何探索动态分类器选择模型超参数对分类准确率的影响？

**用我的新书[Python 集成学习算法](https://machinelearningmastery.com/ensemble-learning-algorithms-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Develop Dynamic Classifier Selection in Python](img/6ec1a1a597fceb9c4ce404acc0e28166.png)

如何在 Python 中开发动态分类器选择
图片由 [Jean 和 Fred](https://www.flickr.com/photos/jean_hort/37727673731/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  动态分类器选择
2.  基于 Sklearn 的动态分类器选择
    1.  具有整体局部准确率的集散控制系统
    2.  具有本地等级准确率的分布式控制系统
3.  集散控制系统的超参数整定
    1.  在 k-最近邻中探索 k
    2.  探索分类器池的算法

## 动态分类器选择

多分类器系统是指机器学习算法的一个领域，它使用多个模型来解决分类预测建模问题。

这包括熟悉的技术，如[一对其余，一对所有](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)，和输出纠错码技术。它还包括更通用的技术，为每个需要预测的新示例选择一个动态使用的模型。

> 目前有几种方法被用来构建一个监控系统[……]最有前途的监控系统方法之一是动态选择，即根据每个要分类的新样本动态选择基本分类器。

——[动态分类器选择:最新进展与展望](https://www.sciencedirect.com/science/article/pii/S1566253517304074)，2018。

有关这些类型的多分类器系统的更多信息，请参见教程:

*   [如何使用一比一休息和一比一进行多类分类](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

这些方法通常被称为:动态分类器选择，简称 DCS。

*   **动态分类器选择**:从众多训练好的模型中选择一个，根据输入的具体细节进行预测的算法。

考虑到分布式控制系统中使用了多个模型，它被认为是一种集成学习技术。

动态分类器选择算法通常涉及以某种方式划分输入特征空间，并分配特定的模型来负责为每个划分做出预测。有各种不同的分布式控制系统算法，研究工作主要集中在如何评估和分配分类器到输入空间的特定区域。

> 在训练多个个体学习器之后，分布式控制系统为每个测试实例动态选择一个学习器。[……]分布式控制系统通过使用单个学习器进行预测。

—第 93 页，[集成方法:基础和算法](https://amzn.to/32L1yWD)，2012。

早期流行的方法包括首先在训练数据集上拟合一组小的、多样的分类模型。当需要预测时，首先使用 k-最近邻(kNN)算法从训练数据集中找到与该示例匹配的 k 个最相似的示例。然后，在 k 个训练示例的邻居上评估模型中的每个先前匹配的分类器，并且选择表现最好的分类器来对新示例进行预测。

这种方法被简称为“*动态分类器选择局部准确率*”或简称为 DCS-LA，由 Kevin Woods 等人在 1997 年发表的题为“[使用局部准确率估计的多分类器组合](https://ieeexplore.ieee.org/abstract/document/588027)”的论文中进行了描述

> 其基本思想是在未知测试样本周围的特征空间局部区域内估计每个分类器的准确率，然后使用局部准确率最高的分类器的判决。

——[使用局部准确率估计的多个分类器的组合](https://ieeexplore.ieee.org/abstract/document/588027)，1997。

作者描述了选择单个分类器模型来对给定输入示例进行预测的两种方法，它们是:

*   **局部准确率**，通常称为 LA 或整体局部准确率(OLA)。
*   **类准确率**，通常称为 CA 或局部类准确率(LCA)。

**局部准确率** (OLA)涉及在 k 个训练示例的邻域上评估每个模型的分类准确率。然后选择在该邻域中表现最好的模型来为新示例进行预测。

> 每个分类器的 OLA 被计算为局部区域中样本的正确识别的百分比。

——[量词的动态选择——综述](https://www.sciencedirect.com/science/article/abs/pii/S0031320314001885)，2014。

**类准确率** (LCA)包括使用每个模型对新示例进行预测，并记录预测的类。然后，评估 k 个训练示例的邻居上的每个模型的准确性，并且选择对于它在新示例上预测的类具有最佳技能的模型，并且返回它的预测。

> 每个基本分类器的生命周期评估被估计为局部区域内正确分类的百分比，但是只考虑那些分类器给出的类与未知模式给出的类相同的例子。

——[量词的动态选择——综述](https://www.sciencedirect.com/science/article/abs/pii/S0031320314001885)，2014。

在这两种情况下，如果所有拟合模型对新的输入示例进行相同的预测，则直接返回预测。

现在我们已经熟悉了 DCS 和 DCS-LA 算法，让我们看看如何在我们自己的分类预测建模项目中使用它。

## 基于 Sklearn 的动态分类器选择

动态集成选择库(简称 DESlib)是一个开源 Python 库，它提供了许多不同的动态分类器选择算法的实现。

DESlib 是一个易于使用的集成学习库，专注于实现动态分类器和集成选择的最新技术。

*   [动态选择库项目，GitHub](https://github.com/Sklearn-contrib/DESlib) 。

首先，我们可以使用 pip 包管理器来安装 DESlib 库。

```py
sudo pip install deslib
```

安装后，我们可以通过加载库并打印已安装的版本来确认库安装正确并准备好使用。

```py
# check deslib version
import deslib
print(deslib.__version__)
```

运行该脚本将打印您安装的 DESlib 库的版本。

您的版本应该相同或更高。如果没有，您必须升级您的 DESlib 库版本。

```py
0.3
```

DESlib 分别通过 [OLA](https://deslib.readthedocs.io/en/latest/modules/dcs/ola.html) 和 [LCA](https://deslib.readthedocs.io/en/latest/modules/dcs/lca.html) 类为每个分类器选择技术提供了分布式控制系统-学习算法的实现。

每个类都可以直接用作 Sklearn 模型，允许直接使用全套 Sklearn 数据准备、建模管道和模型评估技术。

这两个类都使用 k 最近邻算法来选择默认值为 *k=7* 的邻居。

决策树的[自举聚合](https://machinelearningmastery.com/bagging-ensemble-with-python/)(装袋)集成被用作为默认进行的每个分类考虑的分类器模型池，尽管这可以通过将“ *pool_classifiers* ”设置为模型列表来改变。

我们可以使用 [make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建一个包含 10，000 个示例和 20 个输入特征的合成二进制分类问题。

```py
# synthetic binary classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例将创建数据集并总结输入和输出组件的形状。

```py
(10000, 20) (10000,)
```

现在我们已经熟悉了 DESlib API，让我们来看看如何使用每个 DCS-LA 算法。

### 具有整体局部准确率的集散控制系统

我们可以在合成数据集上使用整体局部准确率来评估分布式控制系统-洛杉矶模型。

在这种情况下，我们将使用默认的模型超参数，包括袋装决策树作为分类器模型池，以及一个 *k=7* 用于在进行预测时选择局部邻域。

我们将使用三次重复和 10 次折叠的重复分层 k 折叠交叉验证来评估模型。我们将报告所有重复和折叠的模型准确率的平均值和标准偏差。

下面列出了完整的示例。

```py
# evaluate dynamic classifier selection DCS-LA with overall local accuracy
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from deslib.dcs.ola import OLA
# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the model
model = OLA()
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例会报告模型的均值和标准差准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到带有 OLA 和默认超参数的 DCS-LA 达到了大约 88.3%的分类准确率。

```py
Mean Accuracy: 0.883 (0.012)
```

我们也可以用带有 OLA 的 DCS-LA 模型作为最终模型，进行分类预测。

首先，模型适合所有可用数据，然后可以调用 *predict()* 函数对新数据进行预测。

下面的示例在我们的二进制类别数据集上演示了这一点。

```py
# make a prediction with DCS-LA using overall local accuracy
from sklearn.datasets import make_classification
from deslib.dcs.ola import OLA
# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the model
model = OLA()
# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
row = [0.2929949,-4.21223056,-1.288332,-2.17849815,-0.64527665,2.58097719,0.28422388,-7.1827928,-1.91211104,2.73729512,0.81395695,3.96973717,-2.66939799,3.34692332,4.19791821,0.99990998,-0.30201875,-4.43170633,-2.82646737,0.44916808]
yhat = model.predict([row])
print('Predicted Class: %d' % yhat[0])
```

运行该示例使 DCS-LA 与 OLA 模型在整个数据集上匹配，然后用于对新的数据行进行预测，就像我们在应用程序中使用该模型时可能做的那样。

```py
Predicted Class: 0
```

现在我们已经熟悉了将 DCS-LA 与 OLA 一起使用，让我们来看看 LCA 方法。

### 具有本地等级准确率的分布式控制系统

我们可以在合成数据集上使用局部类准确率来评估分布式控制系统-人工智能模型。

在这种情况下，我们将使用默认的模型超参数，包括袋装决策树作为分类器模型池，以及一个 *k=7* 用于在进行预测时选择局部邻域。

我们将使用三次重复和 10 次折叠的重复分层 k 折叠交叉验证来评估模型。我们将报告所有重复和折叠的模型准确率的平均值和标准偏差。

下面列出了完整的示例。

```py
# evaluate dynamic classifier selection DCS-LA using local class accuracy
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from deslib.dcs.lca import LCA
# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the model
model = LCA()
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例会报告模型的均值和标准差准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到带有生命周期评估和默认超参数的分布式控制系统达到了大约 92.2%的分类准确率。

```py
Mean Accuracy: 0.922 (0.007)
```

我们也可以使用带有生命周期评估的分布式控制系统-洛杉矶模型作为最终模型，并进行分类预测。

首先，模型适合所有可用数据，然后可以调用 *predict()* 函数对新数据进行预测。

下面的示例在我们的二进制类别数据集上演示了这一点。

```py
# make a prediction with DCS-LA using local class accuracy
from sklearn.datasets import make_classification
from deslib.dcs.lca import LCA
# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the model
model = LCA()
# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
row = [0.2929949,-4.21223056,-1.288332,-2.17849815,-0.64527665,2.58097719,0.28422388,-7.1827928,-1.91211104,2.73729512,0.81395695,3.96973717,-2.66939799,3.34692332,4.19791821,0.99990998,-0.30201875,-4.43170633,-2.82646737,0.44916808]
yhat = model.predict([row])
print('Predicted Class: %d' % yhat[0])
```

运行该示例使 DCS-LA 与 LCA 模型在整个数据集上匹配，然后用于对新的数据行进行预测，就像我们在应用程序中使用该模型时可能做的那样。

```py
Predicted Class: 0
```

现在，我们已经熟悉了使用 Sklearn API 来评估和使用 DCS-LA 模型，接下来让我们看看如何配置模型。

## 集散控制系统的超参数整定

在本节中，我们将仔细研究一些您应该考虑为分布式控制系统-线性模型进行调整的超参数，以及它们对模型表现的影响。

对于 DCS-LA，我们可以查看许多超参数，尽管在这种情况下，我们将查看在模型的局部评估中使用的 k 最近邻模型中的 k 值，以及如何使用自定义的分类器池。

我们将使用带有 OLA 的 DCS-LA 作为这些实验的基础，尽管具体方法的选择是任意的。

### 在 k-最近邻中探索 k

k-最近邻算法的配置对 DCS-LA 模型至关重要，因为它定义了考虑选择每个分类器的邻域范围。

*k* 值控制邻域的大小，将其设置为适合数据集的值非常重要，特别是特征空间中样本的密度。太小的值意味着训练集中的相关例子可能被排除在邻域之外，而太大的值可能意味着信号被太多的例子冲掉。

下面的例子探索了 k 值从 2 到 21 的带有 OLA 的 DCS-LA 的分类准确率。

```py
# explore k in knn for DCS-LA with overall local accuracy
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from deslib.dcs.ola import OLA
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of models to evaluate
def get_models():
	models = dict()
	for n in range(2,22):
		models[str(n)] = OLA(k=n)
	return models

# evaluate a give model using cross-validation
def evaluate_model(model):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

运行该示例首先报告每个配置的邻域大小的平均准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到准确率会随着邻域大小的增加而增加，可能会增加到 k=13 或 k=14，在这种情况下准确率会趋于平稳。

```py
>2 0.873 (0.009)
>3 0.874 (0.013)
>4 0.880 (0.009)
>5 0.881 (0.009)
>6 0.883 (0.010)
>7 0.883 (0.011)
>8 0.884 (0.012)
>9 0.883 (0.010)
>10 0.886 (0.012)
>11 0.886 (0.011)
>12 0.885 (0.010)
>13 0.888 (0.010)
>14 0.886 (0.009)
>15 0.889 (0.010)
>16 0.885 (0.012)
>17 0.888 (0.009)
>18 0.886 (0.010)
>19 0.889 (0.012)
>20 0.889 (0.011)
>21 0.886 (0.011)
```

为每个配置的邻域大小的准确率分数分布创建一个方框和须图。

我们可以看到模型表现和 k 值在达到平稳之前增加的总体趋势。

![Box and Whisker Plots of Accuracy Distributions for k Values in DCS-LA With OLA](img/04e7f78300bcab707b2fca3ca5686c5e.png)

分布式控制系统中 k 值准确率分布的盒式和触须图

### 探索分类器池的算法

分布式控制系统-人工智能池中使用的算法选择是另一个重要的超参数。

默认情况下，使用袋装决策树，因为它已被证明是一系列分类任务的有效方法。然而，可以考虑定制分类器池。

这需要首先定义一个分类器模型列表，以便在训练数据集中使用和拟合每个分类器模型。不幸的是，这意味着 Sklearn 中的自动 k-fold 交叉验证模型评估方法不能用于这种情况。相反，我们将使用训练-测试分割，这样我们就可以在训练数据集上手动调整分类器池。

然后，可以通过“*池分类器*”参数将拟合分类器列表指定给 OLA(或 LCA)类。在这种情况下，我们将使用包含逻辑回归、决策树和朴素贝叶斯分类器的池。

下面列出了使用 OLA 和合成数据集上的一组自定义分类器评估 DCS-LA 的完整示例。

```py
# evaluate DCS-LA using OLA with a custom pool of algorithms
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deslib.dcs.ola import OLA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# define classifiers to use in the pool
classifiers = [
	LogisticRegression(),
	DecisionTreeClassifier(),
	GaussianNB()]
# fit each classifier on the training set
for c in classifiers:
	c.fit(X_train, y_train)
# define the DCS-LA model
model = OLA(pool_classifiers=classifiers)
# fit the model
model.fit(X_train, y_train)
# make predictions on the test set
yhat = model.predict(X_test)
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % (score))
```

运行该示例首先报告带有自定义分类器池的模型的平均准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型达到了大约 91.2%的准确率。

```py
Accuracy: 0.913
```

为了采用分布式控制系统模型，它必须比任何贡献模型表现得更好。否则，我们只会使用表现更好的贡献模型。

我们可以通过评估测试集中每个有贡献的分类器的表现来检查这一点。

```py
...
# evaluate contributing models
for c in classifiers:
	yhat = c.predict(X_test)
	score = accuracy_score(y_test, yhat)
	print('>%s: %.3f' % (c.__class__.__name__, score))
```

下面列出了 DCS-LA 的更新示例，它具有一个定制的分类器池，这些分类器也是独立评估的。

```py
# evaluate DCS-LA using OLA with a custom pool of algorithms
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deslib.dcs.ola import OLA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# define classifiers to use in the pool
classifiers = [
	LogisticRegression(),
	DecisionTreeClassifier(),
	GaussianNB()]
# fit each classifier on the training set
for c in classifiers:
	c.fit(X_train, y_train)
# define the DCS-LA model
model = OLA(pool_classifiers=classifiers)
# fit the model
model.fit(X_train, y_train)
# make predictions on the test set
yhat = model.predict(X_test)
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % (score))
# evaluate contributing models
for c in classifiers:
	yhat = c.predict(X_test)
	score = accuracy_score(y_test, yhat)
	print('>%s: %.3f' % (c.__class__.__name__, score))
```

运行该示例首先报告带有自定义分类器池的模型的平均准确率和每个贡献模型的准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以再次看到，分布式控制系统-洛杉矶实现了约 91.3%的准确性，这优于任何贡献模型。

```py
Accuracy: 0.913
>LogisticRegression: 0.878
>DecisionTreeClassifier: 0.884
>GaussianNB: 0.873
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 相关教程

*   [如何使用一比一休息和一比一进行多类分类](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

### 报纸

*   [使用局部准确率估计的多个分类器的组合](https://ieeexplore.ieee.org/abstract/document/588027)，1997。
*   [量词的动态选择——综述](https://www.sciencedirect.com/science/article/abs/pii/S0031320314001885)，2014。
*   [动态分类器选择:最新进展和展望](https://www.sciencedirect.com/science/article/pii/S1566253517304074)，2018。

### 书

*   [集成方法:基础与算法](https://amzn.to/32L1yWD)，2012。

### 蜜蜂

*   [动态选择库项目，GitHub](https://github.com/Sklearn-contrib/DESlib) 。
*   [脱 lib API 文件](https://deslib.readthedocs.io/en/latest/api.html)。

## 摘要

在本教程中，您发现了如何在 Python 中开发动态分类器选择集成。

具体来说，您了解到:

*   动态分类器选择算法从许多模型中选择一个来为每个新的例子做出预测。
*   如何使用 Sklearn API 为分类任务开发和评估动态分类器选择模型。
*   如何探索动态分类器选择模型超参数对分类准确率的影响？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。