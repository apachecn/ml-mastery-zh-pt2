# 执行数据准备时如何避免数据泄露

> 原文：<https://machinelearningmastery.com/data-preparation-without-data-leakage/>

最后更新于 2020 年 8 月 17 日

数据准备是将原始数据转换成适合建模的形式的过程。

准备数据的简单方法是在评估模型的表现之前对整个数据集应用转换。这导致了一个被称为**数据泄漏**的问题，其中，保持测试集的知识泄漏到用于训练模型的数据集中。在对新数据进行预测时，这可能导致对模型表现的不正确估计。

为了避免数据泄露，需要仔细应用数据准备技术，这取决于所使用的模型评估方案，如列车测试分割或 k 倍交叉验证。

在本教程中，您将发现如何在评估机器学习模型时避免数据准备过程中的数据泄漏。

完成本教程后，您将知道:

*   将数据准备方法天真地应用于整个数据集会导致数据泄漏，从而导致对模型表现的不正确估计。
*   数据准备必须仅在训练集上准备，以避免数据泄露。
*   如何在 Python 中实现训练-测试拆分和 k-fold 交叉验证的无数据泄露的数据准备。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Avoid Data Leakage When Performing Data Preparation](img/dce1643c6d1cd9cbc0a6f3d51e059e6f.png)

执行数据准备时如何避免数据泄露
图片由 [kuhnmi](https://flickr.com/photos/31176607@N05/43531214070/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  原始数据准备的问题
2.  用训练集和测试集准备数据
    1.  利用原始数据准备进行列车测试评估
    2.  正确准备数据的列车测试评估
3.  k 倍交叉验证的数据准备
    1.  原始数据准备的交叉验证评估
    2.  正确准备数据的交叉验证评估

## 原始数据准备的问题

数据准备技术应用于数据事务的方式。

一种常见的方法是首先对整个数据集应用一个或多个转换。然后将数据集分割成训练集和测试集，或者使用 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)来拟合和评估机器学习模型。

*   1.准备数据集
*   2.分割数据
*   3.评估模型

虽然这是一种常见的方法，但在大多数情况下，这是危险的错误。

在拆分数据进行模型评估之前应用数据准备技术的问题是，它可能导致数据泄露，进而可能导致对模型在该问题上的表现的不正确估计。

[数据泄露](https://machinelearningmastery.com/data-leakage-machine-learning/)指的是关于保持数据集的信息(例如测试或验证数据集)对训练数据集中的模型可用的问题。这种泄漏通常很小且很微妙，但会对表现产生显著影响。

> ……泄漏意味着信息被透露给模型，使其具有不切实际的优势来做出更好的预测。当测试数据泄露到训练集中时，或者当来自未来的数据泄露到过去时，就会发生这种情况。任何时候，当一个模型在生产中实时进行预测时，它被给予了它不应该得到的信息，就会有泄漏。

—第 93 页，[机器学习的特征工程](https://amzn.to/2zZOQXN)，2018。

我们通过对整个数据集应用数据准备技术来获得数据泄漏。

这不是一种直接的数据泄漏，我们将在测试数据集上训练模型。相反，这是一种间接类型的数据泄漏，其中在汇总统计中捕获的关于测试数据集的一些知识在训练期间对模型可用。这可能会使发现数据泄漏变得更加困难，尤其是对于初学者。

> 重采样的另一个方面与信息泄漏的概念有关，这是在训练过程中使用测试集数据(直接或间接)的地方。这可能会导致过于乐观的结果，不会在未来的数据点上复制，并且会以微妙的方式出现。

—第 55 页，[特征工程与选择](https://amzn.to/2Yvcupn)，2019 年。

例如，考虑我们想要规范化数据的情况，也就是将输入变量缩放到 0-1 的范围。

当我们标准化输入变量时，这要求我们首先计算每个变量的最小值和最大值，然后使用这些值来缩放变量。然后将数据集分为训练数据集和测试数据集，但是训练数据集中的示例对测试数据集中的数据有所了解；他们已经按全局最小值和最大值进行了缩放，因此他们比他们应该的更了解变量的全局分布。

几乎所有的数据准备技术都会出现相同类型的泄漏；例如，标准化估计来自域的平均值和标准偏差值，以便缩放变量；即使是使用模型或汇总统计数据估计缺失值的模型，也会利用整个数据集来填充训练数据集中的值。

**解决方法很简单**。

数据准备必须只适合训练数据集。也就是说，为数据准备过程准备的任何系数或模型必须只使用训练数据集中的数据行。

一旦适合，数据准备算法或模型就可以应用于训练数据集和测试数据集。

*   1.拆分数据。
*   2.在训练数据集上拟合数据准备。
*   3.将数据准备应用于训练和测试数据集。
*   4.评估模型。

更一般地说，必须只在训练数据集上准备整个建模管道，以避免数据泄漏。这可能包括数据转换，但也包括其他技术，如特征选择、降维、特征工程等。这意味着所谓的“模型评估”真的应该叫做“建模管道评估”。

> 为了使任何重采样方案产生推广到新数据的表现估计，它必须包含建模过程中可能显著影响模型有效性的所有步骤。

—第 54-55 页，[特征工程与选择](https://amzn.to/2Yvcupn)，2019。

既然我们已经熟悉了如何应用数据准备来避免数据泄漏，那么让我们来看看一些工作示例。

## 用训练集和测试集准备数据

在本节中，我们将在输入变量已经标准化的合成二进制类别数据集上使用训练和测试集来评估逻辑回归模型。

首先，让我们定义我们的合成数据集。

我们将使用 [make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建包含 1000 行数据和 20 个数字输入特征的数据集。下面的示例创建数据集并总结输入和输出变量数组的形状。

```py
# test classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例将创建数据集，并确认数据集的输入部分对于 20 个输入变量有 1，000 行和 20 列，输出变量有 1，000 个示例来匹配 1，000 行输入数据，每行一个值。

```py
(1000, 20) (1000,)
```

接下来，我们可以在缩放的数据集上评估我们的模型，从他们幼稚或不正确的方法开始。

### 利用原始数据准备进行列车测试评估

幼稚的方法包括首先应用数据准备方法，然后在最终评估模型之前拆分数据。

我们可以使用[最小最大缩放器类](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)来归一化输入变量，该类首先用默认配置定义，将数据缩放到 0-1 的范围，然后调用 *fit_transform()* 函数来拟合数据集上的变换，并在一个步骤中将其应用于数据集。结果是输入变量的规范化版本，其中数组中的每一列都被单独规范化(例如，计算出它自己的最小值和最大值)。

```py
...
# standardize the dataset
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
```

接下来，我们可以使用 [train_test_split()函数](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)将数据集分割成训练集和测试集。我们将 67%用于训练集，33%用于测试集。

```py
...
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

然后，我们可以通过[logisticreduction 类](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)定义我们的逻辑回归算法，使用默认配置，并使其适合训练数据集。

```py
...
# fit the model
model = LogisticRegression()
model.fit(X_train, y_train)
```

然后，拟合模型可以对测试集的输入数据进行预测，我们可以将预测值与期望值进行比较，并计算分类准确率分数。

```py
...
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % (accuracy*100))
```

将这些联系在一起，完整的示例如下所示。

```py
# naive approach to normalizing the data before splitting the data and evaluating the model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# standardize the dataset
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
model = LogisticRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % (accuracy*100))
```

运行该示例将数据标准化，将数据分成训练集和测试集，然后拟合和评估模型。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型的估计值约为 84.848%。

```py
Accuracy: 84.848
```

鉴于我们知道存在数据泄露，我们知道这种对模型准确率的估计是错误的。

接下来，让我们探讨如何正确准备数据以避免数据泄漏。

### 正确准备数据的列车测试评估

使用训练-测试分割评估执行数据准备的正确方法是将数据准备适合训练集，然后将转换应用于训练集和测试集。

这要求我们首先将数据分成训练集和测试集。

```py
...
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

然后我们可以定义*最小最大缩放器*并在训练集上调用 *fit()* 函数，然后在训练集和测试集上应用*变换()*函数来创建每个数据集的规范化版本。

```py
...
# define the scaler
scaler = MinMaxScaler()
# fit on the training dataset
scaler.fit(X_train)
# scale the training dataset
X_train = scaler.transform(X_train)
# scale the test dataset
X_test = scaler.transform(X_test)
```

这避免了数据泄漏，因为每个输入变量的最小值和最大值的计算仅使用训练数据集( *X_train* )而不是整个数据集( *X* )来计算。

然后可以像以前一样评估模型。

将这些联系在一起，完整的示例如下所示。

```py
# correct approach for normalizing the data after the data is split before the model is evaluated
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# define the scaler
scaler = MinMaxScaler()
# fit on the training dataset
scaler.fit(X_train)
# scale the training dataset
X_train = scaler.transform(X_train)
# scale the test dataset
X_test = scaler.transform(X_test)
# fit the model
model = LogisticRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % (accuracy*100))
```

运行该示例将数据分为训练集和测试集，正确地标准化数据，然后拟合和评估模型。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型的估计值约为 85.455%，这比上一节中具有数据泄漏的估计值更准确，后者达到了 84.848%的准确率。

我们预计数据泄露会导致对模型表现的错误估计。我们希望这是一个乐观的估计，数据泄漏会带来更好的表现，尽管在这种情况下，我们可以看到数据泄漏会导致表现略微下降。这可能是因为预测任务的难度。

```py
Accuracy: 85.455
```

## k 倍交叉验证的数据准备

在本节中，我们将在输入变量已经标准化的合成二进制类别数据集上使用 k 倍交叉验证来评估逻辑回归模型。

您可能还记得 [K 折交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)涉及将数据集拆分为 k 个不重叠的行组。该模型然后在除一个组之外的所有组上训练以形成训练数据集，然后在保持折叠上评估。重复此过程，以便每个折叠都有机会用作保持测试集。最后，报告所有评估的平均绩效。

k-fold 交叉验证程序通常比训练测试分割给出更可靠的模型表现估计，尽管考虑到模型的重复拟合和评估，它的计算成本更高。

让我们首先来看看带有 k 倍交叉验证的幼稚数据准备。

### 原始数据准备的交叉验证评估

交叉验证的原始数据准备包括首先应用数据转换，然后使用交叉验证过程。

我们将使用上一节中准备的合成数据集，并直接对数据进行规范化。

```py
...
# standardize the dataset
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
```

必须首先定义 k 倍交叉验证程序。我们将使用重复的分层 10 倍交叉验证，这是分类的最佳实践。重复意味着整个交叉验证过程重复多次，在本例中为三次。分层意味着每组行将具有来自每个类的示例作为整个数据集的相对组成。我们将使用 *k=10* 或 10 倍交叉验证。

这可以通过使用[repeated stratifiedfold](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)来实现，它可以配置为三次重复和 10 次折叠，然后使用 [cross_val_score()函数](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)来执行该过程，传入定义的模型、交叉验证对象和度量来计算准确率，在这种情况下。

```py
...
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model using cross-validation
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

然后，我们可以报告所有重复和折叠的平均准确率。

将所有这些结合起来，下面列出了使用数据准备和数据泄漏来评估模型的交叉验证的完整示例。

```py
# naive data preparation for model evaluation with k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# standardize the dataset
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# define the model
model = LogisticRegression()
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model using cross-validation
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores)*100, std(scores)*100))
```

运行该示例首先标准化数据，然后使用重复的分层交叉验证来评估模型。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型达到了大约 85.300%的估计准确率，考虑到数据准备过程允许的数据泄漏，我们知道这是不正确的。

```py
Accuracy: 85.300 (3.607)
```

接下来，让我们看看如何通过交叉验证评估模型并避免数据泄漏。

### 正确准备数据的交叉验证评估

当使用交叉验证时，没有数据泄漏的数据准备工作稍微更具挑战性。

它要求在训练集上准备数据准备方法，并将其应用于交叉验证程序中的训练集和测试集，例如行的折叠组。

我们可以通过定义一个建模管道来实现这一点，该管道定义了一系列要执行的数据准备步骤，并在模型中结束以进行拟合和评估。

> 为了提供一个可靠的方法，我们应该限制自己开发预处理技术的列表，仅在训练数据点存在的情况下估计它们，然后将这些技术应用于未来的数据(包括测试集)。

—第 55 页，[特征工程与选择](https://amzn.to/2Yvcupn)，2019 年。

评估过程从简单且不正确地评估模型转变为正确地评估数据准备的整个管道和作为单个原子单元的模型。

这可以使用[管道类](https://Sklearn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)来实现。

这个类列出了定义管道的步骤。列表中的每个步骤都是一个包含两个元素的元组。第一个元素是步骤的名称(字符串)，第二个元素是步骤的配置对象，例如转换或模型。该模型仅作为最后一步被支持，尽管我们可以在序列中拥有任意多的变换。

```py
...
# define the pipeline
steps = list()
steps.append(('scaler', MinMaxScaler()))
steps.append(('model', LogisticRegression()))
pipeline = Pipeline(steps=steps)
```

然后，我们可以将配置的对象传递给 *cross_val_score()* 函数进行评估。

```py
...
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model using cross-validation
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

将这些结合在一起，下面列出了使用交叉验证时正确执行数据准备而不泄漏数据的完整示例。

```py
# correct data preparation for model evaluation with k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the pipeline
steps = list()
steps.append(('scaler', MinMaxScaler()))
steps.append(('model', LogisticRegression()))
pipeline = Pipeline(steps=steps)
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model using cross-validation
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores)*100, std(scores)*100))
```

运行该示例可以在评估过程的交叉验证折叠中正确地标准化数据，以避免数据泄漏。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，该模型的估计准确率约为 85.433%，而数据泄露的方法达到了约 85.300%的准确率。

与上一节中的训练测试示例一样，当我们的直觉认为数据泄漏通常会导致对模型表现的乐观估计时，消除数据泄漏会使表现略有提高。尽管如此，这些示例清楚地表明，数据泄漏确实会影响模型表现的估计，以及如何在数据拆分后通过正确执行数据准备来纠正数据泄漏。

```py
Accuracy: 85.433 (3.471)
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [如何为机器学习准备数据](https://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/)
*   [应用机器学习过程](https://machinelearningmastery.com/process-for-working-through-machine-learning-problems/)
*   [机器学习中的数据泄露](https://machinelearningmastery.com/data-leakage-machine-learning/)

### 书

*   [特征工程和选择:预测模型的实用方法](https://amzn.to/2VLgpex)，2019。
*   [应用预测建模](https://amzn.to/2VMhnat)，2013。
*   [数据挖掘:实用机器学习工具与技术](https://amzn.to/2Kk6tn0)，2016。
*   [机器学习的特征工程](https://amzn.to/2zZOQXN)，2018。

### 蜜蜂

*   [sklearn . datasets . make _ classification API](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)。
*   [硬化。预处理。MinMaxScaler API](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) 。
*   [sklearn . model _ selection . train _ test _ split API](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)。
*   [sklearn.linear_model。物流配送应用编程接口](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。
*   [sklearn.model_selection。重复的策略应用编程接口](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)。
*   [sklearn . model _ selection . cross _ val _ score API](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)。

### 文章

*   [数据准备，维基百科](https://en.wikipedia.org/wiki/Data_preparation)。
*   数据清理，维基百科。
*   [数据预处理，维基百科](https://en.wikipedia.org/wiki/Data_pre-processing)。

## 摘要

在本教程中，您发现了如何在评估机器学习模型时避免数据准备过程中的数据泄漏。

具体来说，您了解到:

*   将数据准备方法天真地应用于整个数据集会导致数据泄漏，从而导致对模型表现的不正确估计。
*   数据准备必须仅在训练集上准备，以避免数据泄露。
*   如何在 Python 中实现训练-测试拆分和 k-fold 交叉验证的无数据泄露的数据准备。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。