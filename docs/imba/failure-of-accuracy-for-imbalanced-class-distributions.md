# 不平衡类别分布的分类准确率故障

> 原文：<https://machinelearningmastery.com/failure-of-accuracy-for-imbalanced-class-distributions/>

最后更新于 2021 年 1 月 22 日

分类准确率是一种度量，它将分类模型的表现总结为正确预测的数量除以预测的总数。

它易于计算和直观理解，使其成为评估分类器模型最常用的指标。当实例在类中的分布严重偏斜时，这种直觉就会崩溃。

从业者在平衡数据集上开发的直觉，例如 99%代表熟练的模型，在不平衡分类预测建模问题上可能是不正确的和危险的误导。

在本教程中，您将发现不平衡分类问题的分类准确率故障。

完成本教程后，您将知道:

*   准确性和错误率是总结分类模型表现的事实上的标准度量。
*   由于从业者在具有相等类别分布的数据集上开发的直觉，分类准确率在具有偏斜类别分布的分类问题上失败。
*   用一个实例说明偏斜类分布的准确率故障的直觉。

**用我的新书[Python 不平衡分类](https://machinelearningmastery.com/imbalanced-classification-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 1 月更新**:针对 Sklearn v0.22 API 的变化进行了更新。

![Classification Accuracy Is Misleading for Skewed Class Distributions](img/ba011acaaadb85a2931645a43150dae6.png)

分类准确率对倾斜的类别分布有误导性
图片由[Esqui-Ando con tnho](https://flickr.com/photos/esqui-ando-con-tonho/41295716874/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  什么是分类准确率？
2.  不平衡分类的准确性失败
3.  不平衡分类的准确度示例

## 什么是分类准确率？

分类预测建模包括预测问题领域中给定示例的类别标签。

用于评估分类预测模型表现的最常见指标是分类准确率。通常，预测模型的准确性很好(高于 90%的准确性)，因此，根据模型的错误率来总结模型的表现也是非常常见的。

> 准确性及其补充错误率是在分类问题中评估学习系统表现最常用的指标。

——[不平衡分布下的预测建模综述](https://arxiv.org/abs/1505.01658)，2015 年。

[分类准确率](https://en.wikipedia.org/wiki/Accuracy_and_precision)首先使用分类模型对测试数据集中的每个示例进行预测。然后，将预测与测试集中那些示例的已知标签进行比较。准确度随后被计算为测试集中正确预测的示例的比例除以测试集中做出的所有预测。

*   准确性=正确预测/总预测

相反，错误率可以计算为在测试集上做出的不正确预测的总数除以在测试集上做出的所有预测。

*   错误率=错误预测/总预测

准确性和错误率是相辅相成的，这意味着我们总是可以从一个计算另一个。例如:

*   准确率= 1–误差率
*   误差率= 1–准确率

思考准确性的另一个有价值的方式是根据[混淆矩阵](https://machinelearningmastery.com/confusion-matrix-machine-learning/)。

混淆矩阵是分类模型所做预测的汇总，按类别组织成表格。表中的每一行都表示实际的类，每一列都表示预测的类。单元格中的值是对一个类所做的实际上是给定类的预测数的计数。对角线上的单元格表示正确的预测，其中预测的类和预期的类对齐。

> 评估分类器表现的最直接的方法是基于混淆矩阵分析。[……]从这样的矩阵中，可以提取出许多广泛使用的度量标准来衡量学习系统的表现，例如错误率[……]和准确性……

——[平衡机器学习训练数据的几种方法的行为研究](https://dl.acm.org/citation.cfm?id=1007735)，2004。

混淆矩阵不仅可以更深入地了解预测模型的准确性，还可以了解哪些类别被正确预测，哪些被错误预测，以及出现了什么类型的错误。

最简单的混淆矩阵是针对两类分类问题，有负(0 类)和正(1 类)类。

在这种类型的混淆矩阵中，表中的每个单元格都有一个具体且易于理解的名称，总结如下:

```py
               | Positive Prediction | Negative Prediction
Positive Class | True Positive (TP)  | False Negative (FN)
Negative Class | False Positive (FP) | True Negative (TN)
```

分类准确率可以从这个混淆矩阵中计算出来，即表格中正确单元格的总和(真阳性和真阴性)除以表格中的所有单元格。

*   准确率= (TP + TN) / (TP + FN + FP + TN)

同样，错误率也可以从混淆矩阵中计算出来，即表格中错误单元格的总和(假阳性和假阴性)除以表格中的所有单元格。

*   错误率= (FP + FN) / (TP + FN + FP + TN)

既然我们已经熟悉了分类准确率和它的补码错误率，让我们来发现为什么用它们来解决不平衡的分类问题可能是个坏主意。

## 不平衡分类的准确性失败

分类准确率是评估分类模型最常用的指标。

它被广泛使用的原因是因为它易于计算，易于解释，并且是一个单一的数字来概括模型的能力。

因此，在不平衡的分类问题上使用它是很自然的，在不平衡的分类问题上，训练数据集中的例子在类之间的分布是不相等的。

这是初学者对不平衡分类最常犯的错误。

当类别分布稍有偏差时，准确性仍然是一个有用的指标。当类分布中的偏差严重时，准确率可能成为模型表现的不可靠度量。

这种不可靠性的原因集中在一般的机器学习从业者和分类准确性的直觉上。

典型地，分类预测建模是用类分布相等或非常接近相等的小数据集来实践的。因此，大多数从业者会产生一种直觉，即大的准确度分数(或者相反，小的错误率分数)是好的，90%以上的值是好的。

达到 90%的分类准确率，甚至 99%的分类准确率，对于不平衡的分类问题来说可能是微不足道的。

这意味着，基于平衡类分布开发的分类准确率直觉将被应用，并且将是错误的，误导从业者认为模型具有良好甚至优秀的表现，而事实上，它没有。

### 准确性悖论

考虑 1:100 类不平衡的不平衡数据集的情况。

在这个问题中，少数类(类 1)的每个例子将有对应的多数类(类 0)的 100 个例子。

在这类问题中，多数类代表“*正常*”，少数类代表“*异常*”，如故障、诊断或欺诈。少数族裔的优秀表现将优先于两个阶层的优秀表现。

> 考虑到用户对少数(正)类示例的偏好，准确性是不合适的，因为与多数类相比，表示最少但更重要的示例的影响会降低。

——[不平衡分布下的预测建模综述](https://arxiv.org/abs/1505.01658)，2015 年。

在这个问题上，预测测试集中所有示例的多数类(类 0)的模型将具有 99%的分类准确率，反映了测试集中预期的主要和次要示例的平均分布。

许多机器学习模型是围绕平衡类分布的假设设计的，并且经常学习简单的规则(显式的或其他的)，比如总是预测多数类，导致它们达到 99%的准确率，尽管在实践中表现不比不熟练的多数类分类器好。

初学者会看到一个复杂模型在这种类型的不平衡数据集上达到 99%的表现，并相信他们的工作已经完成，而事实上，他们已经被误导了。

这种情况如此普遍，以至于它有了一个名字，被称为“[准确度悖论](https://en.wikipedia.org/wiki/Accuracy_paradox)”

> ……在不平衡数据集的框架内，准确性不再是一个恰当的衡量标准，因为它不能区分不同类别的正确分类的例子的数量。因此，它可能导致错误的结论…

——[阶级不平衡问题的集成综述:基于装袋、提升和混合的方法](https://ieeexplore.ieee.org/document/5978225)，2011。

严格来说，准确性确实报告了一个正确的结果；只有练习者对高准确度分数的直觉才是失败的点。通常使用替代度量来总结不平衡分类问题的模型表现，而不是纠正错误的直觉。

既然我们已经熟悉了分类可能会产生误导的观点，让我们来看一个成功的例子。

## 不平衡分类的准确度示例

虽然已经给出了为什么准确性对于不平衡分类是一个坏主意的解释，但它仍然是一个抽象的想法。

我们可以用一个工作示例来具体说明准确性的失败，并尝试反驳您可能已经开发的平衡类分布的任何准确性直觉，或者更有可能劝阻对不平衡数据集使用准确性。

首先，我们可以定义一个 1:100 类分布的合成数据集。

[make_blobs() Sklearn](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) 函数将始终创建类分布相等的合成数据集。

然而，我们可以使用这个函数来创建具有任意类分布的合成类别数据集，只需要几行额外的代码。类别分布可以定义为字典，其中关键字是类别值(例如 0 或 1)，值是要包含在数据集中的随机生成的示例数。

下面这个名为 *get_dataset()* 的函数将采用一个类分布，并返回一个具有该类分布的合成数据集。

```py
# create a dataset with a given class distribution
def get_dataset(proportions):
	# determine the number of classes
	n_classes = len(proportions)
	# determine the number of examples to generate for each class
	largest = max([v for k,v in proportions.items()])
	n_samples = largest * n_classes
	# create dataset
	X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1, cluster_std=3)
	# collect the examples
	X_list, y_list = list(), list()
	for k,v in proportions.items():
		row_ix = where(y == k)[0]
		selected = row_ix[:v]
		X_list.append(X[selected, :])
		y_list.append(y[selected])
	return vstack(X_list), hstack(y_list)
```

该函数可以接受任意数量的类，尽管我们将把它用于简单的二进制分类问题。

接下来，我们可以使用上一节中的代码为已创建的数据集创建散点图，并将其放入助手函数中。下面是 *plot_dataset()* 函数，该函数将绘制数据集并显示一个图例，以指示颜色到类标签的映射。

```py
# scatter plot of dataset, different color for each class
def plot_dataset(X, y):
	# create scatter plot for samples from each class
	n_classes = len(unique(y))
	for class_value in range(n_classes):
		# get row indexes for samples with this class
		row_ix = where(y == class_value)[0]
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(class_value))
	# show a legend
	pyplot.legend()
	# show the plot
	pyplot.show()
```

最后，我们可以测试这些新功能。

我们将定义一个 1:100 比率的数据集，少数类有 1000 个示例，多数类有 10000 个示例，并绘制结果。

下面列出了完整的示例。

```py
# define an imbalanced dataset with a 1:100 class ratio
from numpy import unique
from numpy import hstack
from numpy import vstack
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_blobs

# create a dataset with a given class distribution
def get_dataset(proportions):
	# determine the number of classes
	n_classes = len(proportions)
	# determine the number of examples to generate for each class
	largest = max([v for k,v in proportions.items()])
	n_samples = largest * n_classes
	# create dataset
	X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1, cluster_std=3)
	# collect the examples
	X_list, y_list = list(), list()
	for k,v in proportions.items():
		row_ix = where(y == k)[0]
		selected = row_ix[:v]
		X_list.append(X[selected, :])
		y_list.append(y[selected])
	return vstack(X_list), hstack(y_list)

# scatter plot of dataset, different color for each class
def plot_dataset(X, y):
	# create scatter plot for samples from each class
	n_classes = len(unique(y))
	for class_value in range(n_classes):
		# get row indexes for samples with this class
		row_ix = where(y == class_value)[0]
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(class_value))
	# show a legend
	pyplot.legend()
	# show the plot
	pyplot.show()

# define the class distribution 1:100
proportions = {0:10000, 1:1000}
# generate dataset
X, y = get_dataset(proportions)
# summarize class distribution:
major = (len(where(y == 0)[0]) / len(X)) * 100
minor = (len(where(y == 1)[0]) / len(X)) * 100
print('Class 0: %.3f%%, Class 1: %.3f%%' % (major, minor))
# plot dataset
plot_dataset(X, y)
```

运行该示例首先创建数据集并打印类分布。

我们可以看到，数据集中超过 99%的示例属于多数类，不到 1%的示例属于少数类。

```py
Class 0: 99.010%, Class 1: 0.990%
```

创建了一个数据集图，我们可以看到每个类都有更多的示例，还有一个有用的图例来指示图颜色到类标签的映射。

![Scatter Plot of Binary Classification Dataset With 1 to 100 Class Distribution](img/afe50700e057358309604c08a088f278.png)

1 到 100 类分布的二元类别数据集的散点图

接下来，我们可以拟合一个总是预测多数类的朴素分类器模型。

我们可以使用 scikit 中的 [DummyClassifier](https://Sklearn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) 来实现这一点——学习并使用“*最频繁*策略，该策略将始终预测在训练数据集中观察到最多的类标签。

```py
...
# define model
model = DummyClassifier(strategy='most_frequent')
```

然后，我们可以使用重复的 k 倍交叉验证在训练数据集上评估该模型。重要的是，我们使用分层交叉验证来确保数据集的每个分割都具有与训练数据集相同的类分布。这可以通过使用[repeated stratifiedfold 类](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)来实现。

下面的 *evaluate_model()* 函数实现了这一点，并返回模型每次评估的分数列表。

```py
# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y, metric):
	# define model
	model = DummyClassifier(strategy='most_frequent')
	# evaluate a model with repeated stratified k fold cv
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores
```

然后，我们可以评估模型，并计算每个评估得分的平均值。

我们预计朴素分类器将达到大约 99%的分类准确率，这是我们知道的，因为这是训练数据集中大多数类的分布。

```py
...
# evaluate model
scores = evaluate_model(X, y, 'accuracy')
# report score
print('Accuracy: %.3f%%' % (mean(scores) * 100))
```

将所有这些联系在一起，下面列出了在具有 1:100 类分布的合成数据集上评估朴素分类器的完整示例。

```py
# evaluate a majority class classifier on an 1:100 imbalanced dataset
from numpy import mean
from numpy import hstack
from numpy import vstack
from numpy import where
from sklearn.datasets import make_blobs
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# create a dataset with a given class distribution
def get_dataset(proportions):
	# determine the number of classes
	n_classes = len(proportions)
	# determine the number of examples to generate for each class
	largest = max([v for k,v in proportions.items()])
	n_samples = largest * n_classes
	# create dataset
	X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1, cluster_std=3)
	# collect the examples
	X_list, y_list = list(), list()
	for k,v in proportions.items():
		row_ix = where(y == k)[0]
		selected = row_ix[:v]
		X_list.append(X[selected, :])
		y_list.append(y[selected])
	return vstack(X_list), hstack(y_list)

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y, metric):
	# define model
	model = DummyClassifier(strategy='most_frequent')
	# evaluate a model with repeated stratified k fold cv
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores

# define the class distribution 1:100
proportions = {0:10000, 1:1000}
# generate dataset
X, y = get_dataset(proportions)
# summarize class distribution:
major = (len(where(y == 0)[0]) / len(X)) * 100
minor = (len(where(y == 1)[0]) / len(X)) * 100
print('Class 0: %.3f%%, Class 1: %.3f%%' % (major, minor))
# evaluate model
scores = evaluate_model(X, y, 'accuracy')
# report score
print('Accuracy: %.3f%%' % (mean(scores) * 100))
```

运行该示例首先再次报告训练数据集的类分布。

然后对模型进行评估，并报告平均准确率。我们可以看到，正如预期的那样，朴素分类器的表现与类分布完全匹配。

通常，达到 99%的分类准确率是值得庆祝的。虽然，正如我们所看到的，因为类分布不平衡，99%实际上是这个数据集可接受的最低准确率，也是更复杂的模型必须改进的起点。

```py
Class 0: 99.010%, Class 1: 0.990%
Accuracy: 99.010%
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [什么是机器学习中的混淆矩阵](https://machinelearningmastery.com/confusion-matrix-machine-learning/)

### 报纸

*   [不平衡分布下的预测建模综述](https://arxiv.org/abs/1505.01658)，2015。
*   [班级不平衡问题的集合研究综述:基于装袋、提升和混合的方法](https://ieeexplore.ieee.org/document/5978225)，2011。

### 书

*   [不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。
*   [从不平衡数据集中学习](https://amzn.to/307Xlva)，2018。

### 蜜蜂

*   [sklearn . dataset . make _ blobs API](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)。
*   [硬化. dummy . dummy class ification API](https://Sklearn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)。
*   [sklearn.model_selection。重复的策略应用编程接口](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)。

### 文章

*   [准确度和精密度，维基百科](https://en.wikipedia.org/wiki/Accuracy_and_precision)。
*   [准确性悖论，维基百科](https://en.wikipedia.org/wiki/Accuracy_paradox)。

## 摘要

在本教程中，您发现了不平衡分类问题的分类准确率故障。

具体来说，您了解到:

*   准确性和错误率是总结分类模型表现的事实上的标准度量。
*   由于从业者在具有相等类别分布的数据集上开发的直觉，分类准确率在具有偏斜类别分布的分类问题上失败。
*   用一个实例说明偏斜类分布的准确率故障的直觉。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。