# 如何为 Sklearn 创建自定义数据转换

> 原文：<https://machinelearningmastery.com/create-custom-data-transforms-for-Sklearn/>

最后更新于 2020 年 7 月 19 日

用于机器学习的 Sklearn Python 库提供了一套数据转换，用于改变输入数据的规模和分布，以及删除输入特征(列)。

有许多简单的数据清理操作，如移除异常值和移除观察值很少的列，这些操作通常是手动对数据执行的，需要自定义代码。

Sklearn 库提供了一种以标准方式包装这些**自定义数据转换**的方法，因此它们可以像任何其他转换一样使用，无论是直接用于数据还是作为建模管道的一部分。

在本教程中，您将发现如何为 Sklearn 定义和使用自定义数据转换。

完成本教程后，您将知道:

*   可以使用 FunctionTransformer 类为 Sklearn 创建自定义数据转换。
*   如何开发和应用自定义转换来移除几乎没有唯一值的列。
*   如何开发和应用自定义转换来替换每列的异常值？

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Create Custom Data Transforms for Sklearn](img/3a57fb9146c84aa3368b13756395c5c7.png)

如何为 Scikit 创建自定义数据转换-了解[贝瑞特·沃特金](https://www.flickr.com/photos/ben124/8126877692/)拍摄的
照片，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  Sklearn 中的自定义数据转换
2.  漏油数据集
3.  移除列的自定义转换
4.  替换异常值的自定义转换

## Sklearn 中的自定义数据转换

数据准备是指以某种方式改变原始数据，使其更适合用机器学习算法进行预测建模。

Sklearn Python 机器学习库直接提供了许多不同的数据准备技术，例如缩放数值输入变量和改变变量概率分布的技术。

这些变换可以拟合，然后应用于数据集或用作预测建模管道的一部分，从而允许在使用数据采样技术(如 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/))评估模型表现时，正确应用一系列变换，而不会出现数据泄漏。

尽管 Sklearn 中可用的数据准备技术非常广泛，但可能还需要额外的数据准备步骤。

通常，这些额外的步骤是在建模之前手动执行的，并且需要编写自定义代码。风险在于这些数据准备步骤的执行可能不一致。

解决方案是在 Sklearn 中使用 [FunctionTransformer 类](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)创建自定义数据转换。

此类允许您指定一个被调用来转换数据的函数。您可以定义函数并执行任何有效的更改，例如更改值或删除数据列(而不是删除行)。

该类可以像 Sklearn 中的任何其他数据转换一样使用，例如直接转换数据，或者在建模管道中使用。

问题是**转换是无状态的**，也就是说没有状态可以保持。

这意味着转换不能用于计算训练数据集中的统计数据，这些统计数据随后用于转换训练和测试数据集。

除了自定义缩放操作之外，这对于标准的数据清理操作也很有帮助，例如识别和移除几乎没有唯一值的列，以及识别和移除相对异常值。

我们将探索这两种情况，但是首先，让我们定义一个数据集，我们可以将其用作探索的基础。

## 漏油数据集

所谓的“[漏油](https://machinelearningmastery.com/imbalanced-classification-model-to-detect-oil-spills/)”数据集是标准的机器学习数据集。

这项任务包括预测一块区域是否有石油泄漏，例如非法或意外倾倒在海洋中的石油，给定一个描述卫星图像一块区域内容的向量。

有 937 例。每个案例由 48 个数值计算机视觉衍生特征、一个补丁号和一个类别标签组成。

正常情况下，没有漏油被指定为 0 级标签，而漏油被指定为 1 级标签。无漏油 896 例，漏油 41 例。

您可以在这里访问整个数据集:

*   [漏油数据集(漏油 csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/oil-spill.csv)
*   [漏油数据集描述(漏油名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/oil-spill.names)

查看文件的内容。

文件的前几行应该如下所示:

```py
1,2558,1506.09,456.63,90,6395000,40.88,7.89,29780,0.19,214.7,0.21,0.26,0.49,0.1,0.4,99.59,32.19,1.84,0.16,0.2,87.65,0,0.47,132.78,-0.01,3.78,0.22,3.2,-3.71,-0.18,2.19,0,2.19,310,16110,0,138.68,89,69,2850,1000,763.16,135.46,3.73,0,33243.19,65.74,7.95,1
2,22325,79.11,841.03,180,55812500,51.11,1.21,61900,0.02,901.7,0.02,0.03,0.11,0.01,0.11,6058.23,4061.15,2.3,0.02,0.02,87.65,0,0.58,132.78,-0.01,3.78,0.84,7.09,-2.21,0,0,0,0,704,40140,0,68.65,89,69,5750,11500,9593.48,1648.8,0.6,0,51572.04,65.73,6.26,0
3,115,1449.85,608.43,88,287500,40.42,7.34,3340,0.18,86.1,0.21,0.32,0.5,0.17,0.34,71.2,16.73,1.82,0.19,0.29,87.65,0,0.46,132.78,-0.01,3.78,0.7,4.79,-3.36,-0.23,1.95,0,1.95,29,1530,0.01,38.8,89,69,1400,250,150,45.13,9.33,1,31692.84,65.81,7.84,1
4,1201,1562.53,295.65,66,3002500,42.4,7.97,18030,0.19,166.5,0.21,0.26,0.48,0.1,0.38,120.22,33.47,1.91,0.16,0.21,87.65,0,0.48,132.78,-0.01,3.78,0.84,6.78,-3.54,-0.33,2.2,0,2.2,183,10080,0,108.27,89,69,6041.52,761.58,453.21,144.97,13.33,1,37696.21,65.67,8.07,1
5,312,950.27,440.86,37,780000,41.43,7.03,3350,0.17,232.8,0.15,0.19,0.35,0.09,0.26,289.19,48.68,1.86,0.13,0.16,87.65,0,0.47,132.78,-0.01,3.78,0.02,2.28,-3.44,-0.44,2.19,0,2.19,45,2340,0,14.39,89,69,1320.04,710.63,512.54,109.16,2.58,0,29038.17,65.66,7.35,0
...
```

我们可以看到第一列包含补丁号的整数。我们还可以看到，计算机视觉导出的特征是实值的，具有不同的比例，例如第二列中的千分之一和其他列中的分数。

该数据集包含具有极少唯一值的列和具有异常值的列，这为数据清理提供了良好的基础。

下面的示例下载数据集并将其加载为 numPy 数组，并总结了行数和列数。

```py
# load the oil dataset
from pandas import read_csv
# define the location of the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/oil-spill.csv'
# load the dataset
df = read_csv(path, header=None)
# split data into inputs and outputs
data = df.values
X = data[:, :-1]
y = data[:, -1]
print(X.shape, y.shape)
```

运行该示例将加载数据集并确认预期的行数和列数。

```py
(937, 49) (937,)
```

现在我们有了一个数据集，可以用作数据转换的基础，让我们看看如何使用 *FunctionTransformer* 类定义一些自定义的数据清理转换。

## 移除列的自定义转换

几乎没有唯一值的列可能对预测目标值没有任何帮助。

这并不是绝对正确的，但是您应该在删除了这种类型的列的数据集上测试模型拟合的表现，这是足够正确的。

这是一种类型的[数据清理](https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/)，Sklearn 中提供了一种称为[变量阈值](https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/)的数据转换，试图利用每一列的方差来解决这个问题。

另一种方法是移除具有少于指定数量的唯一值(如 1)的列。

我们可以开发一个应用这种转换的函数，并使用最小数量的唯一值作为可配置的默认参数。我们还将添加一些调试，以确认它是否如我们预期的那样工作。

首先，可以计算每列的唯一值的数量。可以识别等于或小于唯一值最小数量的十列。最后，可以从数据集中删除那些已识别的列。

下面的 *cust_transform()* 函数实现了这一点。

```py
# remove columns with few unique values
def cust_transform(X, min_values=1, verbose=True):
	# get number of unique values for each column
	counts = [len(unique(X[:, i])) for i in range(X.shape[1])]
	if verbose:
		print('Unique Values: %s' % counts)
	# select columns to delete
	to_del = [i for i,v in enumerate(counts) if v <= min_values]
	if verbose:
		print('Deleting: %s' % to_del)
	if len(to_del) is 0:
		return X
	# select all but the columns that are being removed
	ix = [i for i in range(X.shape[1]) if i not in to_del]
	result = X[:, ix]
	return result
```

然后我们可以在 FunctionTransformer 中使用这个函数。

这种转换的一个限制是，它根据提供的数据选择要删除的列。这意味着如果一个训练和测试数据集相差很大，那么就有可能从每一个中移除不同的列，使得模型评估具有挑战性(*不稳定*！？).因此，最好将唯一值的最小数量保持较小，例如 1。

我们可以在漏油数据集上使用这种转换。下面列出了完整的示例。

```py
# custom data transform for removing columns with few unique values
from numpy import unique
from pandas import read_csv
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder

# load a dataset
def load_dataset(path):
	# load the dataset
	df = read_csv(path, header=None)
	data = df.values
	# split data into inputs and outputs
	X, y = data[:, :-1], data[:, -1]
	# minimally prepare dataset
	X = X.astype('float')
	y = LabelEncoder().fit_transform(y.astype('str'))
	return X, y

# remove columns with few unique values
def cust_transform(X, min_values=1, verbose=True):
	# get number of unique values for each column
	counts = [len(unique(X[:, i])) for i in range(X.shape[1])]
	if verbose:
		print('Unique Values: %s' % counts)
	# select columns to delete
	to_del = [i for i,v in enumerate(counts) if v <= min_values]
	if verbose:
		print('Deleting: %s' % to_del)
	if len(to_del) is 0:
		return X
	# select all but the columns that are being removed
	ix = [i for i in range(X.shape[1]) if i not in to_del]
	result = X[:, ix]
	return result

# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/oil-spill.csv'
# load the dataset
X, y = load_dataset(url)
print(X.shape, y.shape)
# define the transformer
trans = FunctionTransformer(cust_transform)
# apply the transform
X = trans.fit_transform(X)
# summarize new shape
print(X.shape)
```

运行该示例首先报告原始数据集中的行数和列数。

接下来，将打印一个列表，显示数据集中每列观察到的唯一值的数量。我们可以看到许多列只有很少的唯一值。

然后识别并报告具有一个(或更少)唯一值的列。在这种情况下，列索引为 22。此列将从数据集中删除。

最后，报告转换后数据集的形状，显示 48 列而不是 49 列，确认具有单个唯一值的列已被删除。

```py
(937, 49) (937,)
Unique Values: [238, 297, 927, 933, 179, 375, 820, 618, 561, 57, 577, 59, 73, 107, 53, 91, 893, 810, 170, 53, 68, 9, 1, 92, 9, 8, 9, 308, 447, 392, 107, 42, 4, 45, 141, 110, 3, 758, 9, 9, 388, 220, 644, 649, 499, 2, 937, 169, 286]
Deleting: [22]
(937, 48)
```

您可以探索这种转换的许多扩展，例如:

*   确保它仅适用于数值输入变量。
*   尝试不同的最小唯一值数。
*   使用百分比而不是唯一值的绝对值。

如果您探索这些扩展中的任何一个，请在下面的评论中告诉我。

接下来，让我们看一下替换数据集中的值的转换。

## 替换异常值的自定义转换

[异常值](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)是与其他观测值不同或不一样的观测值。

如果我们一次只考虑一个变量，那么离群值将是远离质心(其余值)的值，这意味着它很少或被观察到的概率很低。

对于常见的概率分布，有识别异常值的标准方法。对于高斯数据，我们可以将异常值识别为偏离平均值三个或更多标准差的观测值。

对于具有许多输入变量的数据，这可能是识别异常值的理想方法，也可能不是，但在某些情况下可能是有效的。

我们可以通过这种方式识别异常值，并用修正值(如平均值)替换它们的值。

每一列被认为是一次一个，并计算平均和标准偏差统计。使用这些统计数据，定义了“*正常*”值的上下限，然后可以识别所有超出这些界限的值。如果识别出一个或多个异常值，则用已经计算的平均值替换它们的值。

下面的 *cust_transform()* 函数将其实现为应用于数据集的函数，在这里我们参数化平均值的标准偏差数以及是否显示调试信息。

```py
# replace outliers
def cust_transform(X, n_stdev=3, verbose=True):
	# copy the array
	result = X.copy()
	# enumerate each column
	for i in range(result.shape[1]):
		# retrieve values for column
		col = X[:, i]
		# calculate statistics
		mu, sigma = mean(col), std(col)
		# define bounds
		lower, upper = mu-(sigma*n_stdev), mu+(sigma*n_stdev)
		# select indexes that are out of bounds
		ix = where(logical_or(col < lower, col > upper))[0]
		if verbose and len(ix) > 0:
			print('>col=%d, outliers=%d' % (i, len(ix)))
		# replace values
		result[ix, i] = mu
	return result
```

然后我们可以在 FunctionTransformer 中使用这个函数。

异常值检测的方法假设一个[高斯概率分布](https://machinelearningmastery.com/continuous-probability-distributions-for-machine-learning/)并独立适用于每个变量，两者都是强假设。

此实现的另一个限制是，平均值和标准偏差统计是在提供的数据集上计算的，这意味着异常值的定义及其替换值都是相对于数据集而言的。这意味着，如果在训练集和测试集上使用变换，可以使用不同的异常值定义和不同的替换值。

我们可以在漏油数据集上使用这种转换。下面列出了完整的示例。

```py
# custom data transform for replacing outliers
from numpy import mean
from numpy import std
from numpy import where
from numpy import logical_or
from pandas import read_csv
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder

# load a dataset
def load_dataset(path):
	# load the dataset
	df = read_csv(path, header=None)
	data = df.values
	# split data into inputs and outputs
	X, y = data[:, :-1], data[:, -1]
	# minimally prepare dataset
	X = X.astype('float')
	y = LabelEncoder().fit_transform(y.astype('str'))
	return X, y

# replace outliers
def cust_transform(X, n_stdev=3, verbose=True):
	# copy the array
	result = X.copy()
	# enumerate each column
	for i in range(result.shape[1]):
		# retrieve values for column
		col = X[:, i]
		# calculate statistics
		mu, sigma = mean(col), std(col)
		# define bounds
		lower, upper = mu-(sigma*n_stdev), mu+(sigma*n_stdev)
		# select indexes that are out of bounds
		ix = where(logical_or(col < lower, col > upper))[0]
		if verbose and len(ix) > 0:
			print('>col=%d, outliers=%d' % (i, len(ix)))
		# replace values
		result[ix, i] = mu
	return result

# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/oil-spill.csv'
# load the dataset
X, y = load_dataset(url)
print(X.shape, y.shape)
# define the transformer
trans = FunctionTransformer(cust_transform)
# apply the transform
X = trans.fit_transform(X)
# summarize new shape
print(X.shape)
```

运行该示例首先报告任何更改之前数据集的形状。

接下来，计算每列的异常值数量，并且在输出中只报告那些具有一个或多个异常值的列。我们可以看到数据集中总共有 32 列有一个或多个异常值。

然后移除异常值，并报告结果数据集的形状，确认行数或列数没有变化。

```py
(937, 49) (937,)
>col=0, outliers=10
>col=1, outliers=8
>col=3, outliers=8
>col=5, outliers=7
>col=6, outliers=1
>col=7, outliers=12
>col=8, outliers=15
>col=9, outliers=14
>col=10, outliers=19
>col=11, outliers=17
>col=12, outliers=22
>col=13, outliers=2
>col=14, outliers=16
>col=15, outliers=8
>col=16, outliers=8
>col=17, outliers=6
>col=19, outliers=12
>col=20, outliers=20
>col=27, outliers=14
>col=28, outliers=18
>col=29, outliers=2
>col=30, outliers=13
>col=32, outliers=3
>col=34, outliers=14
>col=35, outliers=15
>col=37, outliers=13
>col=40, outliers=18
>col=41, outliers=13
>col=42, outliers=12
>col=43, outliers=12
>col=44, outliers=19
>col=46, outliers=21
(937, 49)
```

您可以探索这种转换的许多扩展，例如:

*   确保它仅适用于数值输入变量。
*   用与平均值不同数量的标准偏差进行实验，例如 2 或 4。
*   使用不同的异常值定义，如 IQR 或模型。

如果您探索这些扩展中的任何一个，请在下面的评论中告诉我。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 相关教程

*   [如何去除机器学习的异常值](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)
*   [如何用 Python 进行机器学习的数据清洗](https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/)

### 蜜蜂

*   [定制变压器，sci kit-学习用户指南](https://Sklearn.org/stable/modules/preprocessing.html#custom-transformers)。
*   [sklearn . preferences . function transformer API](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)。

### 摘要

在本教程中，您发现了如何为 Sklearn 定义和使用自定义数据转换。

具体来说，您了解到:

*   可以使用 FunctionTransformer 类为 Sklearn 创建自定义数据转换。
*   如何开发和应用自定义转换来移除几乎没有唯一值的列。
*   如何开发和应用自定义转换来替换每列的异常值？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。