# 机器学习的数据准备（7 天迷你课程）

> 原文：<https://machinelearningmastery.com/data-preparation-for-machine-learning-7-day-mini-course/>

最后更新于 2020 年 6 月 30 日

#### 机器学习速成班的数据准备。
用 Python 7 天搞定数据准备。

数据准备包括将原始数据转换成更适合建模的形式。

准备数据可能是预测建模项目中最重要的部分，也是最耗时的部分，尽管它似乎是讨论最少的部分。相反，重点是机器学习算法，其使用和参数化已经变得非常常规。

实际的数据准备需要数据清理、特征选择、数据转换、降维等知识。

在本速成课程中，您将发现如何在七天内开始并自信地用 Python 为预测建模项目准备数据。

这是一个又大又重要的岗位。你可能想把它做成书签。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 6 月更新**:更改了马绞痛数据集的目标。

![Data Preparation for Machine Learning (7-Day Mini-Course)](img/86c115f622d5613d1e4b197edbfef3ba.png)

机器学习数据准备(7 天迷你课程)
图片由[克里斯蒂安·科林斯](https://flickr.com/photos/collins_family/31071050105/)提供，保留部分权利。

## 这个速成班是给谁的？

在我们开始之前，让我们确保你在正确的地方。

本课程面向可能了解一些应用机器学习的开发人员。也许你知道如何用流行的工具来从头到尾解决一个预测建模问题，或者至少是大部分的主要步骤。

本课程中的课程假设了您的一些情况，例如:

*   你对编程的基本 Python 很熟悉。
*   您可能知道一些用于数组操作的基本 NumPy。
*   你可能知道一些基本的 sci kit-学习建模。

你不需要:

*   数学天才！
*   机器学习专家！

这门速成课程将把你从一个懂得一点机器学习的开发人员带到一个能够有效和胜任地为预测建模项目准备数据的开发人员。

注意:本速成课程假设您有一个至少安装了 NumPy 的工作 Python 3 SciPy 环境。如果您需要环境方面的帮助，可以遵循这里的逐步教程:

*   [如何用 Anaconda](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/) 设置机器学习的 Python 环境

## 速成班概述

这门速成课分为七节课。

您可以每天完成一节课(推荐)或一天内完成所有课程(硬核)。这真的取决于你有多少时间和你的热情程度。

下面是用 Python 进行数据准备的七个经验教训:

*   **第 01 课**:数据准备的重要性
*   **第 02 课**:用插补填充缺失值
*   **第 03 课**:使用 RFE 选择要素
*   **第 04 课**:通过标准化来缩放数据
*   **第 05 课**:用一次编码转换类别
*   **第 06 课**:用 kBins 将数字转换为类别
*   **第 07 课**:主成分分析降维

每节课可能需要你 60 秒或 30 分钟。慢慢来，按照自己的节奏完成课程。提问，甚至在下面的评论中发布结果。

这些课程可能期望你去发现如何做事。我会给你一些提示，但是每节课的部分要点是迫使你学习去哪里寻找关于 Python 中的算法和最佳工具的帮助。(**提示**:我有这个博客上所有的答案；使用搜索框。)

在评论中发布您的结果；我会为你加油的！

坚持住。不要放弃。

## 第 01 课:数据准备的重要性

在本课中，您将发现数据准备在机器学习预测建模中的重要性。

预测建模项目涉及从数据中学习。

数据指的是领域中的例子或案例，它们描述了您想要解决的问题。

在预测建模项目中，例如分类或回归，原始数据通常不能直接使用。

出现这种情况有四个主要原因:

*   **数据类型**:机器学习算法要求数据为数字。
*   **数据要求**:有些机器学习算法对数据有要求。
*   **数据错误**:数据中的统计噪声和错误可能需要修正。
*   **数据复杂度**:数据中可能会梳理出复杂的非线性关系。

原始数据必须在用于拟合和评估机器学习模型之前进行预处理。预测建模项目中的这一步被称为“数据准备”

在机器学习项目的数据准备步骤中，您可以使用或探索一些常见或标准的任务。

这些任务包括:

*   **数据清理**:识别并纠正数据中的错误或差错。
*   **特征选择**:识别那些与任务最相关的输入变量。
*   **数据转换**:改变变量的规模或分布。
*   **特征工程**:从可用数据中导出新变量。
*   **降维**:创建数据的紧凑投影。

这些任务中的每一项都是具有专门算法的整个研究领域。

### 你的任务

在本课中，您必须列出三种您知道的或以前可能使用过的数据准备算法，并给出一行摘要。

数据准备算法的一个例子是数据标准化，它将数字变量缩放到 0 到 1 之间的范围。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何修复缺失值的数据，称为数据插补。

## 第 02 课:用插补填补缺失值

在本课中，您将了解如何识别和填充数据中缺失的值。

真实世界的数据往往有[缺失值](https://machinelearningmastery.com/handle-missing-data-python/)。

由于多种原因，数据可能会丢失值，例如未记录的观察值和数据损坏。处理缺失数据很重要，因为许多机器学习算法不支持缺失值的数据。

用数据填充缺失值称为数据插补，一种流行的数据插补方法是计算每一列的统计值(如平均值)，并用统计值替换该列的所有缺失值。

[马绞痛数据集](https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv)描述了患有绞痛的马的医学特征以及它们是活的还是死的。它缺少标有问号“？”的值。我们可以用 [read_csv()函数](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)加载数据集，并确保问号值标记为 NaN。

一旦加载，我们就可以使用[simple importer](https://Sklearn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)类将所有用 NaN 值标记的缺失值转换为列的平均值。

下面列出了完整的示例。

```py
# statistical imputation transform for the horse colic dataset
from numpy import isnan
from pandas import read_csv
from sklearn.impute import SimpleImputer
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')
# split into input and output elements
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# print total missing
print('Missing: %d' % sum(isnan(X).flatten()))
# define imputer
imputer = SimpleImputer(strategy='mean')
# fit on the dataset
imputer.fit(X)
# transform the dataset
Xtrans = imputer.transform(X)
# print total missing
print('Missing: %d' % sum(isnan(Xtrans).flatten()))
```

### 你的任务

在本课中，您必须运行示例并查看数据插补转换前后数据集中缺失值的数量。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将了解如何选择数据集中最重要的要素。

## 第 03 课:使用 RFE 选择要素

在本课中，您将了解如何选择数据集中最重要的要素。

特征选择是在开发预测模型时减少输入变量数量的过程。

希望减少输入变量的数量，以降低建模的计算成本，并在某些情况下提高模型的表现。

递归特征消除，简称 RFE，是一种流行的特征选择算法。

RFE 之所以受欢迎，是因为它易于配置和使用，并且可以有效地选择训练数据集中与预测目标变量更相关或最相关的特征(列)。

Sklearn Python 机器学习库为机器学习提供了一个 [RFE](https://Sklearn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) 的实现。RFE 是一个转变。要使用它，首先，该类配置有通过“估计器”参数指定的所选算法，以及通过“ *n_features_to_select* ”参数选择的特征数量。

下面的示例定义了一个具有五个冗余输入要素的合成类别数据集。然后，使用决策树算法使用 RFE 来选择五个特征。

```py
# report which features were selected by RFE
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define RFE
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
# fit RFE
rfe.fit(X, y)
# summarize all features
for i in range(X.shape[1]):
	print('Column: %d, Selected=%s, Rank: %d' % (i, rfe.support_[i], rfe.ranking_[i]))
```

### 你的任务

在本课中，您必须运行示例并查看选择了哪些要素以及每个输入要素的相对排名。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何缩放数字数据。

## 第 04 课:通过标准化来缩放数据

在本课中，您将发现如何为机器学习缩放数字数据。

当数值输入变量被缩放到标准范围时，许多机器学习算法表现更好。

这包括使用输入加权和的算法，如线性回归，以及使用距离度量的算法，如 k 近邻。

建模前缩放数值数据的最流行技术之一是标准化。规范化将每个输入变量分别缩放到 0-1 的范围，这是我们准确率最高的浮点值范围。它要求你知道或能够准确估计每个变量的最小和最大可观察值。您可能能够从您的可用数据中估计这些值。

您可以使用 Sklearn 对象[最小最大缩放器](http://Sklearn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)来规范化数据集。

下面的例子定义了一个合成类别数据集，然后使用最小最大缩放器来规范化输入变量。

```py
# example of normalizing input data
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0, random_state=1)
# summarize data before the transform
print(X[:3, :])
# define the scaler
trans = MinMaxScaler()
# transform the data
X_norm = trans.fit_transform(X)
# summarize data after the transform
print(X_norm[:3, :])
```

### 你的任务

对于本课，您必须运行该示例，并在规范化转换之前和之后报告输入变量的比例。

对于奖励点，计算转换前后每个变量的最小值和最大值，以确认它是否按预期应用。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何将分类变量转换为数字。

## 第 05 课:使用单一热编码转换类别

在本课中，您将发现如何将分类输入变量编码为数字。

机器学习模型要求所有的输入和输出变量都是数字。这意味着，如果您的数据包含类别数据，您必须将其编码为数字，然后才能拟合和评估模型。

将分类变量转换为数字的最流行的技术之一是 one-hot 编码。

[类别数据](https://en.wikipedia.org/wiki/Categorical_variable)是包含标签值而不是数值的变量。

分类变量的每个标签可以映射到一个唯一的整数，称为序数编码。然后，可以对序数表示应用一次性编码。这是指为变量中的每个唯一整数值向数据集中添加一个新的二进制变量，并从数据集中移除原始分类变量。

例如，假设我们有一个带有三个类别的“*颜色*”变量(*红色*、*绿色*、*蓝色*)。在这种情况下，需要三个二进制变量。颜色的二进制变量中有一个“1”值，其他颜色的二进制变量中有“0”值。

例如:

```py
red,	green,	blue
1,		0,		0
0,		1,		0
0,		0,		1
```

通过 [OneHotEncoder 类](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)，在 Sklearn Python 机器学习库中可以获得这种一次性编码转换。

乳腺癌数据集仅包含分类输入变量。

以下示例加载数据集，并对每个分类输入变量进行热编码。

```py
# one-hot encode the breast cancer dataset
from pandas import read_csv
from sklearn.preprocessing import OneHotEncoder
# define the location of the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
# load the dataset
dataset = read_csv(url, header=None)
# retrieve the array of data
data = dataset.values
# separate into input and output columns
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)
# summarize the raw data
print(X[:3, :])
# define the one hot encoding transform
encoder = OneHotEncoder(sparse=False)
# fit and apply the transform to the input data
X_oe = encoder.fit_transform(X)
# summarize the transformed data
print(X_oe[:3, :])
```

### 你的任务

对于本课，您必须运行该示例，并报告转换前的原始数据，以及应用一次性编码后对数据的影响。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何将数字变量转换为类别。

## 第 06 课:用 kBins 将数字转换为类别

在本课中，您将发现如何将数字变量转换为分类变量。

一些机器学习算法可能更喜欢或需要分类或顺序输入变量，例如一些决策树和基于规则的算法。

这可能是由数据中的异常值、多模态分布、高度指数分布等引起的。

当具有非标准分布的数值输入变量被转换为具有新的分布或全新的数据类型时，许多机器学习算法更喜欢或表现得更好。

一种方法是使用数值变量的变换来具有离散的概率分布，其中每个数值被分配一个标签，并且标签具有有序(序数)关系。

这被称为离散化变换，通过使数值输入变量的概率分布离散化，可以提高一些机器学习模型对数据集的表现。

离散化转换可通过[KBinsDistrictzer 类](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)在 Sklearn Python 机器学习库中获得。

它允许您指定要创建的离散面元的数量(*n _ 面元*)、转换的结果是序数还是单热编码(encode)以及用于划分变量值的分布(策略)，例如“*统一*”

下面的示例创建了一个包含 10 个数字输入变量的合成输入变量，然后用序数编码将每个变量编码到 10 个离散的容器中。

```py
# discretize numeric input variables
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer
# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0, random_state=1)
# summarize data before the transform
print(X[:3, :])
# define the transform
trans = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
# transform the data
X_discrete = trans.fit_transform(X)
# summarize data after the transform
print(X_discrete[:3, :])
```

### 你的任务

对于本课，您必须在转换之前运行示例并报告原始数据，然后报告转换对数据的影响。

对于奖励点数，探索变换的替代配置，例如不同的策略和箱数。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何降低输入数据的维度。

## 第七课:用主成分分析进行降维

在本课中，您将发现如何使用[降维](https://machinelearningmastery.com/dimensionality-reduction-for-machine-learning/)来减少数据集中输入变量的数量。

数据集的输入变量或特征的数量称为其维数。

降维是指减少数据集中输入变量数量的技术。

更多的输入特征通常会使预测建模任务更具挑战性，更一般地说，这被称为维数灾难。

尽管在高维统计中，降维技术经常用于数据可视化，但是这些技术可以用于应用机器学习来简化分类或回归数据集，以便更好地拟合预测模型。

也许机器学习中最流行的降维技术是主成分分析，简称 PCA。这是一种来自线性代数领域的技术，可以用作数据准备技术，在拟合模型之前创建数据集的投影。

得到的数据集，即投影，可以用作训练机器学习模型的输入。

Sklearn 库提供了[主成分分析类](https://Sklearn.org/stable/modules/generated/sklearn.decomposition.PCA.html)，它可以适合数据集，并用于转换训练数据集和未来的任何附加数据集。

下面的示例创建了一个包含 10 个输入变量的合成二进制类别数据集，然后使用主成分分析将数据集的维度减少到三个最重要的组成部分。

```py
# example of pca for dimensionality reduction
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=3, n_redundant=7, random_state=1)
# summarize data before the transform
print(X[:3, :])
# define the transform
trans = PCA(n_components=3)
# transform the data
X_dim = trans.fit_transform(X)
# summarize data after the transform
print(X_dim[:3, :])
```

### 你的任务

在本课中，您必须运行示例，并在应用转换后报告原始数据集和数据集的结构和形式。

对于加分，探索具有不同数量的选定组件的变换。

在下面的评论中发表你的答案。我想看看你有什么想法。

这是迷你课程的最后一课。

## 末日！
(看你走了多远)

你成功了。干得好！

花一点时间，回头看看你已经走了多远。

你发现了:

*   预测建模机器学习项目中数据准备的重要性。
*   如何标记缺失数据并使用统计插补来估计缺失值？
*   如何利用递归特征消除去除冗余输入变量？
*   如何将不同比例的输入变量转换为标准范围，称为标准化。
*   如何将分类输入变量转换为数字称为一热编码？
*   如何将数值变量转换成离散类别称为离散化。
*   如何使用主成分分析将数据集投影到较少的维度。

## 摘要

**你觉得迷你课程怎么样？**
你喜欢这个速成班吗？

**你有什么问题吗？有什么症结吗？**
让我知道。请在下面留言。