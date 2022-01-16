# 如何为机器学习在表格数据上使用特征提取

> 原文：<https://machinelearningmastery.com/feature-extraction-on-tabular-data/>

最后更新于 2020 年 8 月 17 日

机器学习预测建模表现只和你的数据一样好，你的数据只和你准备建模的方式一样好。

最常见的数据准备方法是研究数据集并回顾机器学习算法的期望，然后仔细选择最合适的数据准备技术来转换原始数据，以最好地满足算法的期望。这是缓慢的，昂贵的，需要大量的专业知识。

数据准备的另一种方法是对原始数据并行应用一套常见且通常有用的数据准备技术，并将所有转换的结果组合成单个大数据集，从中可以拟合和评估模型。

这是数据准备的另一种理念，它将数据转换视为从原始数据中提取显著特征的方法，以向学习算法展示问题的结构。它需要学习可缩放权重输入特征的算法，并使用与预测目标最相关的输入特征。

这种方法需要较少的专业知识，与数据准备方法的全网格搜索相比，计算效率高，并且有助于发现非直观的数据准备解决方案，这些解决方案对于给定的预测建模问题实现了良好或最佳的表现。

在本教程中，您将了解如何使用特征提取来准备表格数据。

完成本教程后，您将知道:

*   特征提取为表格数据的数据准备提供了一种替代方法，其中所有数据转换并行应用于原始输入数据，并组合在一起创建一个大型数据集。
*   如何使用数据准备的特征提取方法来提高标准类别数据集的模型表现。
*   如何将特征选择添加到特征提取建模管道中，以进一步提升标准数据集的建模表现。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Use Feature Extraction on Tabular Data for Data Preparation](img/f9cb478a4a690d5779b02831a6003840.png)

如何使用表格数据的特征提取进行数据准备
图片由[尼古拉斯·瓦尔德斯](https://www.flickr.com/photos/vonfer/42261101585/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  用于数据准备的特征提取技术
2.  数据集和表现基线
    1.  葡萄酒类别数据集
    2.  基线模型表现
3.  数据准备的特征提取方法

## 用于数据准备的特征提取技术

数据准备可能具有挑战性。

最常规定和遵循的方法是分析数据集，审查算法的要求，并转换原始数据以最好地满足算法的期望。

这可能是有效的，但也很慢，可能需要数据分析和机器学习算法方面的深厚专业知识。

另一种方法是将输入变量的准备视为建模管道的超参数，并随着算法和算法配置的选择对其进行调整。

这也是一种有效的方法，可以暴露非直观的解决方案，并且只需要很少的专业知识，尽管它在计算上可能很昂贵。

在这两种数据准备方法之间寻求中间立场的一种方法是将输入数据的转换视为**特征工程**或**特征提取**过程。这包括对原始数据应用一套通用或常用的数据准备技术，然后将所有要素聚合在一起以创建一个大数据集，然后根据该数据拟合和评估模型。

该方法的原理是将每种数据准备技术视为一种转换，从原始数据中提取显著特征，并将其呈现给学习算法。理想情况下，这种变换可以解开复杂的关系和复合输入变量，进而允许使用更简单的建模算法，如线性机器学习技术。

由于没有更好的名称，我们将其称为“**特征工程方法**”或“**特征提取方法**”，用于为预测建模项目配置数据准备。

它允许在选择数据准备方法时使用数据分析和算法专业知识，并允许以低得多的计算成本找到非直观的解决方案。

输入特征数量的排除也可以通过使用特征选择技术来明确解决，该技术试图对大量提取特征的重要性或值进行排序，并且仅选择与预测目标变量最相关的一小部分。

我们可以通过一个工作示例来探索这种数据准备方法。

在深入研究一个工作示例之前，让我们首先选择一个标准数据集，并开发一个表现基线。

## 数据集和表现基线

在本节中，我们将首先选择一个标准的机器学习数据集，并在该数据集上建立表现基线。这将为下一节探讨数据准备的特征提取方法提供背景。

### 葡萄酒类别数据集

我们将使用葡萄酒类别数据集。

该数据集有 13 个输入变量，用于描述葡萄酒样品的化学成分，并要求将葡萄酒分为三种类型。

您可以在此了解有关数据集的更多信息:

*   [葡萄酒数据集(wine.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv)
*   [葡萄酒数据集描述(葡萄酒.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.names)

不需要下载数据集，因为我们将自动下载它作为我们工作示例的一部分。

打开数据集并查看原始数据。下面列出了前几行数据。

我们可以看到，这是一个带有数值输入变量的[多类分类](https://machinelearningmastery.com/types-of-classification-in-machine-learning/)预测建模问题，每个变量都有不同的尺度。

```py
14.23,1.71,2.43,15.6,127,2.8,3.06,.28,2.29,5.64,1.04,3.92,1065,1
13.2,1.78,2.14,11.2,100,2.65,2.76,.26,1.28,4.38,1.05,3.4,1050,1
13.16,2.36,2.67,18.6,101,2.8,3.24,.3,2.81,5.68,1.03,3.17,1185,1
14.37,1.95,2.5,16.8,113,3.85,3.49,.24,2.18,7.8,.86,3.45,1480,1
13.24,2.59,2.87,21,118,2.8,2.69,.39,1.82,4.32,1.04,2.93,735,1
...
```

该示例加载数据集并将其拆分为输入和输出列，然后汇总数据数组。

```py
# example of loading and summarizing the wine dataset
from pandas import read_csv
# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv'
# load the dataset as a data frame
df = read_csv(url, header=None)
# retrieve the numpy array
data = df.values
# split the columns into input and output variables
X, y = data[:, :-1], data[:, -1]
# summarize the shape of the loaded data
print(X.shape, y.shape)
```

运行该示例，我们可以看到数据集被正确加载，并且有 179 行数据，包含 13 个输入变量和一个目标变量。

```py
(178, 13) (178,)
```

接下来，让我们在这个数据集上评估一个模型，并建立一个表现基线。

### 基线模型表现

我们可以通过评估原始输入数据的模型来建立葡萄酒分类任务的表现基线。

在这种情况下，我们将评估逻辑回归模型。

首先，我们可以通过确保输入变量是数字的并且目标变量是标签编码的来执行最少的数据准备工作，正如 Sklearn 库所期望的那样。

```py
...
# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))
```

接下来，我们可以定义我们的预测模型。

```py
...
# define the model
model = LogisticRegression(solver='liblinear')
```

我们将使用[重复分层 k-fold 交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)的金标准对模型进行评估，重复 10 次，重复 3 次。

将使用分类准确度评估模型表现。

```py
...
model = LogisticRegression(solver='liblinear')
# define the cross-validation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

在运行结束时，我们将报告在所有重复和评估折叠中收集的准确度分数的平均值和标准偏差。

```py
...
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

将这些联系在一起，下面列出了在原酒类别数据集上评估逻辑回归模型的完整示例。

```py
# baseline model performance on the wine dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv'
df = read_csv(url, header=None)
data = df.values
X, y = data[:, :-1], data[:, -1]
# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))
# define the model
model = LogisticRegression(solver='liblinear')
# define the cross-validation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例评估模型表现，并报告平均和标准偏差分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，适合原始输入数据的逻辑回归模型实现了大约 95.3%的平均分类准确率，提供了表现基线。

```py
Accuracy: 0.953 (0.048)
```

接下来，让我们探索是否可以使用基于特征提取的数据准备方法来提高表现。

## 数据准备的特征提取方法

在本节中，我们可以探索是否可以使用特征提取方法来提高数据准备的表现。

第一步是选择一套常见且常用的数据准备技术。

在这种情况下，假设输入变量是数字，我们将使用一系列转换来更改输入变量的比例，如最小最大缩放器、标准缩放器和[鲁棒缩放器](https://machinelearningmastery.com/robust-scaler-transforms-for-machine-learning/)，以及用于链接输入变量分布的转换，如[量化转换器](https://machinelearningmastery.com/quantile-transforms-for-machine-learning/)和[kbins 离散器](https://machinelearningmastery.com/discretization-transforms-for-machine-learning/)。最后，我们还将使用消除输入变量之间线性依赖关系的变换，如[主成分分析](https://machinelearningmastery.com/principal-components-analysis-for-dimensionality-reduction-in-python/)和[截断变量](https://machinelearningmastery.com/singular-value-decomposition-for-dimensionality-reduction-in-python/)。

[功能联合类](https://Sklearn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)可用于定义要执行的转换列表，其结果将被聚合在一起，即联合。这将创建一个包含大量列的新数据集。

列数的估计是 13 个输入变量乘以 5 个变换，或者 65 加上从主成分分析和奇异值分解降维方法输出的 14 列，得到总共约 79 个特征。

```py
...
# transforms for the feature union
transforms = list()
transforms.append(('mms', MinMaxScaler()))
transforms.append(('ss', StandardScaler()))
transforms.append(('rs', RobustScaler()))
transforms.append(('qt', QuantileTransformer(n_quantiles=100, output_distribution='normal')))
transforms.append(('kbd', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')))
transforms.append(('pca', PCA(n_components=7)))
transforms.append(('svd', TruncatedSVD(n_components=7)))
# create the feature union
fu = FeatureUnion(transforms)
```

然后我们可以创建一个建模[管道](https://Sklearn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)，第一步是特征联合，最后一步是逻辑回归模型。

```py
...
# define the model
model = LogisticRegression(solver='liblinear')
# define the pipeline
steps = list()
steps.append(('fu', fu))
steps.append(('m', model))
pipeline = Pipeline(steps=steps)
```

然后可以像以前一样，使用重复的分层 k-fold 交叉验证来评估管道。

将这些联系在一起，完整的示例如下所示。

```py
# data preparation as feature engineering for wine dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv'
df = read_csv(url, header=None)
data = df.values
X, y = data[:, :-1], data[:, -1]
# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))
# transforms for the feature union
transforms = list()
transforms.append(('mms', MinMaxScaler()))
transforms.append(('ss', StandardScaler()))
transforms.append(('rs', RobustScaler()))
transforms.append(('qt', QuantileTransformer(n_quantiles=100, output_distribution='normal')))
transforms.append(('kbd', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')))
transforms.append(('pca', PCA(n_components=7)))
transforms.append(('svd', TruncatedSVD(n_components=7)))
# create the feature union
fu = FeatureUnion(transforms)
# define the model
model = LogisticRegression(solver='liblinear')
# define the pipeline
steps = list()
steps.append(('fu', fu))
steps.append(('m', model))
pipeline = Pipeline(steps=steps)
# define the cross-validation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例评估模型表现，并报告平均和标准偏差分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到表现比基线表现有所提升，平均分类准确率约为 96.8%，而上一节为 95.3%。

```py
Accuracy: 0.968 (0.037)
```

尝试向 FeatureUnion 添加更多数据准备方法，看看是否可以提高表现。

**能不能取得更好的表现？**
让我知道你在下面的评论中发现了什么。

我们还可以使用特征选择将大约 80 个提取的特征减少到与模型最相关的那些特征的子集。除了降低模型的复杂性，它还可以通过移除不相关和冗余的输入特征来提升表现。

在这种情况下，我们将使用[递归特征消除](https://machinelearningmastery.com/rfe-feature-selection-in-python/)或 RFE 技术进行特征选择，并将其配置为选择 15 个最相关的特征。

```py
...
# define the feature selection
rfe = RFE(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=15)
```

然后，我们可以在*特征联合*之后和*物流配送*算法之前，将 RFE 特征选择添加到建模管道中。

```py
...
# define the pipeline
steps = list()
steps.append(('fu', fu))
steps.append(('rfe', rfe))
steps.append(('m', model))
pipeline = Pipeline(steps=steps)
```

将这些结合在一起，下面列出了带有特征选择的特征选择数据准备方法的完整示例。

```py
# data preparation as feature engineering with feature selection for wine dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv'
df = read_csv(url, header=None)
data = df.values
X, y = data[:, :-1], data[:, -1]
# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))
# transforms for the feature union
transforms = list()
transforms.append(('mms', MinMaxScaler()))
transforms.append(('ss', StandardScaler()))
transforms.append(('rs', RobustScaler()))
transforms.append(('qt', QuantileTransformer(n_quantiles=100, output_distribution='normal')))
transforms.append(('kbd', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')))
transforms.append(('pca', PCA(n_components=7)))
transforms.append(('svd', TruncatedSVD(n_components=7)))
# create the feature union
fu = FeatureUnion(transforms)
# define the feature selection
rfe = RFE(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=15)
# define the model
model = LogisticRegression(solver='liblinear')
# define the pipeline
steps = list()
steps.append(('fu', fu))
steps.append(('rfe', rfe))
steps.append(('m', model))
pipeline = Pipeline(steps=steps)
# define the cross-validation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例评估模型表现，并报告平均和标准偏差分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

同样，我们可以看到表现进一步提升，从所有提取特征的 96.8%提升到建模前使用特征选择的 98.9%左右。

```py
Accuracy: 0.989 (0.022)
```

**使用不同的特征选择技术，或者选择更多或更少的特征，可以获得更好的表现吗？**
让我知道你在下面的评论中发现了什么。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 相关教程

*   [标准分类和回归机器学习数据集的结果](https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/)

### 书

*   [特征工程与选择](https://amzn.to/3aydNGf)，2019。
*   [机器学习的特征工程](https://amzn.to/2XZJNR2)，2018。

### 蜜蜂

*   [sklearn . pipeline . pipeline API](https://Sklearn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)。
*   [sklearn . pipeline . feature union API](https://Sklearn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html)。

## 摘要

在本教程中，您发现了如何使用特征提取来准备表格数据。

具体来说，您了解到:

*   特征提取为表格数据的数据准备提供了一种替代方法，其中所有数据转换并行应用于原始输入数据，并组合在一起创建一个大型数据集。
*   如何使用数据准备的特征提取方法来提高标准类别数据集的模型表现。
*   如何将特征选择添加到特征提取建模管道中，以进一步提升标准数据集的建模表现。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。