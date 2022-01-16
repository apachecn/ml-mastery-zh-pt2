# 如何将列转换器用于数据准备

> 原文：<https://machinelearningmastery.com/columntransformer-for-numerical-and-categorical-data/>

最后更新于 2020 年 12 月 31 日

在拟合机器学习模型之前，您必须使用数据转换来准备原始数据。

这是确保您最好地将预测建模问题的结构暴露给学习算法所必需的。

当所有输入变量都是相同类型时，应用数据转换(如缩放或编码分类变量)非常简单。当您有一个混合类型的数据集，并且您想要有选择地将数据转换应用于一些(但不是全部)输入要素时，这可能会很有挑战性。

值得庆幸的是，Sklearn Python 机器学习库提供了 **ColumnTransformer** ，允许您选择性地将数据转换应用到数据集中的不同列。

在本教程中，您将了解如何使用 ColumnTransformer 有选择地将数据转换应用于具有混合数据类型的数据集中的列。

完成本教程后，您将知道:

*   对具有混合数据类型的数据集使用数据转换的挑战。
*   如何定义、调整和使用 ColumnTransformer 选择性地将数据转换应用于列。
*   如何处理具有混合数据类型的真实数据集，并使用 ColumnTransformer 对分类和数字数据列应用不同的转换。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 12 月更新**:修正了 API 示例中的小错别字。

![Use the ColumnTransformer for Numerical and Categorical Data in Python](img/7606632cf9520fb67ac4f712c468fb67.png)

使用 Python 中的数值和类别数据的 ColumnTransformer】图片由 [Kari](https://flickr.com/photos/designsbykari/6205452745/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  转换不同数据类型的挑战
2.  如何使用列转换器
3.  鲍鱼回归数据集的数据准备

## 转换不同数据类型的挑战

建模前准备数据很重要。

这可能涉及替换丢失的值、缩放数值和一个热编码类别数据。

可以使用 Sklearn 库执行数据转换；例如，[simplementor](https://Sklearn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)类可以用来替换缺失的值， [MinMaxScaler](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) 类可以用来缩放数值， [OneHotEncoder](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) 可以用来编码分类变量。

例如:

```py
...
# prepare transform
scaler = MinMaxScaler()
# fit transform on training data
scaler.fit(train_X)
# transform training data
train_X = scaler.transform(train_X)
```

不同变换的序列也可以使用[管道](https://Sklearn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)链接在一起，例如输入缺失值，然后缩放数值。

例如:

```py
...
# define pipeline
pipeline = Pipeline(steps=[('i', SimpleImputer(strategy='median')), ('s', MinMaxScaler())])
# transform training data
train_X = pipeline.fit_transform(train_X)
```

希望对输入数据中的不同列执行不同的数据准备技术是非常常见的。

例如，您可能希望用一个中值来估计缺失的数值，然后缩放这些值，并使用最频繁的值和一个热编码类别来估计缺失的分类值。

传统上，这将需要您分离数值和类别数据，然后在将列组合在一起之前，手动对这些要素组应用转换，以便拟合和评估模型。

现在，您可以使用 ColumnTransformer 为您执行此操作。

## 如何使用列转换器

[ColumnTransformer](https://Sklearn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) 是 Sklearn Python 机器学习库中的一个类，允许您选择性地应用数据准备转换。

例如，它允许您将特定的转换或转换序列仅应用于数字列，将单独的转换序列仅应用于分类列。

要使用 ColumnTransformer，必须指定一个转换器列表。

每个转换器都是一个三元组，它定义了转换器的名称、要应用的转换以及要应用的列索引。例如:

*   (名称、对象、列)

例如，下面的 ColumnTransformer 对第 0 列和第 1 列应用了一个 OneHotEncoder。

```py
...
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), [0, 1])])
```

以下示例对数值列 0 和 1 应用中值输入的简单估计器，对分类列 2 和 3 应用最频繁输入的简单估计器。

```py
...
t = [('num', SimpleImputer(strategy='median'), [0, 1]), ('cat', SimpleImputer(strategy='most_frequent'), [2, 3])]
transformer = ColumnTransformer(transformers=t)
```

默认情况下，任何未在“*变压器*列表中指定的列都会从数据集中删除；这可以通过设置“*余数*参数来更改。

设置*余数=“通过”*将意味着所有未在“*变压器*列表中指定的列都将在没有转换的情况下通过，而不是被删除。

例如，如果第 0 列和第 1 列是数字的，第 2 列和第 3 列是分类的，并且我们只想转换类别数据并不变地通过数字列，那么我们可以如下定义 ColumnTransformer:

```py
...
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), [2, 3])], remainder='passthrough')
```

一旦定义了转换器，就可以使用它来转换数据集。

例如:

```py
...
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), [0, 1])])
# transform training data
train_X = transformer.fit_transform(train_X)
```

ColumnTransformer 也可以在管道中使用，以便在对转换后的数据拟合模型之前，有选择地准备数据集的列。

这是最有可能的用例，因为它确保在拟合模型和进行预测时(例如，通过交叉验证在测试数据集上评估模型或在未来对新数据进行预测时)对原始数据自动执行转换。

例如:

```py
...
# define model
model = LogisticRegression()
# define transform
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), [0, 1])])
# define pipeline
pipeline = Pipeline(steps=[('t', transformer), ('m',model)])
# fit the pipeline on the transformed data
pipeline.fit(train_X, train_y)
# make predictions
yhat = pipeline.predict(test_X)
```

现在我们已经熟悉了如何配置和使用 ColumnTransformer，让我们来看一个工作示例。

## 鲍鱼回归数据集的数据准备

鲍鱼数据集是一个标准的机器学习问题，包括在给定鲍鱼测量值的情况下预测鲍鱼的年龄。

您可以下载数据集并在此了解更多信息:

*   [下载鲍鱼数据集(鲍鱼. csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv)
*   [了解更多鲍鱼数据集(鲍鱼. name)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.names)

数据集有 4，177 个示例，8 个输入变量，目标变量是一个整数。

一个简单的模型可以通过预测平均值达到大约 2.363(标准 0.092)的平均绝对误差(MAE)，通过 10 倍交叉验证进行评估。

我们可以用支持向量机模型( [SVR](https://Sklearn.org/stable/modules/generated/sklearn.svm.SVR.html) )将其建模为回归预测建模问题。

查看数据，您可以看到前几行如下:

```py
M,0.455,0.365,0.095,0.514,0.2245,0.101,0.15,15
M,0.35,0.265,0.09,0.2255,0.0995,0.0485,0.07,7
F,0.53,0.42,0.135,0.677,0.2565,0.1415,0.21,9
M,0.44,0.365,0.125,0.516,0.2155,0.114,0.155,10
I,0.33,0.255,0.08,0.205,0.0895,0.0395,0.055,7
...
```

我们可以看到第一列是分类的，其余的列是数字的。

我们可能希望对第一列进行热编码，并对剩余的数字列进行规范化，这可以使用 ColumnTransformer 来实现。

首先，我们需要加载数据集。我们可以使用 [read_csv()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) Pandas 函数直接从 URL 加载数据集，然后将数据拆分为两个数据帧:一个用于输入，一个用于输出。

下面列出了加载数据集的完整示例。

```py
# load the dataset
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv'
dataframe = read_csv(url, header=None)
# split into inputs and outputs
last_ix = len(dataframe.columns) - 1
X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
print(X.shape, y.shape)
```

**注意**:如果在从 URL 加载数据集时遇到问题，可以下载名为“*鲍鱼. csv* 的 CSV 文件，并将其放在与您的 Python 文件相同的目录中，并将调用更改为 *read_csv()* ，如下所示:

```py
...
dataframe = read_csv('abalone.csv', header=None)
```

运行该示例，我们可以看到数据集被正确加载，并被分成八个输入列和一个目标列。

```py
(4177, 8) (4177,)
```

接下来，我们可以使用*select _ dt types()*函数来选择匹配不同数据类型的列索引。

我们感兴趣的是熊猫中标记为“ *float64* 或“ *int64* 的数字列列表，以及熊猫中标记为“ *object* 或“ *bool* 类型的分类列列表。

```py
...
# determine categorical and numerical features
numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
categorical_ix = X.select_dtypes(include=['object', 'bool']).columns
```

然后，我们可以在 ColumnTransformer 中使用这些列表对分类变量进行热编码，这应该只是第一列。

我们还可以使用数字列列表来规范化剩余的数据。

```py
...
# define the data preparation for the columns
t = [('cat', OneHotEncoder(), categorical_ix), ('num', MinMaxScaler(), numerical_ix)]
col_transform = ColumnTransformer(transformers=t)
```

接下来，我们可以定义我们的 SVR 模型，并定义一个管道，该管道首先使用 ColumnTransformer，然后在准备好的数据集上拟合模型。

```py
...
# define the model
model = SVR(kernel='rbf',gamma='scale',C=100)
# define the data preparation and modeling pipeline
pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])
```

最后，我们可以使用 10 倍交叉验证来评估模型，并计算管道的所有 10 次评估的平均绝对误差。

```py
...
# define the model cross-validation configuration
cv = KFold(n_splits=10, shuffle=True, random_state=1)
# evaluate the pipeline using cross validation and calculate MAE
scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# convert MAE scores to positive values
scores = absolute(scores)
# summarize the model performance
print('MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
```

将这些结合在一起，完整的示例如下所示。

```py
# example of using the ColumnTransformer for the Abalone dataset
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv'
dataframe = read_csv(url, header=None)
# split into inputs and outputs
last_ix = len(dataframe.columns) - 1
X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
print(X.shape, y.shape)
# determine categorical and numerical features
numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
categorical_ix = X.select_dtypes(include=['object', 'bool']).columns
# define the data preparation for the columns
t = [('cat', OneHotEncoder(), categorical_ix), ('num', MinMaxScaler(), numerical_ix)]
col_transform = ColumnTransformer(transformers=t)
# define the model
model = SVR(kernel='rbf',gamma='scale',C=100)
# define the data preparation and modeling pipeline
pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])
# define the model cross-validation configuration
cv = KFold(n_splits=10, shuffle=True, random_state=1)
# evaluate the pipeline using cross validation and calculate MAE
scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# convert MAE scores to positive values
scores = absolute(scores)
# summarize the model performance
print('MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例使用 10 倍交叉验证来评估数据准备管道。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们实现了大约 1.4 的平均 MAE，这比基线分数 2.3 要好。

```py
(4177, 8) (4177,)
MAE: 1.465 (0.047)
```

现在，您有了一个模板，可以在具有混合数据类型的数据集中使用 ColumnTransformer，您可以在将来的项目中使用和调整该模板。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 应用程序接口

*   [硬化。化合物。ColumnTransformer API](https://Sklearn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) 。
*   [熊猫. read_csv API](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) 。
*   [巩理。计费。简单计费 API](https://Sklearn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) 。
*   [硬化。预处理。OneHotEncoder API](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) 。
*   [硬化。预处理。MinMaxScaler API](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
*   [sklearn . pipeline . pipeline API](https://Sklearn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)。

## 摘要

在本教程中，您发现了如何使用 ColumnTransformer 有选择地将数据转换应用于具有混合数据类型的数据集中的列。

具体来说，您了解到:

*   对具有混合数据类型的数据集使用数据转换的挑战。
*   如何定义、调整和使用 ColumnTransformer 选择性地将数据转换应用于列。
*   如何处理具有混合数据类型的真实数据集，并使用 ColumnTransformer 对分类和数字数据列应用不同的转换。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。