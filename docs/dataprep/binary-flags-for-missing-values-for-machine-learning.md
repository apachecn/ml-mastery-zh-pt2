# 为机器学习的缺失值添加二元标志

> 原文：<https://machinelearningmastery.com/binary-flags-for-missing-values-for-machine-learning/>

最后更新于 2020 年 8 月 17 日

在用机器学习算法建模分类和回归预测问题时，缺失值会导致问题。

一种常见的方法是用计算出的统计数据(如列的平均值)替换缺失的值。这允许数据集按照正常方式建模，但不会向模型显示原始行包含缺失值。

解决这个问题的一种方法是包括附加的二元标志输入特征，该特征指示一行或一列是否包含被输入的缺失值。该附加信息可能对模型预测目标值有帮助，也可能没有帮助。

在本教程中，您将发现如何**为建模的缺失值**添加二元标志。

完成本教程后，您将知道:

*   如何在缺少值的类别数据集上加载和评估带有统计插补的模型。
*   如何添加一个标志来指示一行是否还有一个缺失值，并使用此新功能评估模型。
*   如何为每个缺少值的输入变量添加一个标志，并使用这些新特性评估模型。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 7 月更新**:修复了标志变量创建中的 bug。

![Add Binary Flags for Missing Values for Machine Learning](img/18bfeb77fed8c2170fa562197df9f0f9.png)

为机器学习的缺失值添加二元标志
Keith o connell 摄，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  马结肠数据集的输入
2.  带有缺失值二元标志的模型
3.  带有所有缺失值指示器的模型

## 马结肠数据集的输入

马绞痛数据集描述了患有绞痛的马的医学特征以及它们是活的还是死的。

有 300 行 26 个输入变量和一个输出变量。这是一个二分类预测任务，包括预测 1 如果马活了，2 如果马死了。

在这个数据集中，我们可以选择许多字段进行预测。在这种情况下，我们将预测问题是否是外科手术(列索引 23)，使其成为二分类问题。

对于许多列，数据集有许多[缺失值](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/)，其中每个缺失值都用问号字符(“？”)标记).

下面提供了数据集中带有标记缺失值的行的示例。

```py
2,1,530101,38.50,66,28,3,3,?,2,5,4,4,?,?,?,3,5,45.00,8.40,?,?,2,2,11300,00000,00000,2
1,1,534817,39.2,88,20,?,?,4,1,3,4,2,?,?,?,4,2,50,85,2,2,3,2,02208,00000,00000,2
2,1,530334,38.30,40,24,1,1,3,1,3,3,1,?,?,?,1,1,33.00,6.70,?,?,1,2,00000,00000,00000,1
1,9,5290409,39.10,164,84,4,1,6,2,2,4,4,1,2,5.00,3,?,48.00,7.20,3,5.30,2,1,02208,00000,00000,1
...
```

您可以在此了解有关数据集的更多信息:

*   [马绞痛数据集](https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv)
*   [马绞痛数据集描述](https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.names)

不需要下载数据集，因为我们将在工作示例中自动下载它。

使用 Python 在加载的数据集中用 NaN(而不是数字)值标记缺失值是最佳实践。

我们可以使用 [read_csv() Pandas](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) 函数加载数据集，并指定“ *na_values* ”来加载“？”作为缺失，用 NaN 值标记。

下面的示例下载数据集，标记“？”值为 NaN(缺少)并总结数据集的形状。

```py
# summarize the horse colic dataset
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')
data = dataframe.values
# split into input and output elements
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
print(X.shape, y.shape)
```

运行该示例会下载数据集并报告行数和列数，符合我们的预期。

```py
(300, 27) (300,)
```

接下来，我们可以在这个数据集上评估一个模型。

我们可以使用[simple 插补器类](https://Sklearn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)进行统计插补，并用每一列的平均值替换缺失的值。然后，我们可以在数据集上拟合一个随机森林模型。

有关如何使用简单估计器类的更多信息，请参见教程:

*   [机器学习中缺失值的统计插补](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/)

为了实现这一点，我们将定义一个管道，首先执行插补，然后拟合模型，并使用三次重复和 10 次折叠的重复分层 k-fold 交叉验证来评估该建模管道。

下面列出了完整的示例。

```py
# evaluate mean imputation and random forest for the horse colic dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')
# split into input and output elements
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# define modeling pipeline
model = RandomForestClassifier()
imputer = SimpleImputer()
pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例使用马结肠数据集上的平均统计插补来评估随机森林。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，管道实现了大约 86.2%的估计分类准确率。

```py
Mean Accuracy: 0.862 (0.056)
```

接下来，让我们看看是否可以通过提供更多关于缺失值的信息来提高模型的表现。

## 带有缺失值二元标志的模型

在前面的部分中，我们用计算的统计数据替换了缺失的值。

模型不知道丢失的值已被替换。

在进行预测时，了解一行是否包含缺失值可能对模型有用。

向模型公开这一知识的一种方法是提供一个额外的列，该列是一个二元标志，指示该行是否有丢失的值。

*   0:行不包含缺失值。
*   1:行包含一个缺失值(过去/将来会被估计)。

这可以直接在加载的数据集上实现。首先，我们可以对每一行的值求和，以创建一个新的列，其中如果该行至少包含一个 NaN，那么总和将是一个 NaN。

然后我们可以将新列中的所有值标记为 1(如果它们包含 NaN)，否则标记为 0。

最后，我们可以将该列添加到加载的数据集中。

将这些联系在一起，下面列出了添加二元标志来指示每行中一个或多个缺失值的完整示例。

```py
# add a binary flag that indicates if a row contains a missing value
from numpy import isnan
from numpy import hstack
from pandas import read_csv
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')
# split into input and output elements
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
print(X.shape)
# sum each row where rows with a nan will sum to nan
a = X.sum(axis=1)
# mark all non-nan as 0
a[~isnan(a)] = 0
# mark all nan as 1
a[isnan(a)] = 1
a = a.reshape((len(a), 1))
# add to the dataset as another column
X = hstack((X, a))
print(X.shape)
```

运行该示例首先下载数据集，并按照预期报告行数和列数。

然后创建新的二进制变量，指示一行是否包含缺失值，并将其添加到输入变量的末尾。然后报告输入数据的形状，确认增加了特征，从 27 列到 28 列。

```py
(300, 27)
(300, 28)
```

然后，我们可以像上一节一样使用额外的二元标志来评估模型，看看它是否会影响模型表现。

下面列出了完整的示例。

```py
# evaluate model performance with a binary flag for missing values and imputed missing
from numpy import isnan
from numpy import hstack
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')
# split into input and output elements
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# sum each row where rows with a nan will sum to nan
a = X.sum(axis=1)
# mark all non-nan as 0
a[~isnan(a)] = 0
# mark all nan as 1
a[isnan(a)] = 1
a = a.reshape((len(a), 1))
# add to the dataset as another column
X = hstack((X, a))
# define modeling pipeline
model = RandomForestClassifier()
imputer = SimpleImputer()
pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例报告了具有附加特征和插补的马结肠数据集的平均和标准偏差分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们看到表现从 86.2%适度提升到 86.3%。差异很小，可能没有统计学意义。

```py
Mean Accuracy: 0.863 (0.055)
```

此数据集中的大多数行都有缺失值，这种方法在缺失值较少的数据集上可能更有好处。

接下来，让我们看看是否可以向模型提供更多关于缺失值的信息。

## 带有所有缺失值指示器的模型

在前一节中，我们添加了一个额外的列来指示一行是否包含缺失值。

下一步是指出每个输入值是否缺失和估计。这实际上为每个包含缺失值的输入变量增加了一列，并可能为模型带来好处。

这可以通过在定义[简单估计器实例](https://Sklearn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)时将“ *add_indicator* ”参数设置为 *True* 来实现。

```py
...
# impute and mark missing values
X = SimpleImputer(add_indicator=True).fit_transform(X)
```

我们可以用一个成功的例子来证明这一点。

下面的示例像以前一样加载 horse colic 数据集，然后估计整个数据集的缺失值，并为每个缺失值的输入变量添加指示变量

```py
# impute and add indicators for columns with missing values
from pandas import read_csv
from sklearn.impute import SimpleImputer
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')
data = dataframe.values
# split into input and output elements
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
print(X.shape)
# impute and mark missing values
X = SimpleImputer(strategy='mean', add_indicator=True).fit_transform(X)
print(X.shape)
```

运行该示例首先按照预期下载并汇总数据集的形状，然后应用插补并添加二进制(1 和 0 值)列，以指示每行是否包含给定输入变量的缺失值。

我们可以看到输入变量的数量从 27 个增加到了 48 个，这表明增加了 21 个二进制输入变量，反过来，27 个输入变量中的 21 个必须包含至少一个缺失值。

```py
(300, 27)
(300, 48)
```

接下来，我们可以使用这些附加信息来评估模型。

下面完整的例子演示了这一点。

```py
# evaluate imputation with added indicators features on the horse colic dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')
# split into input and output elements
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# define modeling pipeline
model = RandomForestClassifier()
imputer = SimpleImputer(add_indicator=True)
pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

运行该示例报告了马结肠数据集的平均和标准偏差分类准确率，以及附加的指标特征和插补。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们看到表现从上一部分的 86.3%提升到了 86.7%。

这可能提供强有力的证据，表明在这个数据集和所选模型上，为输入的每一列添加一个标志是更好的策略。

```py
Mean Accuracy: 0.867 (0.055)
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 相关教程

*   [标准机器学习数据集的最佳结果](https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/)
*   [机器学习中缺失值的统计插补](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/)
*   [如何用 Python 处理缺失数据](https://machinelearningmastery.com/handle-missing-data-python/)

## 摘要

在本教程中，您发现了如何为建模的缺失值添加二元标志。

具体来说，您了解到:

*   如何在缺少值的类别数据集上加载和评估带有统计插补的模型。
*   如何添加一个标志来指示一行是否还有一个缺失值，并使用此新功能评估模型。
*   如何为每个缺少值的输入变量添加一个标志，并使用这些新特性评估模型。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。