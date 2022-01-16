# 类别数据的顺序编码和单热编码

> 原文：<https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/>

最后更新于 2020 年 8 月 17 日

机器学习模型要求所有的输入和输出变量都是数字。

这意味着，如果您的数据包含类别数据，您必须将其编码为数字，然后才能拟合和评估模型。

两种最流行的技术是**顺序编码**和**单热编码**。

在本教程中，您将发现如何对分类机器学习数据使用编码方案。

完成本教程后，您将知道:

*   当处理机器学习算法的类别数据时，编码是必需的预处理步骤。
*   如何对具有自然排序的分类变量使用序数编码？
*   如何对没有自然排序的分类变量使用一热编码？

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Ordinal and One-Hot Encoding Transforms for Machine Learning](img/3731b61d0a386f84e830b26cf835e3f3.png)

机器学习的顺序和单热编码转换
图片由[费利佩·瓦尔杜加](https://flickr.com/photos/felipevalduga/20666899856/)提供，保留部分权利。

## 教程概述

本教程分为六个部分；它们是:

1.  名义变量和顺序变量
2.  类别数据编码
    1.  序数编码
    2.  一次性编码
    3.  虚拟变量编码
3.  乳腺癌数据集
4.  序码变换
5.  OneHotEncoder 转换
6.  常见问题

## 名义变量和顺序变量

顾名思义，数字数据包括仅由数字组成的特征，例如整数或浮点值。

[类别数据](https://en.wikipedia.org/wiki/Categorical_variable)是包含标签值而不是数值的变量。

可能值的数量通常限于一个固定的集合。

分类变量通常被称为名义变量。

一些例子包括:

*   一个“*宠物*”变量的值为:“*狗*”和“*猫*”。
*   一个“*颜色*”变量，值为:“*红色*”、“*绿色*”和“*蓝色*”。
*   一个“*放置*”变量的值为:“*第一个*”、“*第二个*”和“*第三个*”。

每个值代表不同的类别。

有些类别可能彼此有自然的关系，例如自然的排序。

上面的“ *place* ”变量确实有一个自然的值排序。这种类型的分类变量被称为序数变量，因为值可以排序或排序。

通过将数值变量的范围划分为若干个区间并为每个区间赋值，可以将数值变量转换为序数变量。例如，一个介于 1 和 10 之间的数字变量可以分为一个序数变量，该序数变量有 5 个具有序数关系的标签:1-2、3-4、5-6、7-8、9-10。这叫做离散化。

*   **名义变量** ( *分类*)。变量由一组有限的离散值组成，这些值之间没有关系。
*   **序数变量**。变量由一组有限的离散值组成，这些值之间具有等级排序。

有些算法可以直接处理类别数据。

例如，决策树可以直接从类别数据中学习，不需要数据转换(这取决于具体的实现)。

许多机器学习算法不能直接对标签数据进行操作。它们要求所有输入变量和输出变量都是数字。

一般来说，这主要是机器学习算法有效实现的一个限制，而不是算法本身的硬性限制。

机器学习算法的一些实现要求所有数据都是数字的。比如 Sklearn 就有这个要求。

这意味着类别数据必须转换成数字形式。如果分类变量是输出变量，您可能还希望将模型的预测转换回分类形式，以便在某些应用中呈现或使用它们。

## 类别数据编码

有三种常见的方法将序数和分类变量转换为数值。它们是:

*   序数编码
*   一次性编码
*   虚拟变量编码

让我们依次仔细看看每一个。

### 序数编码

在序数编码中，每个唯一的类别值都被赋予一个整数值。

例如“*红色*为 1，“*绿色*为 2，*蓝色*为 3。

这被称为序数编码或整数编码，并且很容易可逆。通常，使用从零开始的整数值。

对于某些变量，序数编码可能就足够了。整数值之间具有自然的有序关系，机器学习算法可能能够理解和利用这种关系。

它是序数变量的自然编码。对于分类变量，它强加了一种序数关系，而这种关系可能不存在。这可能会导致问题，可以使用一个热编码来代替。

这种序数编码转换可以通过[序数编码器类](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)在 Sklearn Python 机器学习库中获得。

默认情况下，它将按照数据中观察到的顺序为标签分配整数。如果需要特定的顺序，可以通过“*类别*”参数将其指定为具有所有预期标签的等级顺序的列表。

我们可以通过将颜色类别“红色”、“绿色”和“蓝色”转换为整数来演示这个类的用法。首先对类别进行排序，然后应用数字。对于字符串，这意味着标签按字母顺序排序，蓝色=0，绿色=1，红色=2。

下面列出了完整的示例。

```py
# example of a ordinal encoding
from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
# define data
data = asarray([['red'], ['green'], ['blue']])
print(data)
# define ordinal encoding
encoder = OrdinalEncoder()
# transform data
result = encoder.fit_transform(data)
print(result)
```

运行该示例首先报告 3 行标签数据，然后报告序号编码。

我们可以看到，这些数字是按照我们的预期分配给标签的。

```py
[['red']
 ['green']
 ['blue']]
[[2.]
 [1.]
 [0.]]
```

这个序数编码器类用于输入变量，这些变量被组织成行和列，例如矩阵。

如果分类预测建模问题需要对分类目标变量进行编码，那么可以使用[标签编码器类](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)。它做的事情与 OrdinalEncoder 相同，尽管它期望对单个目标变量进行一维输入。

### 一次性编码

对于不存在序数关系的分类变量，整数编码充其量可能不够，最坏可能会误导模型。

通过序数编码强制序数关系，并允许模型假设类别之间的自然排序，可能会导致较差的表现或意外的结果(类别之间的预测)。

在这种情况下，一个热编码可以应用于序数表示。这是删除整数编码变量，并为变量中的每个唯一整数值添加一个新的二进制变量的地方。

> 每一位代表一个可能的类别。如果变量不能同时属于多个类别，那么组中只有一位可以是“开”这被称为一热编码…

—第 78 页，[机器学习的特征工程](https://amzn.to/3b9tp3s)，2018。

在“ *color* ”变量示例中，有三个类别，因此需要三个二元变量。颜色的二进制变量中有一个“1”值，其他颜色的二进制变量中有“0”值。

通过 [OneHotEncoder 类](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)，在 Sklearn Python 机器学习库中可以获得这种一次性编码转换。

我们可以演示 OneHotEncoder 在颜色类别上的用法。首先对类别进行排序，在这种情况下是按字母顺序排序，因为它们是字符串，然后依次为每个类别创建二进制变量。这意味着对于第一个二进制变量，蓝色将表示为[1，0，0]，然后是绿色，最后是红色。

下面列出了完整的示例。

```py
# example of a one hot encoding
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
# define data
data = asarray([['red'], ['green'], ['blue']])
print(data)
# define one hot encoding
encoder = OneHotEncoder(sparse=False)
# transform data
onehot = encoder.fit_transform(data)
print(onehot)
```

运行该示例首先列出了三行标签数据，然后是一个热编码，与我们对 3 个二进制变量的期望相匹配，顺序为“蓝色”、“绿色”和“红色”。

```py
[['red']
 ['green']
 ['blue']]
[[0\. 0\. 1.]
 [0\. 1\. 0.]
 [1\. 0\. 0.]]
```

如果您知道数据中预期的所有标签，则可以通过列表形式的“*类别*”参数来指定它们。

编码器适合训练数据集，如果不指定标签列表，该数据集可能至少包含每个分类变量的所有预期标签的一个示例。如果新数据包含训练数据集中未出现的类别，可以将“ *handle_unknown* ”参数设置为“ *ignore* ”以不引发错误，这将导致每个标签的值为零。

### 虚拟变量编码

单一热编码为每个类别创建一个二进制变量。

问题是，这种表示包括冗余。例如，如果我们知道[1，0，0]代表“*蓝色*”,[ 0，1，0]代表“*绿色*”,我们不需要另一个二进制变量来代表“*红色*”，相反，我们可以单独为“*蓝色*”和“*绿色*”使用 0 值，例如[0，0]。

这被称为虚拟变量编码，并且总是用 C-1 二进制变量表示 C 类。

> 当预测值有 C 个可能值，且仅使用 C–1 个虚拟变量时，可以计算矩阵求逆，对比方法称为全秩参数化

—第 95 页，[特征工程与选择](https://amzn.to/2Wu7zlL)，2019 年。

除了稍微减少冗余之外，一些模型还需要虚拟变量表示。

例如，在线性回归模型(以及具有偏差项的其他回归模型)的情况下，一次热编码将使输入数据的矩阵变成奇异的，这意味着它不能被反转，并且不能使用[线性代数](https://machinelearningmastery.com/linear-algebra-machine-learning-7-day-mini-course/)来计算线性回归系数。对于这些类型的模型，必须使用虚拟变量编码来代替。

> 如果模型包含一个截距并包含虚拟变量[…]，那么[…]列将与截距相加(按行)，这种线性组合将阻止矩阵求逆的计算(因为它是奇异的)。

—第 95 页，[特征工程与选择](https://amzn.to/2Wu7zlL)，2019 年。

在评估机器学习算法时，我们在实践中很少遇到这个问题，当然，除非我们使用线性回归。

> …有时一整套虚拟变量是有用的。例如，当虚拟变量编码预测器的所有信息时，基于树的模型中的拆分更容易解释。我们建议在使用基于树的模型时使用完整的 if 虚拟变量集。

—第 56 页，[应用预测建模](https://amzn.to/3b2LHTL)，2013 年。

我们可以使用 *OneHotEncoder* 类来实现一个虚拟编码和一个热编码。

可以设置“ *drop* ”参数来指示哪个类别将被分配所有零值，称为“*基线*”。我们可以先将此设置为“*”，这样就可以使用第一个类别。当标签按字母顺序排序时，第一个“蓝色”标签将是第一个，并将成为基线。*

 *> 虚拟变量总是比级别数少一个。没有虚拟变量[…]的水平称为基线。

—第 86 页，[R，2014 年《统计学习与应用导论》](https://amzn.to/3dfRtms)。

我们可以用我们的颜色类别来证明这一点。下面列出了完整的示例。

```py
# example of a dummy variable encoding
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
# define data
data = asarray([['red'], ['green'], ['blue']])
print(data)
# define one hot encoding
encoder = OneHotEncoder(drop='first', sparse=False)
# transform data
onehot = encoder.fit_transform(data)
print(onehot)
```

运行该示例首先列出了分类变量的三行，然后是虚拟变量编码，显示绿色编码为[1，0]，红色编码为[0，1]，蓝色编码为[0，0]。

```py
[['red']
 ['green']
 ['blue']]
[[0\. 1.]
 [1\. 0.]
 [0\. 0.]]
```

现在我们已经熟悉了编码分类变量的三种方法，让我们来看看一个包含分类变量的数据集。

## 乳腺癌数据集

作为本教程的基础，我们将使用自 20 世纪 80 年代以来在机器学习中广泛研究的“[乳腺癌](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer)”数据集。

数据集将乳腺癌患者数据分类为癌症复发或不复发。有 286 个例子和 9 个输入变量。这是一个二分类问题。

该数据集上合理的分类准确度分数在 68%到 73%之间。我们将针对这个区域，但请注意，本教程中的模型并未优化:它们旨在演示编码方案。

不需要下载数据集，因为我们将直接从代码示例中访问它。

*   [乳腺癌数据集(乳腺癌. csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv)
*   [乳腺癌数据集描述(乳腺癌.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.names)

查看数据，我们可以看到所有九个输入变量都是绝对的。

具体来说，所有变量都是带引号的字符串。有些变量显示了一个明显的数值范围的序数关系(如年龄范围)，有些则没有。

```py
'40-49','premeno','15-19','0-2','yes','3','right','left_up','no','recurrence-events'
'50-59','ge40','15-19','0-2','no','1','right','central','no','no-recurrence-events'
'50-59','ge40','35-39','0-2','no','2','left','left_low','no','recurrence-events'
'40-49','premeno','35-39','0-2','yes','3','right','left_low','yes','no-recurrence-events'
'40-49','premeno','30-34','3-5','yes','2','left','right_up','no','recurrence-events'
...
```

请注意，该数据集缺少标有“ *nan* 值的值。

在本教程中，我们将保持这些值不变，并使用编码方案将“nan”编码为另一个值。这是处理分类变量缺失值的一种可能且非常合理的方法。

我们可以使用熊猫库将这个数据集加载到内存中。

```py
...
# load the dataset
dataset = read_csv(url, header=None)
# retrieve the array of data
data = dataset.values
```

加载后，我们可以将列拆分为输入(X)和输出(y)进行建模。

```py
...
# separate into input and output columns
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)
```

利用这个函数，下面列出了加载和汇总原始类别数据集的完整示例。

```py
# load and summarize the dataset
from pandas import read_csv
# define the location of the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
# load the dataset
dataset = read_csv(url, header=None)
# retrieve the array of data
data = dataset.values
# separate into input and output columns
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)
# summarize
print('Input', X.shape)
print('Output', y.shape)
```

运行该示例会报告数据集的输入和输出元素的大小。

我们可以看到我们有 286 个例子和 9 个输入变量。

```py
Input (286, 9)
Output (286,)
```

现在我们已经熟悉了数据集，让我们看看如何对它进行编码以进行建模。

## 序码变换

序数编码包括将每个唯一标签映射到一个整数值。

只有当类别之间存在已知的关系时，这种类型的编码才是真正合适的。对于我们数据集中的一些变量，这种关系确实存在，理想情况下，在准备数据时应该利用这种关系。

在这种情况下，我们将忽略任何可能存在的序数关系，并假设所有变量都是分类的。使用序数编码仍然是有帮助的，至少作为其他编码方案的参考点。

我们可以使用 scikit 中的[序数编码器](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)-学会将每个变量编码成整数。这是一个灵活的类，如果已知任何这样的顺序，它确实允许将类别的顺序指定为参数。

注意:我将把它作为一个练习留给您来更新下面的示例，以尝试为那些具有自然顺序的变量指定顺序，并查看它是否对模型表现有影响。

一旦定义，我们就可以调用 *fit_transform()* 函数，并将其传递给我们的数据集，以创建数据集的分位数转换版本。

```py
...
# ordinal encode input variables
ordinal = OrdinalEncoder()
X = ordinal.fit_transform(X)
```

我们也可以用同样的方式准备目标。

```py
...
# ordinal encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
```

让我们在乳腺癌数据集上尝试一下。

下面列出了创建乳腺癌数据集的序数编码转换并总结结果的完整示例。

```py
# ordinal encode the breast cancer dataset
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
# define the location of the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
# load the dataset
dataset = read_csv(url, header=None)
# retrieve the array of data
data = dataset.values
# separate into input and output columns
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)
# ordinal encode input variables
ordinal_encoder = OrdinalEncoder()
X = ordinal_encoder.fit_transform(X)
# ordinal encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# summarize the transformed data
print('Input', X.shape)
print(X[:5, :])
print('Output', y.shape)
print(y[:5])
```

运行该示例会转换数据集并报告结果数据集的形状。

除了所有字符串值现在都是整数值之外，我们希望行数(在本例中是列数)保持不变。

不出所料，在这种情况下，我们可以看到变量的数量没有变化，但是所有的值现在都是序数编码的整数。

```py
Input (286, 9)
[[2\. 2\. 2\. 0\. 1\. 2\. 1\. 2\. 0.]
 [3\. 0\. 2\. 0\. 0\. 0\. 1\. 0\. 0.]
 [3\. 0\. 6\. 0\. 0\. 1\. 0\. 1\. 0.]
 [2\. 2\. 6\. 0\. 1\. 2\. 1\. 1\. 1.]
 [2\. 2\. 5\. 4\. 1\. 1\. 0\. 4\. 0.]]
Output (286,)
[1 0 1 0 1]
```

接下来，让我们用这种编码在这个数据集上评估机器学习。

对变量进行编码的最佳实践是在训练数据集上进行编码，然后将其应用于训练和测试数据集。

我们将首先分割数据集，然后在训练集上准备编码，并将其应用于测试集。

```py
...
# split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

然后，我们可以在训练数据集上拟合*序数编码器*，并使用它来转换训练和测试数据集。

```py
...
# ordinal encode input variables
ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train)
X_train = ordinal_encoder.transform(X_train)
X_test = ordinal_encoder.transform(X_test)
```

同样的方法可以用来准备目标变量。然后，我们可以在训练数据集上拟合逻辑回归算法，并在测试数据集上对其进行评估。

下面列出了完整的示例。

```py
# evaluate logistic regression on the breast cancer dataset with an ordinal encoding
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
# define the location of the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
# load the dataset
dataset = read_csv(url, header=None)
# retrieve the array of data
data = dataset.values
# separate into input and output columns
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)
# split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# ordinal encode input variables
ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train)
X_train = ordinal_encoder.transform(X_train)
X_test = ordinal_encoder.transform(X_test)
# ordinal encode target variable
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
# define the model
model = LogisticRegression()
# fit on the training set
model.fit(X_train, y_train)
# predict on test set
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))
```

运行该示例以正确的方式准备数据集，然后对转换后的数据进行模型拟合评估。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，该模型实现了大约 75.79%的分类准确率，这是一个合理的分数。

```py
Accuracy: 75.79
```

接下来，让我们仔细看看一热编码。

## OneHotEncoder 转换

单热编码适用于类别之间不存在关系的类别数据。

Sklearn 库提供了 [OneHotEncoder](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) 类来自动对一个或多个变量进行热编码。

默认情况下， *OneHotEncoder* 将输出具有[稀疏表示](https://machinelearningmastery.com/sparse-matrices-for-machine-learning/)的数据，这在编码表示中大多数值为 0 的情况下是有效的。我们将通过将“*稀疏*”参数设置为*假*来禁用此功能，以便我们可以查看编码的效果。

一旦定义，我们就可以调用 *fit_transform()* 函数，并将其传递给我们的数据集，以创建数据集的分位数转换版本。

```py
...
# one hot encode input variables
onehot_encoder = OneHotEncoder(sparse=False)
X = onehot_encoder.fit_transform(X)
```

和以前一样，我们必须对目标变量进行标签编码。

下面列出了创建乳腺癌数据集的一次性编码转换并总结结果的完整示例。

```py
# one-hot encode the breast cancer dataset
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
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
# one hot encode input variables
onehot_encoder = OneHotEncoder(sparse=False)
X = onehot_encoder.fit_transform(X)
# ordinal encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# summarize the transformed data
print('Input', X.shape)
print(X[:5, :])
```

运行该示例会转换数据集并报告结果数据集的形状。

我们希望行数保持不变，但列数会急剧增加。

不出所料，在这种情况下，我们可以看到变量的数量从 9 个跃升到 43 个，现在所有的值都是二进制值 0 或 1。

```py
Input (286, 43)
[[0\. 0\. 1\. 0\. 0\. 0\. 0\. 0\. 1\. 0\. 0\. 1\. 0\. 0\. 0\. 0\. 0\. 0\. 0\. 0\. 1\. 0\. 0\. 0.
  0\. 0\. 0\. 0\. 1\. 0\. 0\. 0\. 1\. 0\. 1\. 0\. 0\. 1\. 0\. 0\. 0\. 1\. 0.]
 [0\. 0\. 0\. 1\. 0\. 0\. 1\. 0\. 0\. 0\. 0\. 1\. 0\. 0\. 0\. 0\. 0\. 0\. 0\. 0\. 1\. 0\. 0\. 0.
  0\. 0\. 0\. 1\. 0\. 0\. 1\. 0\. 0\. 0\. 1\. 1\. 0\. 0\. 0\. 0\. 0\. 1\. 0.]
 [0\. 0\. 0\. 1\. 0\. 0\. 1\. 0\. 0\. 0\. 0\. 0\. 0\. 0\. 0\. 1\. 0\. 0\. 0\. 0\. 1\. 0\. 0\. 0.
  0\. 0\. 0\. 1\. 0\. 0\. 0\. 1\. 0\. 1\. 0\. 0\. 1\. 0\. 0\. 0\. 0\. 1\. 0.]
 [0\. 0\. 1\. 0\. 0\. 0\. 0\. 0\. 1\. 0\. 0\. 0\. 0\. 0\. 0\. 1\. 0\. 0\. 0\. 0\. 1\. 0\. 0\. 0.
  0\. 0\. 0\. 0\. 1\. 0\. 0\. 0\. 1\. 0\. 1\. 0\. 1\. 0\. 0\. 0\. 0\. 0\. 1.]
 [0\. 0\. 1\. 0\. 0\. 0\. 0\. 0\. 1\. 0\. 0\. 0\. 0\. 0\. 1\. 0\. 0\. 0\. 0\. 0\. 0\. 0\. 0\. 0.
  1\. 0\. 0\. 0\. 1\. 0\. 0\. 1\. 0\. 1\. 0\. 0\. 0\. 0\. 0\. 1\. 0\. 1\. 0.]]
```

接下来，让我们像上一节所做的那样，用这种编码来评估这个数据集上的机器学习。

编码适合训练集，然后像以前一样应用于训练集和测试集。

```py
...
# one-hot encode input variables
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(X_train)
X_train = onehot_encoder.transform(X_train)
X_test = onehot_encoder.transform(X_test)
```

将这些联系在一起，完整的示例如下所示。

```py
# evaluate logistic regression on the breast cancer dataset with an one-hot encoding
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
# define the location of the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
# load the dataset
dataset = read_csv(url, header=None)
# retrieve the array of data
data = dataset.values
# separate into input and output columns
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)
# split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# one-hot encode input variables
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(X_train)
X_train = onehot_encoder.transform(X_train)
X_test = onehot_encoder.transform(X_test)
# ordinal encode target variable
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
# define the model
model = LogisticRegression()
# fit on the training set
model.fit(X_train, y_train)
# predict on test set
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))
```

运行该示例以正确的方式准备数据集，然后对转换后的数据进行模型拟合评估。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，该模型实现了大约 70.53%的分类准确率，这比前一节中的序数编码稍差。

```py
Accuracy: 70.53
```

## 常见问题

本节列出了编码类别数据时的一些常见问题和答案。

**问:如果我有数字和类别数据的混合会怎么样？**

或者，如果我有分类和顺序数据的混合呢？

您需要分别准备或编码数据集中的每个变量(列)，然后将所有准备好的变量重新连接到一个数组中，以便拟合或评估模型。

或者，您可以使用[列转换器](https://machinelearningmastery.com/columntransformer-for-numerical-and-categorical-data/)有条件地将不同的数据转换应用于不同的输入变量。

**问:如果我有上百个类别呢？**

或者，如果我连接多个一热编码向量来创建一个多千元素的输入向量呢？

你可以使用一个热门的编码多达成千上万的类别。此外，使用大向量作为输入听起来很吓人，但是模型通常可以处理它。

**问:什么编码技术最好？**

这是不可知的。

使用您选择的模型在数据集上测试每种技术(以及更多)，并发现最适合您的案例的技术。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [深度学习分类变量的 3 种编码方式](https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/)
*   [机器学习中为什么一热编码数据？](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
*   [如何在 Python 中对序列数据进行一次热编码](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)

### 书

*   [机器学习的特征工程](https://amzn.to/3b9tp3s)，2018。
*   [应用预测建模](https://amzn.to/3b2LHTL)，2013。
*   [统计学习与应用导论](https://amzn.to/3dfRtms)R，2014。

### 蜜蜂

*   [硬化。预处理。OneHotEncoder API](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) 。
*   硬化。预处理。标签编码 API 。
*   [硬化。预处理。序编码器 API](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) 。

### 文章

*   [分类变量，维基百科](https://en.wikipedia.org/wiki/Categorical_variable)。
*   [名义类别，维基百科](https://en.wikipedia.org/wiki/Nominal_category)。

## 摘要

在本教程中，您发现了如何对分类机器学习数据使用编码方案。

具体来说，您了解到:

*   当处理机器学习算法的类别数据时，编码是必需的预处理步骤。
*   如何对具有自然排序的分类变量使用序数编码？
*   如何对没有自然排序的分类变量使用一热编码？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。*