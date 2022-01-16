# 使用描述性统计更好地理解你的 R 数据

> 原文：<https://machinelearningmastery.com/descriptive-statistics-examples-with-r/>

最后更新于 2019 年 8 月 22 日

你必须熟悉你的数据。

你建立的任何机器学习模型都只有你提供的数据好。了解数据的第一步是实际查看一些原始值并计算一些基本统计数据。

在这篇文章中，您将发现如何通过 r 中的描述性统计示例和方法快速处理数据集。

如果你是一个刚刚开始使用 R 进行机器学习的开发人员，这些秘籍非常适合你。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

*   **2016 年 11 月更新**:作为一个有用的更新，本教程假设您已经安装了 **mlbench** 和 **e1071** R 软件包。可以通过键入以下命令来安装它们:install . packages(“e 1071”、“mlbench”)

![Descriptive Statistics Examples](img/e322f1e71051f92a023fda9648b6e175.png)

使用描述性统计
了解你在 R 中的数据。

## 您必须了解您的数据

理解你所拥有的数据是至关重要的。

您可以对您的数据运行技术和算法，但是直到您花时间真正理解您的数据集，您才能完全理解您所获得的结果的上下文。

### 更好的理解等于更好的结果

对数据的深入了解会给你带来更好的结果。

花时间研究你所拥有的数据会在一些不太明显的方面帮助你。您为数据以及单个记录或观察所代表的实体建立了直觉。这些可能会让你偏向于特定的技术(不管是好是坏)，但你也可以受到启发。

例如，详细检查您的数据可能会引发对特定技术进行调查的想法:

*   **数据清理**。您可能会发现丢失或损坏的数据，并考虑执行各种数据清理操作，例如标记或删除坏数据以及输入丢失的数据。
*   **数据转换**。您可能会发现一些属性具有熟悉的分布，例如高斯或指数分布，这为您提供了缩放或对数或其他可以应用的变换的概念。
*   **数据建模**。您可能会注意到数据的属性，例如建议使用(或不使用)特定机器学习算法的分布或数据类型。

### 使用描述性统计

你需要看看你的数据。你需要从不同的角度来看待你的数据。

检查你的数据将帮助你建立你的直觉，并促使你开始询问关于你所拥有的数据的问题。

多角度将挑战你从不同的角度思考数据，帮助你提出更多更好的问题。

查看数据的两种方法是:

1.  描述统计学
2.  数据可视化

首先也是最好的开始是计算数据的基本汇总描述性统计数据。

你需要学习你所拥有的数据的形状、大小、类型和总体布局。

让我们看看一些使用 r 总结数据的方法。

## 用描述性统计汇总 R 中的数据

在本节中，您将发现总结数据集的 8 种快速简单的方法。

每种方法都有简要描述，并在 R 中包含一个秘籍，您可以自己运行或复制并适应自己的需求。

### 1.查看您的数据

首先要做的就是查看数据集的一些原始数据。

如果数据集很小，您可能可以在屏幕上显示所有内容。通常不是，所以你可以拿一个小样本来回顾一下。

```py
# load the library
library(mlbench)
# load the dataset
data(PimaIndiansDiabetes)
# display first 20 rows of data
head(PimaIndiansDiabetes, n=20)
```

head 函数将显示前 20 行数据，供您查看和思考。

```py
   pregnant glucose pressure triceps insulin mass pedigree age diabetes
1         6     148       72      35       0 33.6    0.627  50      pos
2         1      85       66      29       0 26.6    0.351  31      neg
3         8     183       64       0       0 23.3    0.672  32      pos
4         1      89       66      23      94 28.1    0.167  21      neg
5         0     137       40      35     168 43.1    2.288  33      pos
6         5     116       74       0       0 25.6    0.201  30      neg
7         3      78       50      32      88 31.0    0.248  26      pos
8        10     115        0       0       0 35.3    0.134  29      neg
9         2     197       70      45     543 30.5    0.158  53      pos
10        8     125       96       0       0  0.0    0.232  54      pos
11        4     110       92       0       0 37.6    0.191  30      neg
12       10     168       74       0       0 38.0    0.537  34      pos
13       10     139       80       0       0 27.1    1.441  57      neg
14        1     189       60      23     846 30.1    0.398  59      pos
15        5     166       72      19     175 25.8    0.587  51      pos
16        7     100        0       0       0 30.0    0.484  32      pos
17        0     118       84      47     230 45.8    0.551  31      pos
18        7     107       74       0       0 29.6    0.254  31      pos
19        1     103       30      38      83 43.3    0.183  33      neg
20        1     115       70      30      96 34.6    0.529  32      pos
```

### 2.数据的维度

你有多少数据？你可能有一个大概的想法，但是有一个精确的数字要好得多。

如果您有很多实例，您可能需要使用较小的数据样本，以便模型训练和评估在计算上易于处理。如果你有大量的属性，你可能需要选择那些最相关的。如果属性比实例多，您可能需要选择特定的建模技术。

```py
# load the libraries
library(mlbench)
# load the dataset
data(PimaIndiansDiabetes)
# display the dimensions of the dataset
dim(PimaIndiansDiabetes)
```

这将显示加载数据集的行和列。

```py
[1] 768   9
```

### 3.数据类型

您需要知道数据中属性的类型。

这是无价的。这些类型将表明进一步分析的类型、可视化的类型，甚至是你可以使用的机器学习算法的类型。

此外，也许一些属性被加载为一种类型(例如整数)，并且实际上可以被表示为另一种类型(分类因子)。检查类型有助于尽早暴露这些问题并激发想法。

```py
# load library
library(mlbench)
# load dataset
data(BostonHousing)
# list types for each attribute
sapply(BostonHousing, class)
```

这将列出数据集中每个属性的数据类型。

```py
     crim        zn     indus      chas       nox        rm       age       dis       rad       tax   ptratio         b 
"numeric" "numeric" "numeric"  "factor" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" "numeric" 
    lstat      medv 
"numeric" "numeric"
```

### 4.类别分布

在分类问题中，您必须知道属于每个类值的实例的比例。

这一点很重要，因为它可能会突出数据中的不平衡，如果严重，可能需要通过重新平衡技术来解决。在多类分类问题的情况下，它可能会暴露具有少量或零个实例的类，这些实例可能是要从数据集中移除的候选实例。

```py
# load the libraries
library(mlbench)
# load the dataset
data(PimaIndiansDiabetes)
# distribution of class variable
y <- PimaIndiansDiabetes$diabetes
cbind(freq=table(y), percentage=prop.table(table(y))*100)
```

这个方法创建了一个有用的表格，显示了属于每个类的实例数量以及它在整个数据集中所占的百分比。

```py
    freq percentage
neg  500   65.10417
pos  268   34.89583
```

### 5.数据汇总

有一个最有价值的函数叫做 summary()，它依次汇总数据集中的每个属性。这是最有价值的功能。

该函数为每个属性创建一个表，并列出值的细分。因子被描述为每个类别标签旁边的计数。数字属性描述如下:

*   福建话
*   第 25 百分位
*   中位数
*   均值
*   第 75 百分位
*   最大

细分还包括属性缺失值数量的指示(标记为不适用)。

```py
# load the iris dataset
data(iris)
# summarize the dataset
summary(iris)
```

你可以看到这个秘籍产生了很多信息供你回顾。慢慢来，依次研究每个属性。

```py
  Sepal.Length    Sepal.Width     Petal.Length    Petal.Width          Species  
 Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100   setosa    :50  
 1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300   versicolor:50  
 Median :5.800   Median :3.000   Median :4.350   Median :1.300   virginica :50  
 Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199                  
 3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800                  
 Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500
```

### 6.标准偏差

上面的 summary()函数缺少的一点是标准差。

标准差和平均值有助于了解数据是否具有高斯(或近似高斯)分布。例如，它对于快速和肮脏的异常值去除工具非常有用，其中任何超过平均值标准偏差三倍的值都在数据的 99.7%之外。

```py
# load the libraries
library(mlbench)
# load the dataset
data(PimaIndiansDiabetes)
# calculate standard deviation for all attributes
sapply(PimaIndiansDiabetes[,1:8], sd)
```

这会计算数据集中每个数值属性的标准偏差。

```py
   pregnant     glucose    pressure     triceps     insulin        mass    pedigree         age 
  3.3695781  31.9726182  19.3558072  15.9522176 115.2440024   7.8841603   0.3313286  11.7602315
```

### 7.歪斜

如果一个分布看起来有点像高斯分布，但是被向左或向右推得很远，那么知道偏斜是有用的。

通过数据图，如直方图或密度图，可以更容易地感觉到倾斜。从平均值、标准差和四分位数来看，很难分辨。

然而，预先计算偏斜给了您一个参考，如果您决定纠正属性的偏斜，您可以在以后使用它。

```py
# load libraries
library(mlbench)
library(e1071)
# load the dataset
data(PimaIndiansDiabetes)
# calculate skewness for each variable
skew <- apply(PimaIndiansDiabetes[,1:8], 2, skewness)
# display skewness, larger/smaller deviations from 0 show more skew
print(skew)
```

偏斜值的分布离零越远，向左(负偏斜值)或向右(正偏斜值)的偏斜就越大。

```py
  pregnant    glucose   pressure    triceps    insulin       mass   pedigree        age 
 0.8981549  0.1730754 -1.8364126  0.1089456  2.2633826 -0.4273073  1.9124179  1.1251880
```

### 8.相关

观察和思考属性之间的相互关系非常重要。

对于数字属性，考虑属性间交互的一个好方法是计算每对属性的相关性。

```py
# load the libraries
library(mlbench)
# load the dataset
data(PimaIndiansDiabetes)
# calculate a correlation matrix for numeric variables
correlations <- cor(PimaIndiansDiabetes[,1:8])
# display the correlation matrix
print(correlations)
```

这为数值数据创建了所有属性相关性对的对称表。偏离零表示更多的正相关或负相关。高于 0.75 或低于-0.75 的值可能更有趣，因为它们显示出很高的相关性。值 1 和-1 表示完全正相关或负相关。

```py
            pregnant    glucose   pressure     triceps     insulin       mass    pedigree         age
pregnant  1.00000000 0.12945867 0.14128198 -0.08167177 -0.07353461 0.01768309 -0.03352267  0.54434123
glucose   0.12945867 1.00000000 0.15258959  0.05732789  0.33135711 0.22107107  0.13733730  0.26351432
pressure  0.14128198 0.15258959 1.00000000  0.20737054  0.08893338 0.28180529  0.04126495  0.23952795
triceps  -0.08167177 0.05732789 0.20737054  1.00000000  0.43678257 0.39257320  0.18392757 -0.11397026
insulin  -0.07353461 0.33135711 0.08893338  0.43678257  1.00000000 0.19785906  0.18507093 -0.04216295
mass      0.01768309 0.22107107 0.28180529  0.39257320  0.19785906 1.00000000  0.14064695  0.03624187
pedigree -0.03352267 0.13733730 0.04126495  0.18392757  0.18507093 0.14064695  1.00000000  0.03356131
age       0.54434123 0.26351432 0.23952795 -0.11397026 -0.04216295 0.03624187  0.03356131  1.00000000
```

### 更多秘籍

这个数据汇总方法的列表并不完整，但是它们足以让您快速对数据集有一个初步的了解。

除了上面的秘籍列表之外，您可以研究的一些数据汇总是查看数据子集的统计数据。考虑查看 r 中的*聚合()*函数

您是否使用了未列出的数据汇总方法？在下面留言，我很想听听。

## 要记住的提示

本节为您提供了使用汇总统计数据查看数据时需要记住的一些提示。

*   **查看数字**。生成汇总统计信息是不够的。花点时间停下来，阅读并认真思考你看到的数字。
*   **问为什么**。回顾你的数字，问很多问题。你是如何以及为什么看到具体数字的？想想这些数字是如何与一般的问题领域和观察相关的具体实体联系起来的。
*   **写下想法**。写下你的观察和想法。保存一个小的文本文件或记事本，记下变量之间的关系，数字的含义，以及以后尝试的技巧。当你试图想出新的尝试时，你现在在数据新鲜时写下的东西会很有价值。

## 你可以用 R 总结你的数据

**不需要做 R 程序员**。R 中的数据汇总非常简单，上面的秘籍可以证明。如果你刚刚开始，你可以复制并粘贴上面的秘籍，并使用 R 中的内置帮助开始学习它们是如何工作的(例如:*？功能名称*)。

**不需要擅长统计**。这篇文章中使用的统计数据非常简单，但是您可能已经忘记了一些基本信息。你可以快速浏览维基百科的均值、标准差和四分位数等主题来刷新你的知识。

下面是一个简短的列表:

*   [表示](https://en.wikipedia.org/wiki/Mean)
*   [标准偏差](https://en.wikipedia.org/wiki/Standard_deviation)
*   [四分位数](https://en.wikipedia.org/wiki/Quartile)
*   [皮尔逊相关系数](https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient)
*   [偏斜度](https://en.wikipedia.org/wiki/Skewness)
*   [68–95–99.7 规则](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)

相关帖子见:【机器学习统计】中[速成班](https://machinelearningmastery.com/crash-course-statistics-machine-learning/)。

**不需要自己的数据集**。上面的每个示例都使用内置数据集或 R 包提供的数据集。*数据集* R 包中有很多有趣的数据集，你可以研究和玩。有关更多信息，请参见数据集 R 包的[文档。](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/00Index.html)

## 摘要

在这篇文章中，您发现了在开始机器学习项目之前描述数据集的重要性。

您发现了使用 R 总结数据集的 8 种不同方法:

1.  查看您的数据
2.  数据的维度
3.  数据类型
4.  类别分布
5.  数据汇总
6.  标准偏差
7.  歪斜
8.  相关

您现在也有了可以复制并粘贴到项目中的秘籍。

## 行动步骤

你是想用 R 提高技能还是在 R 练习机器学习？

完成上面的每个例子。

1.  打开 R 交互环境。
2.  键入或复制粘贴每个配方，并了解其工作原理。
3.  潜入更深的地方使用？FunctionName 了解有关使用的特定函数的更多信息。

回电并留言，我很想听听你的进展。

你有问题吗？留言问一问。