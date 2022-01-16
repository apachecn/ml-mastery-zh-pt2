# 如何在 R 中加载机器学习数据

> 原文：<https://machinelearningmastery.com/how-to-load-your-machine-learning-data-into-r/>

最后更新于 2019 年 8 月 22 日

在处理机器学习问题时，您需要能够将数据加载到 R 中。

在这篇短文中，你将发现如何将数据文件加载到 R 中，并开始你的机器学习项目。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

![Load Your Machine Learning Data Into R](img/d79346525e7d9cbda5fd92447f46ea61.png)

保罗·米勒将你的机器学习数据载入 R
图片，版权所有。

## 访问您的数据

机器学习中处理数据最常见的方式是在数据文件中。

数据最初可能以各种格式和不同位置存储。例如:

*   关系数据库表
*   XML 文件
*   JSON 文件
*   固定宽度格式化文件
*   电子表格文件(如微软办公软件)

您需要将数据合并到一个包含行和列的文件中，然后才能在机器学习项目中使用它。表示机器学习数据集的标准格式是 CSV 文件。这是因为机器学习算法在很大程度上处理表格格式的数据(例如矩阵或输入和输出向量)。

R 中的数据集通常表示为矩阵或数据帧结构。

R 中机器学习项目的第一步是将您的数据作为矩阵或数据框加载到 R 中。

## 载入 CSV 数据文件

本节提供了一些方法，您可以将其复制到自己的机器学习项目中，并将其用于将数据加载到 r。

### 从 CSV 文件加载数据

此示例显示了从 CSV 文件加载 iris 数据集。该方法将把当前目录中没有标题(例如列名)的 CSV 文件作为数据帧加载到 R 中。

```py
# define the filename
filename <- "iris.csv"
# load the CSV file from the local directory
dataset <- read.csv(filename, header=FALSE)
# preview the first 5 rows
head(dataset)
```

运行此配方，您将看到:

```py
   V1  V2  V3  V4          V5
1 5.1 3.5 1.4 0.2 Iris-setosa
2 4.9 3.0 1.4 0.2 Iris-setosa
3 4.7 3.2 1.3 0.2 Iris-setosa
4 4.6 3.1 1.5 0.2 Iris-setosa
5 5.0 3.6 1.4 0.2 Iris-setosa
6 5.4 3.9 1.7 0.4 Iris-setosa
```

如果您想用您的 R 脚本在本地存储数据，例如在修订控制下管理的项目中，这个方法很有用。

如果数据不在本地目录中，您可以:

1.  指定本地环境中数据集的完整路径。
2.  使用 *setwd()* 功能将当前工作目录设置为数据集所在的位置

### 从 CSV 网址加载数据

此示例显示了从位于 [UCI 机器学习存储库](https://machinelearningmastery.com/practice-machine-learning-with-small-in-memory-datasets-from-the-uci-machine-learning-repository/)上的 CSV 文件加载 iris 数据。这个方法将把一个没有标题的 CSV 文件从一个网址加载到 R 中作为数据帧。

```py
# load the library
library(RCurl)
# specify the URL for the Iris data CSV
urlfile <-'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# download the file
downloaded <- getURL(urlfile, ssl.verifypeer=FALSE)
# treat the text data as a steam so we can read from it
connection <- textConnection(downloaded)
# parse the downloaded data as CSV
dataset <- read.csv(connection, header=FALSE)
# preview the first 5 rows
head(dataset)
```

运行此配方，您将看到:

```py
   V1  V2  V3  V4          V5
1 5.1 3.5 1.4 0.2 Iris-setosa
2 4.9 3.0 1.4 0.2 Iris-setosa
3 4.7 3.2 1.3 0.2 Iris-setosa
4 4.6 3.1 1.5 0.2 Iris-setosa
5 5.0 3.6 1.4 0.2 Iris-setosa
6 5.4 3.9 1.7 0.4 Iris-setosa
```

如果您的数据集存储在服务器上，例如您的 GitHub 帐户上，则此方法非常有用。如果您想使用 UCI 机器学习存储库中的数据集，但不想在本地存储它们，这也很有用。

### 其他格式的数据

您可能以 CSV 以外的格式存储数据。

我建议您在处理 r 中的数据之前，使用标准工具和库将其转换为 CSV 格式。一旦转换，您就可以使用上面的秘籍来处理它。

## 摘要

在这篇短文中，您发现了如何将数据加载到 r 中。

您学习了两种加载数据的方法:

1.  从本地 CSV 文件加载数据。
2.  从服务器上的 CSV 文件加载数据。

## 下一步

你试过这些秘籍吗？

1.  开始你的互动环境。
2.  键入或复制粘贴上面的秘籍，并尝试它们。
3.  使用 R 中的内置帮助来了解有关所用函数的更多信息。

你有问题吗？在评论里问，我会尽力回答。