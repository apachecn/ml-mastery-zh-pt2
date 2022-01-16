# 通过预处理为机器学习准备好数据

> 原文：<https://machinelearningmastery.com/pre-process-your-dataset-in-r/>

最后更新于 2019 年 8 月 22 日

为了从机器学习算法中获得最佳结果，需要准备数据。

在这篇文章中，您将发现如何转换您的数据，以便最好地使用 caret 包将其结构暴露给 R 中的机器学习算法。

你将通过 8 个流行和强大的数据转换与秘籍，你可以研究或复制和粘贴到你当前或下一个机器学习项目。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

![Pre-Process Your Machine Learning Dataset in R](img/7bf47c0e260d3ba630936c81fd559a25.png)

预处理你的机器学习数据集在 R
照片由[弗雷泽凯恩斯](https://www.flickr.com/photos/catsncarp/2248660994/)，一些权利保留。

## 数据预处理的需求

您希望从数据集上的机器学习算法中获得最佳准确率。

一些机器学习算法要求数据具有特定的形式。而如果数据是以特定的方式准备的，其他算法可以表现得更好，但并不总是如此。最后，您的原始数据可能不是最好的格式，无法最好地向预测变量展示底层结构和关系。

重要的是准备好你的数据，让它给各种不同的机器学习算法最好的机会来解决你的问题。

作为机器学习项目的一部分，您需要预处理原始数据。

## 数据预处理方法

很难知道使用哪些数据预处理方法。

您可以使用经验法则，例如:

*   如果输入属性具有相同的比例，基于实例的方法更有效。
*   回归方法可以更好地工作的输入属性是标准化的。

这些都是试探法，但并不是机器学习的硬性规律，因为有时候如果你忽略它们，你可以得到更好的结果。

你应该尝试一系列不同机器学习算法的数据转换。这将帮助您发现数据的良好表示和更好地利用这些表示公开的结构的算法。

一个好主意是单独抽查多个转换以及转换的组合。

在下一节中，您将发现如何应用数据转换，以便使用 Caret 包在 R 中准备数据。

## R 语言中 Caret 的数据预处理

R 中的 Caret 包提供了许多有用的数据转换。

这些转换可以通过两种方式使用。

*   **独立**:变换可以从训练数据建模并应用于多个数据集。使用*预处理()*函数准备变换模型，并使用*预测()*函数将其应用于数据集。
*   **训练**:模型评估时可以自动准备和应用变换。在训练期间应用的变换使用*预处理()*准备，并通过预处理参数传递给*训练()*函数。

本节介绍了一些数据预处理示例。它们是使用独立的方法呈现的，但是您可以在模型训练过程中轻松地使用准备好的预处理模型。

本节中的所有预处理示例都是针对数值数据的。请注意，预处理函数将跳过非数字数据而不会产生错误。

您可以通过键入以下内容来阅读 preference 函数的帮助，从而了解 caret 包提供的数据转换的更多信息？通过阅读[Caret 预处理页面](https://topepo.github.io/caret/preprocess.html)，进行预处理。

所呈现的数据转换更可能对算法有用，例如回归算法、基于实例的方法(如 kNN 和 LVQ)、支持向量机和神经网络。它们不太可能对基于树和规则的方法有用。

### 变换方法综述

以下是 caret 中*prepare()*函数的方法参数支持的所有转换方法的快速摘要。

*   " *BoxCox* ":应用 Box–Cox 变换，值必须非零且为正。
*   " *YeoJohnson* ":应用一个 Yeo-Johnson 变换，像一个 BoxCox，但是值可以是负数。
*   ”*export trans*”:应用像 BoxCox 和 YeoJohnson 这样的电源变换。
*   “ *zv* ”:移除方差为零的属性(均为相同值)。
*   " *nzv* ":移除方差接近于零(接近于相同值)的属性。
*   “*中心*”:从数值中减去平均值。
*   "*刻度*":用标准差除数值。
*   “*范围*”:归一化值。
*   “ *pca* ”:将数据转换为主成分。
*   “ *ica* ”:将数据转换为独立的分量。
*   “*空间符号*”:将数据投影到单位圆上。

以下几节将演示一些更流行的方法。

### 1.规模

比例变换计算属性的标准偏差，并将每个值除以该标准偏差。

```py
# load libraries
library(caret)
# load the dataset
data(iris)
# summarize data
summary(iris[,1:4])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(iris[,1:4], method=c("scale"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, iris[,1:4])
# summarize the transformed dataset
summary(transformed)
```

运行配方，您将看到:

```py
  Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
 Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
 1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
 Median :5.800   Median :3.000   Median :4.350   Median :1.300  
 Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
 3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
 Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  

Created from 150 samples and 4 variables

Pre-processing:
  - ignored (0)
  - scaled (4)

  Sepal.Length    Sepal.Width      Petal.Length     Petal.Width    
 Min.   :5.193   Min.   : 4.589   Min.   :0.5665   Min.   :0.1312  
 1st Qu.:6.159   1st Qu.: 6.424   1st Qu.:0.9064   1st Qu.:0.3936  
 Median :7.004   Median : 6.883   Median :2.4642   Median :1.7055  
 Mean   :7.057   Mean   : 7.014   Mean   :2.1288   Mean   :1.5734  
 3rd Qu.:7.729   3rd Qu.: 7.571   3rd Qu.:2.8890   3rd Qu.:2.3615  
 Max.   :9.540   Max.   :10.095   Max.   :3.9087   Max.   :3.2798
```

### 2.中心

中心变换计算属性的平均值，并从每个值中减去它。

```py
# load libraries
library(caret)
# load the dataset
data(iris)
# summarize data
summary(iris[,1:4])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(iris[,1:4], method=c("center"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, iris[,1:4])
# summarize the transformed dataset
summary(transformed)
```

运行配方，您将看到:

```py
  Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
 Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
 1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
 Median :5.800   Median :3.000   Median :4.350   Median :1.300  
 Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
 3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
 Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  

Created from 150 samples and 4 variables

Pre-processing:
  - centered (4)
  - ignored (0)

 Sepal.Length       Sepal.Width        Petal.Length     Petal.Width     
 Min.   :-1.54333   Min.   :-1.05733   Min.   :-2.758   Min.   :-1.0993  
 1st Qu.:-0.74333   1st Qu.:-0.25733   1st Qu.:-2.158   1st Qu.:-0.8993  
 Median :-0.04333   Median :-0.05733   Median : 0.592   Median : 0.1007  
 Mean   : 0.00000   Mean   : 0.00000   Mean   : 0.000   Mean   : 0.0000  
 3rd Qu.: 0.55667   3rd Qu.: 0.24267   3rd Qu.: 1.342   3rd Qu.: 0.6007  
 Max.   : 2.05667   Max.   : 1.34267   Max.   : 3.142   Max.   : 1.3007
```

### 3.使标准化

将规模和中心转换结合起来将使您的数据标准化。属性的平均值为 0，标准偏差为 1。

```py
# load libraries
library(caret)
# load the dataset
data(iris)
# summarize data
summary(iris[,1:4])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(iris[,1:4], method=c("center", "scale"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, iris[,1:4])
# summarize the transformed dataset
summary(transformed)
```

请注意，当在 caret 中定义 PRofile 过程时，我们如何在列表中列出多个方法。运行配方，您将看到:

```py
  Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
 Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
 1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
 Median :5.800   Median :3.000   Median :4.350   Median :1.300  
 Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
 3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
 Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  

Created from 150 samples and 4 variables

Pre-processing:
  - centered (4)
  - ignored (0)
  - scaled (4)

 Sepal.Length       Sepal.Width       Petal.Length      Petal.Width     
 Min.   :-1.86378   Min.   :-2.4258   Min.   :-1.5623   Min.   :-1.4422  
 1st Qu.:-0.89767   1st Qu.:-0.5904   1st Qu.:-1.2225   1st Qu.:-1.1799  
 Median :-0.05233   Median :-0.1315   Median : 0.3354   Median : 0.1321  
 Mean   : 0.00000   Mean   : 0.0000   Mean   : 0.0000   Mean   : 0.0000  
 3rd Qu.: 0.67225   3rd Qu.: 0.5567   3rd Qu.: 0.7602   3rd Qu.: 0.7880  
 Max.   : 2.48370   Max.   : 3.0805   Max.   : 1.7799   Max.   : 1.7064
```

### 4.使标准化

数据值可以缩放到[0，1]的范围内，这称为规范化。

```py
# load libraries
library(caret)
# load the dataset
data(iris)
# summarize data
summary(iris[,1:4])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(iris[,1:4], method=c("range"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, iris[,1:4])
# summarize the transformed dataset
summary(transformed)
```

运行配方，您将看到:

```py
  Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
 Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
 1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
 Median :5.800   Median :3.000   Median :4.350   Median :1.300  
 Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
 3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
 Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  

Created from 150 samples and 4 variables

Pre-processing:
  - ignored (0)
  - re-scaling to [0, 1] (4)

  Sepal.Length     Sepal.Width      Petal.Length     Petal.Width     
 Min.   :0.0000   Min.   :0.0000   Min.   :0.0000   Min.   :0.00000  
 1st Qu.:0.2222   1st Qu.:0.3333   1st Qu.:0.1017   1st Qu.:0.08333  
 Median :0.4167   Median :0.4167   Median :0.5678   Median :0.50000  
 Mean   :0.4287   Mean   :0.4406   Mean   :0.4675   Mean   :0.45806  
 3rd Qu.:0.5833   3rd Qu.:0.5417   3rd Qu.:0.6949   3rd Qu.:0.70833  
 Max.   :1.0000   Max.   :1.0000   Max.   :1.0000   Max.   :1.00000
```

### 5.Box-Cox 变换

当一个属性具有类似高斯的分布但被移动时，这被称为偏斜。属性的分布可以移动，以减少偏斜，使其更具高斯性。BoxCox 转换可以执行此操作(假设所有值都是正数)。

```py
# load libraries
library(mlbench)
library(caret)
# load the dataset
data(PimaIndiansDiabetes)
# summarize pedigree and age
summary(PimaIndiansDiabetes[,7:8])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(PimaIndiansDiabetes[,7:8], method=c("BoxCox"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, PimaIndiansDiabetes[,7:8])
# summarize the transformed dataset (note pedigree and age)
summary(transformed)
```

注意，我们只对两个看起来有倾斜的属性应用了转换。运行配方，您将看到:

```py
    pedigree           age       
 Min.   :0.0780   Min.   :21.00  
 1st Qu.:0.2437   1st Qu.:24.00  
 Median :0.3725   Median :29.00  
 Mean   :0.4719   Mean   :33.24  
 3rd Qu.:0.6262   3rd Qu.:41.00  
 Max.   :2.4200   Max.   :81.00  

Created from 768 samples and 2 variables

Pre-processing:
  - Box-Cox transformation (2)
  - ignored (0)

Lambda estimates for Box-Cox transformation:
-0.1, -1.1

    pedigree            age        
 Min.   :-2.5510   Min.   :0.8772  
 1st Qu.:-1.4116   1st Qu.:0.8815  
 Median :-0.9875   Median :0.8867  
 Mean   :-0.9599   Mean   :0.8874  
 3rd Qu.:-0.4680   3rd Qu.:0.8938  
 Max.   : 0.8838   Max.   :0.9019
```

有关该转换的更多信息，请参见 [Box-Cox 转换维基百科](https://en.wikipedia.org/wiki/Power_transform#Box.E2.80.93Cox_transformation)。

### 6.约-约翰逊变换

另一种幂变换，如 Box-Cox 变换，但它支持等于零和负值的原始值。

```py
# load libraries
library(mlbench)
library(caret)
# load the dataset
data(PimaIndiansDiabetes)
# summarize pedigree and age
summary(PimaIndiansDiabetes[,7:8])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(PimaIndiansDiabetes[,7:8], method=c("YeoJohnson"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, PimaIndiansDiabetes[,7:8])
# summarize the transformed dataset (note pedigree and age)
summary(transformed)
```

运行配方，您将看到:

```py
    pedigree           age       
 Min.   :0.0780   Min.   :21.00  
 1st Qu.:0.2437   1st Qu.:24.00  
 Median :0.3725   Median :29.00  
 Mean   :0.4719   Mean   :33.24  
 3rd Qu.:0.6262   3rd Qu.:41.00  
 Max.   :2.4200   Max.   :81.00  

Created from 768 samples and 2 variables

Pre-processing:
  - ignored (0)
  - Yeo-Johnson transformation (2)

Lambda estimates for Yeo-Johnson transformation:
-2.25, -1.15

    pedigree           age        
 Min.   :0.0691   Min.   :0.8450  
 1st Qu.:0.1724   1st Qu.:0.8484  
 Median :0.2265   Median :0.8524  
 Mean   :0.2317   Mean   :0.8530  
 3rd Qu.:0.2956   3rd Qu.:0.8580  
 Max.   :0.4164   Max.   :0.8644
```

### 7.主成分分析

将数据转换成主成分。变换使分量保持在方差阈值以上(默认值=0.95)，或者可以指定分量的数量(pcaComp)。结果是属性是不相关的，对于像线性和广义线性回归这样的算法很有用。

```py
# load the libraries
library(mlbench)
# load the dataset
data(iris)
# summarize dataset
summary(iris)
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(iris, method=c("center", "scale", "pca"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, iris)
# summarize the transformed dataset
summary(transformed)
```

请注意，当我们运行配方时，只选择了两个主要成分。

```py
  Sepal.Length    Sepal.Width     Petal.Length    Petal.Width          Species  
 Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100   setosa    :50  
 1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300   versicolor:50  
 Median :5.800   Median :3.000   Median :4.350   Median :1.300   virginica :50  
 Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199                  
 3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800                  
 Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500                  
Created from 150 samples and 5 variables

Pre-processing:
  - centered (4)
  - ignored (1)
  - principal component signal extraction (4)
  - scaled (4)

PCA needed 2 components to capture 95 percent of the variance

       Species        PC1               PC2          
 setosa    :50   Min.   :-2.7651   Min.   :-2.67732  
 versicolor:50   1st Qu.:-2.0957   1st Qu.:-0.59205  
 virginica :50   Median : 0.4169   Median :-0.01744  
                 Mean   : 0.0000   Mean   : 0.00000  
                 3rd Qu.: 1.3385   3rd Qu.: 0.59649  
                 Max.   : 3.2996   Max.   : 2.64521
```

### 8.独立成分分析

将数据转换为独立的组件。与主成分分析不同，独立成分分析保留了那些独立的成分。您必须使用 *n.comp* 参数指定所需独立组件的数量。对朴素贝叶斯等算法有用。

```py
# load libraries
library(mlbench)
library(caret)
# load the dataset
data(PimaIndiansDiabetes)
# summarize dataset
summary(PimaIndiansDiabetes[,1:8])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(PimaIndiansDiabetes[,1:8], method=c("center", "scale", "ica"), n.comp=5)
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, PimaIndiansDiabetes[,1:8])
# summarize the transformed dataset
summary(transformed)
```

运行配方，您将看到:

```py
    pregnant         glucose         pressure         triceps         insulin           mass          pedigree     
 Min.   : 0.000   Min.   :  0.0   Min.   :  0.00   Min.   : 0.00   Min.   :  0.0   Min.   : 0.00   Min.   :0.0780  
 1st Qu.: 1.000   1st Qu.: 99.0   1st Qu.: 62.00   1st Qu.: 0.00   1st Qu.:  0.0   1st Qu.:27.30   1st Qu.:0.2437  
 Median : 3.000   Median :117.0   Median : 72.00   Median :23.00   Median : 30.5   Median :32.00   Median :0.3725  
 Mean   : 3.845   Mean   :120.9   Mean   : 69.11   Mean   :20.54   Mean   : 79.8   Mean   :31.99   Mean   :0.4719  
 3rd Qu.: 6.000   3rd Qu.:140.2   3rd Qu.: 80.00   3rd Qu.:32.00   3rd Qu.:127.2   3rd Qu.:36.60   3rd Qu.:0.6262  
 Max.   :17.000   Max.   :199.0   Max.   :122.00   Max.   :99.00   Max.   :846.0   Max.   :67.10   Max.   :2.4200  
      age       
 Min.   :21.00  
 1st Qu.:24.00  
 Median :29.00  
 Mean   :33.24  
 3rd Qu.:41.00  
 Max.   :81.00  

Created from 768 samples and 8 variables

Pre-processing:
  - centered (8)
  - independent component signal extraction (8)
  - ignored (0)
  - scaled (8)

ICA used 5 components

      ICA1              ICA2               ICA3              ICA4                ICA5        
 Min.   :-5.7213   Min.   :-4.89818   Min.   :-6.0289   Min.   :-2.573436   Min.   :-1.8815  
 1st Qu.:-0.4873   1st Qu.:-0.48188   1st Qu.:-0.4693   1st Qu.:-0.640601   1st Qu.:-0.8279  
 Median : 0.1813   Median : 0.05071   Median : 0.2987   Median : 0.007582   Median :-0.2416  
 Mean   : 0.0000   Mean   : 0.00000   Mean   : 0.0000   Mean   : 0.000000   Mean   : 0.0000  
 3rd Qu.: 0.6839   3rd Qu.: 0.56462   3rd Qu.: 0.6941   3rd Qu.: 0.638238   3rd Qu.: 0.7048  
 Max.   : 2.1819   Max.   : 4.25611   Max.   : 1.3726   Max.   : 3.761017   Max.   : 2.9622
```

### 数据转换技巧

以下是充分利用数据转换的一些技巧。

*   **实际使用它们**。如果您正在考虑并使用数据转换来准备数据，那么您就领先了一步。这是一个容易忘记或跳过的步骤，通常会对最终模型的准确性产生巨大影响。
*   **使用品种**。使用一套不同的机器学习算法，在您的数据上尝试多种不同的数据转换。
*   **回顾总结**。最好总结一下转换前后的数据，以了解其效果。*总结()*功能非常有用。
*   **可视化数据**。可视化前后数据的分布也是一个好主意，这样可以对变换的效果有一个空间直觉。

## 摘要

在本节中，您发现了 8 种数据预处理方法，可以通过 caret 包对 R 中的数据使用这些方法:

*   数据缩放
*   数据居中
*   数据标准化
*   数据标准化
*   盒子-考克斯变换
*   约-约翰逊变换
*   主成分分析变换
*   独立分量分析变换

你可以用本节介绍的方法进行练习，或者将它们应用到你当前或下一个机器学习项目中。

## 下一步

你试过这些秘籍吗？

1.  开始你的互动环境。
2.  键入或复制粘贴上面的秘籍并试用。
3.  使用 R 中的内置帮助来了解有关所用函数的更多信息。

你有问题吗？在评论里问，我会尽力回答。