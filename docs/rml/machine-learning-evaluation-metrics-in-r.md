# R 中的机器学习评估指标

> 原文：<https://machinelearningmastery.com/machine-learning-evaluation-metrics-in-r/>

最后更新于 2019 年 8 月 22 日

你可以用什么标准来评估你的机器学习算法？

在这篇文章中，您将发现如何使用许多标准评估指标来评估您在 R 中的机器学习算法。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

![Machine Learning Evaluation Metrics in R](img/e0a6ada952efac64fa36211978776d45.png)

R
图片中的机器学习评估指标由[罗兰·唐劳](https://www.flickr.com/photos/roland/15031149899/)提供，保留部分权利。

## R 中的模型评估指标

有许多不同的度量可以用来评估你的机器学习算法

当您使用 Caret 评估模型时，使用的默认度量是用于分类问题的**准确性**和用于回归的 **RMSE** 。但是 caret 支持一系列其他流行的评估指标。

在下一节中，您将逐步了解 Caret 提供的每个评估指标。每个示例都提供了一个完整的案例研究，您可以将其复制并粘贴到项目中，以适应您的问题。

请注意，这篇文章假设您已经知道如何解释这些其他指标。如果它们对你来说是新的，不要担心，我提供了一些进一步阅读的链接，你可以从中了解更多。

## 评估机器学习算法的度量

在本节中，您将发现如何使用许多不同的通用评估指标来评估机器学习算法。

具体来说，本节将向您展示如何在 R:

*   准确度和卡帕值
*   RMSE 和 R^2
*   敏感性和特异性
*   日志丢失

### 准确度和卡帕值

这些是用于评估 caret 中二进制和多类类别数据集算法的默认指标。

**准确度**是所有实例中正确分类实例的百分比。它在二进制分类问题上比多类分类问题更有用，因为它可能不太清楚这些类的准确率是如何分解的(例如，您需要使用[混淆矩阵](https://machinelearningmastery.com/confusion-matrix-machine-learning/)进行更深入的研究)。[在此了解更多关于准确率的信息](https://en.wikipedia.org/wiki/Accuracy_and_precision)。

**Kappa** 或 Cohen 的 Kappa 类似于分类准确率，除了它在你的数据集上的随机机会的基线处被归一化。这是一种更有用的度量，用于类别不平衡的问题(例如，类别 0 和 1 的 70-30 分割，通过预测所有实例都属于类别 0，可以达到 70%的准确性)。[点击这里](https://en.wikipedia.org/wiki/Cohen%27s_kappa)了解更多关于卡帕的信息。

在下面的例子中，使用了皮马印第安人糖尿病数据集。对于消极和积极的结果，它有 65%到 35%的分类分解。

```py
# load libraries
library(caret)
library(mlbench)
# load the dataset
data(PimaIndiansDiabetes)
# prepare resampling method
control <- trainControl(method="cv", number=5)
set.seed(7)
fit <- train(diabetes~., data=PimaIndiansDiabetes, method="glm", metric="Accuracy", trControl=control)
# display results
print(fit)
```

运行这个例子，我们可以看到评估的每个机器学习算法的准确度和 Kappa 表。这包括每个指标的平均值(左)和标准偏差(标为标准差)，取自交叉验证折叠和试验的总体。

您可以看到，模型的准确率约为 76%，比 65%的基线准确率高出 11 个百分点，这并不令人印象深刻。另一方面，Kappa 显示大约 46%，这更有趣。

```py
Generalized Linear Model 

768 samples
  8 predictor
  2 classes: 'neg', 'pos' 

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 614, 614, 615, 615, 614 
Resampling results

  Accuracy   Kappa      Accuracy SD  Kappa SD 
  0.7695442  0.4656824  0.02692468   0.0616666
```

### RMSE 和 R^2

这些是用于评估 caret 中回归数据集算法的默认度量。

**RMSE** 或均方根误差是预测值与观测值的平均偏差。以输出变量为单位，大致了解一个算法做得好不好是很有用的。[在这里了解更多 RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation)。

**R^2** 表示为 r 的平方，也称为决定系数，为观测值的预测提供了“拟合优度”的度量。这是一个介于 0 和 1 之间的值，分别表示非拟合和完美拟合。[在这里了解更多关于 R^2 的信息](https://en.wikipedia.org/wiki/Coefficient_of_determination)。

在本例中，使用了长期经济数据集。输出变量是一个“使用的数字”。不清楚这是实际计数(例如，以百万计)还是百分比。

```py
# load libraries
library(caret)
# load data
data(longley)
# prepare resampling method
control <- trainControl(method="cv", number=5)
set.seed(7)
fit <- train(Employed~., data=longley, method="lm", metric="RMSE", trControl=control)
# display results
print(fit)
```

运行这个例子，我们可以看到每个被评估的机器学习算法的 RMSE 和 R 平方的表格。同样，您可以看到提供了两个指标的平均值和标准偏差。

你可以看到 RMSE 是 0.38 的就业单位(无论这些单位是什么)。然而，R 平方值显示了非常适合值非常接近 1 (0.988)的数据。

```py
Linear Regression 

16 samples
 6 predictor

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 12, 12, 14, 13, 13 
Resampling results

  RMSE       Rsquared   RMSE SD    Rsquared SD
  0.3868618  0.9883114  0.1025042  0.01581824
```

### ROC 曲线下面积

ROC 度量只适用于二进制分类问题(例如两类)。

要计算 ROC 信息，您必须将列车控制中的汇总功能更改为两类汇总。这将计算曲线下面积(AUROC)，也称为曲线下面积(AUC)，灵敏度和特异性。

**ROC** 实际上是 ROC 曲线或 AUC 下的面积。AUC 代表了一个模型区分正类和负类的能力。1.0 的区域代表了一个模型，它完美地做出了所有预测。0.5 的面积代表一个像随机一样好的模型。[在这里了解更多 ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)。

ROC 可分为敏感性和特异性。二分类问题实际上是敏感性和特异性之间的权衡。

**灵敏度**是真阳性率也叫召回率。实际上正确预测的是正(第一)类的实例数。

**特异性**也叫真阴性率。是负类(第二类)中实际预测正确的实例数。[在这里了解更多关于敏感性和特异性的信息](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)。

```py
# load libraries
library(caret)
library(mlbench)
# load the dataset
data(PimaIndiansDiabetes)
# prepare resampling method
control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=twoClassSummary)
set.seed(7)
fit <- train(diabetes~., data=PimaIndiansDiabetes, method="glm", metric="ROC", trControl=control)
# display results
print(fit)
```

在这里，你可以看到 0.833 的“好”但不“优秀”的 AUC 评分。第一级被视为阳性，在这种情况下为“阴性”(没有糖尿病发作)。

```py
Generalized Linear Model 

768 samples
  8 predictor
  2 classes: 'neg', 'pos' 

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 614, 614, 615, 615, 614 
Resampling results

  ROC        Sens   Spec       ROC SD      Sens SD     Spec SD  
  0.8336003  0.882  0.5600978  0.02111279  0.03563706  0.0560184
```

### 对数损失

对数损失或对数损失用于评估二进制分类，但它更常见于多类分类算法。具体来说，它评估计法估计的概率。[在这里了解更多关于日志丢失的信息](https://en.wikipedia.org/wiki/Loss_functions_for_classification)。

在这种情况下，我们看到为鸢尾花多类分类问题计算的对数损失。

```py
# load libraries
library(caret)
# load the dataset
data(iris)
# prepare resampling method
control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=mnLogLoss)
set.seed(7)
fit <- train(Species~., data=iris, method="rpart", metric="logLoss", trControl=control)
# display results
print(fit)
```

Logloss 被最小化，我们可以看到最优 CART 模型的 cp 为 0。

```py
CART 

150 samples
  4 predictor
  3 classes: 'setosa', 'versicolor', 'virginica' 

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 120, 120, 120, 120, 120 
Resampling results across tuning parameters:

  cp    logLoss    logLoss SD
  0.00  0.4105613  0.6491893 
  0.44  0.6840517  0.4963032 
  0.50  1.0986123  0.0000000 

logLoss was used to select the optimal model using  the smallest value.
The final value used for the model was cp = 0.
```

## 摘要

在这篇文章中，你发现了不同的度量标准，可以用来评估你的机器学习算法在 R 中使用 caret 的表现。具体来说:

*   准确度和卡帕值
*   RMSE 和 R^2
*   敏感性和特异性
*   日志丢失

你可以使用这篇文章中的秘籍来评估你当前或下一个机器学习项目中的机器学习算法。

## 下一步

完成这篇文章中的例子。

1.  打开你的 R 交互环境。
2.  复制粘贴类型上面的示例代码。
3.  慢慢来，了解发生了什么，使用帮助阅读函数。

你有什么问题吗？留言评论，我会尽力的。