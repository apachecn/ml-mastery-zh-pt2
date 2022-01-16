# R 中的非线性分类

> 原文：<https://machinelearningmastery.com/non-linear-classification-in-r/>

最后更新于 2019 年 8 月 22 日

在这篇文章中，你将发现 r 中的 8 种非线性分类方法。每种方法都可以为你自己的问题复制、粘贴和修改。

本文所有秘籍均使用[数据集包](http://stat.ethz.ch/R-manual/R-patched/library/datasets/html/00Index.html)中提供的[鸢尾花数据集](http://stat.ethz.ch/R-manual/R-patched/library/datasets/html/iris.html)。该数据集描述了鸢尾花的测量结果，并要求对三种花卉中的一种进行分类。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

[![Irises](img/5ca6400308a8cda8b12f673236f0930d.png)](https://machinelearningmastery.com/wp-content/uploads/2014/03/irises.jpg)

鸢尾花
dottieg 2007 摄影，版权所有

## 混合判别分析

这个方法演示了虹膜数据集上的 MDA 方法。

```py
# load the package
library(mda)
data(iris)
# fit model
fit <- mda(Species~., data=iris)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4])
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于 [mda 包](https://cran.r-project.org/web/packages/mda/index.html)中 **mda** 功能的信息。

## 二次判别分析

QDA 寻求属性之间的二次关系，使类之间的距离最大化。

这个方法演示了虹膜数据集上的 QDA 方法。

```py
# load the package
library(MASS)
data(iris)
# fit model
fit <- qda(Species~., data=iris)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4])$class
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于[质量包](https://cran.r-project.org/web/packages/MASS/index.html)中 **qda** 功能的信息。

## 正则化判别分析

这个配方演示了虹膜数据集上的 RDA 方法。

```py
# load the package
library(klaR)
data(iris)
# fit model
fit <- rda(Species~., data=iris, gamma=0.05, lambda=0.01)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4])$class
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于 [klaR 包](https://cran.r-project.org/web/packages/klaR/index.html)中 **rda** 功能的信息。

## 神经网络

神经网络(NN)是一个计算单元的图形，这些计算单元接收输入并将结果转换成输出并传递出去。这些单元被排序成层，以将输入向量的特征连接到输出向量的特征。通过训练，例如反向传播算法，可以设计和训练神经网络来模拟数据中的潜在关系。

这个秘籍展示了虹膜数据集上的神经网络。

```py
# load the package
library(nnet)
data(iris)
# fit model
fit <- nnet(Species~., data=iris, size=4, decay=0.0001, maxit=500)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4], type="class")
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于[网络包](https://cran.r-project.org/web/packages/nnet/index.html)中**网络**功能的信息。

## 灵活判别分析

这个配方展示了美国食品和药物管理局在虹膜数据集上的方法。

```py
# load the package
library(mda)
data(iris)
# fit model
fit <- fda(Species~., data=iris)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4])
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于 [mda 包](https://cran.r-project.org/web/packages/mda/index.html)中 **fda** 功能的信息。

## 支持向量机

支持向量机(SVM)是一种方法，它使用转换后的问题空间中的点来最好地将类分成两组。一对全方法支持多类分类。SVM 还通过用最小的容许误差量对函数建模来支持回归。

这个方法演示了虹膜数据集上的 SVM 方法。

```py
# load the package
library(kernlab)
data(iris)
# fit model
fit <- ksvm(Species~., data=iris)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4], type="response")
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于 [kernlab 软件包](https://cran.r-project.org/web/packages/kernlab/index.html)中 **ksvm** 功能的信息。

## k-最近邻

k-近邻(kNN)方法通过定位给定数据实例的相似案例(使用相似性函数)并返回最相似数据实例的平均值或大部分来进行预测。

这个方法演示了虹膜数据集上的 kNN 方法。

```py
# load the package
library(caret)
data(iris)
# fit model
fit <- knn3(Species~., data=iris, k=5)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4], type="class")
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于[caret 包](https://cran.r-project.org/web/packages/caret/index.html)中 **knn3** 功能的信息。

## 朴素贝叶斯

朴素贝叶斯使用贝叶斯定理来建模每个属性与类变量的条件关系。

这个方法演示了虹膜数据集上的朴素贝叶斯。

```py
# load the package
library(e1071)
data(iris)
# fit model
fit <- naiveBayes(Species~., data=iris)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4])
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于 [e1071 集成](https://cran.r-project.org/web/packages/e1071/index.html)中**天真贝氏**功能的信息。

## 摘要

在这篇文章中，你发现了 8 个使用鸢尾花数据集进行非线性分类的方法。

每个配方都是通用的，可以根据自己的问题进行复制、粘贴和修改。