# R 中的线性分类

> 原文：<https://machinelearningmastery.com/linear-classification-in-r/>

最后更新于 2019 年 8 月 22 日

在这篇文章中，你将发现 3 种线性分类算法的配方

本文所有秘籍均使用[数据集包](http://stat.ethz.ch/R-manual/R-patched/library/datasets/html/00Index.html)中提供的[鸢尾花数据集](http://stat.ethz.ch/R-manual/R-patched/library/datasets/html/iris.html)。该数据集描述了鸢尾花的测量结果，并要求对三种花卉中的一种进行分类。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

[![binary classification](img/52a79b22ba070fca260bb8e599a45c53.png)](https://machinelearningmastery.com/wp-content/uploads/2014/07/binary-classification.jpg)

罗伯特·库塞-贝克拍摄的红色 vs 蓝色
照片，保留部分权利

## 逻辑回归

逻辑回归是一种分类方法，它对属于两类之一的观测值的概率进行建模。因此，通常用二分类问题(2 类)来证明逻辑回归。逻辑回归也可以用于具有两个以上类别(多项式)的问题，如本例所示。

该方法演示了虹膜数据集上的多项式逻辑回归方法。

```py
# load the package
library(VGAM)
# load data
data(iris)
# fit model
fit <- vglm(Species~., family=multinomial, data=iris)
# summarize the fit
summary(fit)
# make predictions
probabilities <- predict(fit, iris[,1:4], type="response")
predictions <- apply(probabilities, 1, which.max)
predictions[which(predictions=="1")] <- levels(iris$Species)[1]
predictions[which(predictions=="2")] <- levels(iris$Species)[2]
predictions[which(predictions=="3")] <- levels(iris$Species)[3]
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于 [VGAM 套餐](https://cran.r-project.org/web/packages/VGAM/index.html)中 **vglm** 功能的信息。

## 线性判别分析

LDA 是一种分类方法，它可以找到数据属性的线性组合，从而最好地将数据分成类。

这个秘籍演示了虹膜数据集上的线性判别分析方法。

```py
# load the package
library(MASS)
data(iris)
# fit model
fit <- lda(Species~., data=iris)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4])$class
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于[质量包](https://cran.r-project.org/web/packages/MASS/index.html)的 **lda** 功能。

## 偏最小二乘判别分析

偏最小二乘判别分析是线性判别分析在输入数据降维投影(偏最小二乘)上的应用。

这个配方演示了虹膜数据集上的 PLSDA 方法。

```py
# load the package
library(caret)
data(iris)
x <- iris[,1:4]
y <- iris[,5]
# fit model
fit <- plsda(x, y, probMethod="Bayes")
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4])
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于[caret 包](https://cran.r-project.org/web/packages/caret/index.html)中 **plsda** 功能的信息。

## 摘要

在这篇文章中，你发现了 3 个线性分类的方法，你可以复制并粘贴到自己的问题中。