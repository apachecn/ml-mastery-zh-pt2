# R 中的惩罚回归

> 原文：<https://machinelearningmastery.com/penalized-regression-in-r/>

最后更新于 2020 年 8 月 15 日

在这篇文章中，你将发现 3 个 R 平台的惩罚回归方法。

你可以复制粘贴这篇文章中的秘籍，在自己的问题上做一个跳跃性的开始，或者用 r 中的线性回归来学习和练习。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

[![Penalized Regression](img/fca501128748943ade8c948a52631511.png)](https://machinelearningmastery.com/wp-content/uploads/2014/07/Penalized-Regression.jpg)

处罚回归
图片由[湾区偏差](https://www.flickr.com/photos/bayareabias/5907585316/in/photolist-hcgaMV-H9Gde-coHJyw-hnbi1L-9ZZ6n6-9ZZ61i-9ZZ5De-a12Vif-9ZYPbi-cyxYZs-cyxTKY-cyxU1C-cyxZGQ-cyxZVU-cyxZf7-cyxZuf-coHuE7-coHHch-7ispTj-7esmd4-coHGZJ-5dFjX3-ayk99U-7c8t2x)提供，保留部分权利

本文中的每个示例都使用了 r 附带的[数据集包](http://stat.ethz.ch/R-manual/R-devel/library/datasets/html/00Index.html)中提供的 [longley](http://stat.ethz.ch/R-manual/R-devel/library/datasets/html/longley.html) 数据集。longley 数据集描述了从 1947 年到 1962 年观察到的 7 个经济变量，用于预测每年的就业人数。

## 里脊回归

岭回归创建了一个线性回归模型，用平方系数之和的 L2 范数进行惩罚。这具有缩小系数值(以及模型的复杂性)的效果，允许对响应贡献较小的一些系数接近于零。

```py
# load the package
library(glmnet)
# load data
data(longley)
x <- as.matrix(longley[,1:6])
y <- as.matrix(longley[,7])
# fit model
fit <- glmnet(x, y, family="gaussian", alpha=0, lambda=0.001)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, x, type="link")
# summarize accuracy
mse <- mean((y - predictions)^2)
print(mse)
```

了解 [glmnet 包](https://cran.r-project.org/web/packages/glmnet/index.html)中的 **glmnet** 功能。

## 最小绝对收缩和选择算子

最小绝对收缩和选择算子(LASSO)创建一个回归模型，用绝对系数之和的 L1 范数进行惩罚。这具有缩小系数值的效果(以及模型的复杂性)，允许对响应影响较小的一些变为零。

```py
# load the package
library(lars)
# load data
data(longley)
x <- as.matrix(longley[,1:6])
y <- as.matrix(longley[,7])
# fit model
fit <- lars(x, y, type="lasso")
# summarize the fit
summary(fit)
# select a step with a minimum error
best_step <- fit$df[which.min(fit$RSS)]
# make predictions
predictions <- predict(fit, x, s=best_step, type="fit")$fit
# summarize accuracy
mse <- mean((y - predictions)^2)
print(mse)
```

了解 [lars 包](https://cran.r-project.org/web/packages/lars/index.html)中的 **lars** 功能。

## 弹性网

弹性网创建了一个回归模型，该模型同时受到 L1 范数和 L2 范数的惩罚。这具有有效收缩系数(如在岭回归中)和将一些系数设置为零(如在 LASSO 中)的效果。

```py
# load the package
library(glmnet)
# load data
data(longley)
x <- as.matrix(longley[,1:6])
y <- as.matrix(longley[,7])
# fit model
fit <- glmnet(x, y, family="gaussian", alpha=0.5, lambda=0.001)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, x, type="link")
# summarize accuracy
mse <- mean((y - predictions)^2)
print(mse)
```

了解 [glmnet 包](https://cran.r-project.org/web/packages/glmnet/index.html)中的 **glmnet** 功能。

## 摘要

在这篇文章中，你发现了 3 个 r 中惩罚回归的方法。

惩罚是属性选择和提高预测模型准确率的有力方法。有关更多信息，请参见库恩和约翰逊的[应用预测建模](https://amzn.to/3iFPHhq)第 6 章，该章为初学者提供了关于 R 线性回归的出色介绍。