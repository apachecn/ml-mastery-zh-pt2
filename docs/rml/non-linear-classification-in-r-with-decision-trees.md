# R 中的决策树非线性分类

> 原文：<https://machinelearningmastery.com/non-linear-classification-in-r-with-decision-trees/>

最后更新于 2019 年 8 月 22 日

在这篇文章中，你将会发现 7 个非线性分类的方法

本文所有秘籍均使用[数据集包](http://stat.ethz.ch/R-manual/R-patched/library/datasets/html/00Index.html)中提供的[鸢尾花数据集](http://stat.ethz.ch/R-manual/R-patched/library/datasets/html/iris.html)。该数据集描述了鸢尾花的测量结果，并要求对三种花卉中的一种进行分类。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

[![classification with decision trees](img/4cec4d88ec287c897967c86d177005a1.png)](https://machinelearningmastery.com/wp-content/uploads/2014/08/classification-with-decision-trees.jpg)

决策树分类
图片由 [stwn](https://www.flickr.com/photos/stwn/14508509530) 提供，保留部分权利

## 分类和回归树

分类和回归树(CART)根据最小化损失函数的值(如误差平方和)分割属性。

下面的方法演示了虹膜数据集的递归分割决策树方法。

```py
# load the package
library(rpart)
# load data
data(iris)
# fit model
fit <- rpart(Species~., data=iris)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4], type="class")
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于 **rpart** 功能和 [rpart 包](https://cran.r-project.org/web/packages/rpart/index.html)的信息。

## C4.5

C4.5 算法是 ID3 算法的扩展，它构建了一个决策树来最大化信息增益(熵差)。

下面的方法演示了虹膜数据集上的 C4.5(在 Weka 中称为 J48)决策树方法。

```py
# load the package
library(RWeka)
# load data
data(iris)
# fit model
fit <- J48(Species~., data=iris)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4])
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于 **J48** 功能和 [RWeka 集成](https://cran.r-project.org/web/packages/RWeka/index.html)的信息。

## 部分

PART 是一个规则系统，它为数据集创建剪枝的 C4.5 决策树，并提取规则，规则覆盖的那些实例将从训练数据中移除。重复该过程，直到所有实例都被提取的规则覆盖。

以下配方演示了虹膜数据集上的零件规则系统方法。

```py
# load the package
library(RWeka)
# load data
data(iris)
# fit model
fit <- PART(Species~., data=iris)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4])
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于**零件**功能和 [RWeka 集成](https://cran.r-project.org/web/packages/RWeka/index.html)的信息。

## 装袋车

自举聚合(Bagging)是一种集成方法，它从同一数据集的不同子样本创建多个相同类型的模型。来自每个单独模型的预测被组合在一起以提供更好的结果。这种方法对于像决策树这样的高方差方法特别有效。

下面的方法演示了 bagging 在 iris 数据集递归分割决策树中的应用。

```py
# load the package
library(ipred)
# load data
data(iris)
# fit model
fit <- bagging(Species~., data=iris)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4], type="class")
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于**装袋**功能和[智能包装](https://cran.r-project.org/web/packages/ipred/index.html)的信息。

## 随机森林

随机森林是决策树装袋的变体，它通过将每个决策点的树的可用属性减少为随机子样本。这进一步增加了树的方差，需要更多的树。

下面的方法演示了应用于 iris 数据集的随机森林方法。

```py
# load the package
library(randomForest)
# load data
data(iris)
# fit model
fit <- randomForest(Species~., data=iris)
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, iris[,1:4])
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于**随机森林**功能和[随机森林集成](https://cran.r-project.org/web/packages/randomForest/index.html)的信息。

## 梯度增压机

Boosting 是一种集成方法，开发用于分类以减少偏差，其中添加模型来学习现有模型中的错误分类错误。它以梯度提升机(GBM)的形式进行了推广和调整，用于 CART 决策树的分类和回归。

以下配方演示了虹膜数据集中的梯度提升机(GBM)方法。

```py
# load the package
library(gbm)
# load data
data(iris)
# fit model
fit <- gbm(Species~., data=iris, distribution="multinomial")
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, iris)
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于 **gbm** 功能和 [gbm 包](https://cran.r-project.org/web/packages/gbm/index.html)的信息。

## 增压 C5.0

C5.0 方法是 C4.5 的进一步扩展，也是该系列方法的巅峰。它很长一段时间都是私有的，尽管代码是最近发布的，并且在 C50 包中提供。

下面的方法演示了应用于虹膜数据集的 C5.0 增强方法。

```py
# load the package
library(C50)
# load data
data(iris)
# fit model
fit <- C5.0(Species~., data=iris, trials=10)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, iris)
# summarize accuracy
table(predictions, iris$Species)
```

了解更多关于 [C50 集成](https://cran.r-project.org/web/packages/C50/index.html)中 **C5.0** 功能的信息。

## 摘要

在这篇文章中，你发现了 7 个非线性分类的方法，它们使用了 R 中的决策树，使用了鸢尾花数据集。

每个配方都是通用的，可以根据自己的问题进行复制、粘贴和修改。