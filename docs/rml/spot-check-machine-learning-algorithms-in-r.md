# 抽查 R 中的机器学习算法（下一个项目要尝试的算法）

> 原文：<https://machinelearningmastery.com/spot-check-machine-learning-algorithms-in-r/>

最后更新于 2019 年 8 月 22 日

抽查机器学习算法是如何为数据集找到最佳算法的。

但是应该抽查哪些算法呢？

在这篇文章中，你发现了 8 个机器学习算法，你应该抽查你的数据。

您还可以获得每种算法的配方，您可以将其复制并粘贴到当前或下一个机器学习项目中。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

![Spot Check Machine Learning Algorithms in R](img/8b7b9fec87d455596ba86c8c8e2059d7.png)

核管理委员会抽查 R
图片中的机器学习算法，保留部分权利。

## 数据集的最佳算法

在手之前，您无法知道哪种算法最适合您的数据集。

您必须使用反复试验来发现一个在您的问题上做得很好的算法的简短列表，然后您可以加倍并进一步调整。我称这个过程为抽查。

问题不在于:

> 我应该在数据集上使用什么算法？

相反，它是:

> 我应该在数据集上抽查哪些算法？

### 要抽查哪些算法

您可以猜测哪些算法可能在您的数据集上表现良好，这可能是一个很好的起点。

我建议尝试混合算法，看看什么最擅长挑选数据中的结构。

*   尝试混合算法表示(例如实例和树)。
*   尝试混合学习算法(例如，学习同一类型表示的不同算法)。
*   尝试混合建模类型(例如，线性和非线性函数或参数和非参数)。

具体点说吧。在下一节中，我们将寻找算法，您可以使用这些算法来抽查您在 r

## R 中抽查的算法

r 中有数百种机器学习算法可用。

我建议探索其中的许多，特别是如果对数据集做出准确的预测很重要并且你有时间的话。

通常你没有时间，所以你需要知道几个算法，你绝对必须测试你的问题。

在这一节中，你将发现你应该在 r 中抽查的线性和非线性算法。这不包括集成算法，如增强和装袋，一旦你有了基线，它们就会出现。

每种算法将从两个角度呈现:

1.  用于训练和预测算法的包和函数。
2.  算法的 Caret 包装。

对于给定的算法，您需要知道使用哪个包和函数。在以下情况下需要这样做:

*   您正在研究算法参数以及如何从算法中获得最大收益。
*   你已经发现了使用的最佳算法，需要准备一个最终模型。

您需要知道如何将每个算法与 Caret 一起使用，这样您就可以使用 Caret 的预处理、算法评估和调整功能，在看不见的数据上有效地评估计法的准确性。

两个标准数据集用于演示算法:

*   **波士顿房屋数据集**用于回归(波士顿房屋来自 *mlbench* 库)。
*   **皮马印第安人糖尿病数据集**进行分类(皮马糖尿病患者来自 *mlbench* 库)。

算法分为两组:

*   **线性算法**是更简单的方法，具有很强的偏差，但训练速度很快。
*   **非线性算法**是更复杂的方法，具有较大的方差，但通常更准确。

本节中介绍的每个秘籍都是完整的，并将产生一个结果，以便您可以将其复制并粘贴到当前或下一个机器学习项目中。

我们开始吧。

### 线性算法

这些方法对被建模函数的形式做了大量假设。因此，他们有很高的偏见，但往往训练速度很快。

最终模型通常也很容易(或更容易)解释，这使得它们作为最终模型很受欢迎。如果结果适当准确，如果是线性算法，您可能不需要继续使用非线性方法。

#### 1.线性回归

*lm()* 函数位于*统计*库中，使用普通最小二乘法创建线性回归模型。

```py
# load the library
library(mlbench)
# load data
data(BostonHousing)
# fit model
fit <- lm(medv~., BostonHousing)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, BostonHousing)
# summarize accuracy
mse <- mean((BostonHousing$medv - predictions)^2)
print(mse)
```

lm 实现可以在 Caret 中使用，如下所示:

```py
# load libraries
library(caret)
library(mlbench)
# load dataset
data(BostonHousing)
# train
set.seed(7)
control <- trainControl(method="cv", number=5)
fit.lm <- train(medv~., data=BostonHousing, method="lm", metric="RMSE", preProc=c("center", "scale"), trControl=control)
# summarize fit
print(fit.lm)
```

#### 2.逻辑回归

glm 函数位于 stats 库中，它创建了一个广义线性模型。它可以配置为执行适合二分类问题的逻辑回归。

```py
# load the library
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# fit model
fit <- glm(diabetes~., data=PimaIndiansDiabetes, family=binomial(link='logit'))
# summarize the fit
print(fit)
# make predictions
probabilities <- predict(fit, PimaIndiansDiabetes[,1:8], type='response')
predictions <- ifelse(probabilities > 0.5,'pos','neg')
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)
```

glm 算法可用于 Caret，如下所示:

```py
# load libraries
library(caret)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# train
set.seed(7)
control <- trainControl(method="cv", number=5)
fit.glm <- train(diabetes~., data=PimaIndiansDiabetes, method="glm", metric="Accuracy", preProc=c("center", "scale"), trControl=control)
# summarize fit
print(fit.glm)
```

#### 3.线性判别分析

lda 函数在 MASS 库中，它创建了一个分类问题的线性模型。

```py
# load the libraries
library(MASS)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# fit model
fit <- lda(diabetes~., data=PimaIndiansDiabetes)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, PimaIndiansDiabetes[,1:8])$class
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)
```

lda 算法可用于 Caret，如下所示:

```py
# load libraries
library(caret)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# train
set.seed(7)
control <- trainControl(method="cv", number=5)
fit.lda <- train(diabetes~., data=PimaIndiansDiabetes, method="lda", metric="Accuracy", preProc=c("center", "scale"), trControl=control)
# summarize fit
print(fit.lda)
```

#### 4.正则回归

glmnet 函数位于 glmnet 库中，可用于分类或回归。

分类示例:

```py
# load the library
library(glmnet)
library(mlbench)
# load data
data(PimaIndiansDiabetes)
x <- as.matrix(PimaIndiansDiabetes[,1:8])
y <- as.matrix(PimaIndiansDiabetes[,9])
# fit model
fit <- glmnet(x, y, family="binomial", alpha=0.5, lambda=0.001)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, x, type="class")
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)
```

回归示例:

```py
# load the libraries
library(glmnet)
library(mlbench)
# load data
data(BostonHousing)
BostonHousing$chas <- as.numeric(as.character(BostonHousing$chas))
x <- as.matrix(BostonHousing[,1:13])
y <- as.matrix(BostonHousing[,14])
# fit model
fit <- glmnet(x, y, family="gaussian", alpha=0.5, lambda=0.001)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, x, type="link")
# summarize accuracy
mse <- mean((y - predictions)^2)
print(mse)
```

它还可以配置为执行三种重要类型的正则化:套索、脊和弹性网，方法是将 alpha 参数分别配置为 1、0 或 in [0，1]。

glmnet 实现可以在 Caret 中用于分类，如下所示:

```py
# load libraries
library(caret)
library(mlbench)
library(glmnet)
# Load the dataset
data(PimaIndiansDiabetes)
# train
set.seed(7)
control <- trainControl(method="cv", number=5)
fit.glmnet <- train(diabetes~., data=PimaIndiansDiabetes, method="glmnet", metric="Accuracy", preProc=c("center", "scale"), trControl=control)
# summarize fit
print(fit.glmnet)
```

glmnet 实现可以在 caret 中用于回归，如下所示:

```py
# load libraries
library(caret)
library(mlbench)
library(glmnet)
# Load the dataset
data(BostonHousing)
# train
set.seed(7)
control <- trainControl(method="cv", number=5)
fit.glmnet <- train(medv~., data=BostonHousing, method="glmnet", metric="RMSE", preProc=c("center", "scale"), trControl=control)
# summarize fit
print(fit.glmnet)
```

### 非线性算法

这些是机器学习算法，对被建模的函数做较少的假设。因此，它们具有更高的方差，但通常导致更高的准确率。他们增加的灵活性也会使他们训练更慢或者增加他们的记忆需求。

#### 1.k-最近邻

knn3 函数在 caret 库中，不创建模型，而是直接从训练集中进行预测。它可以用于分类或回归。

分类示例:

```py
# knn direct classification

# load the libraries
library(caret)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# fit model
fit <- knn3(diabetes~., data=PimaIndiansDiabetes, k=3)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, PimaIndiansDiabetes[,1:8], type="class")
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)
```

回归示例:

```py
# load the libraries
library(caret)
library(mlbench)
# load data
data(BostonHousing)
BostonHousing$chas <- as.numeric(as.character(BostonHousing$chas))
x <- as.matrix(BostonHousing[,1:13])
y <- as.matrix(BostonHousing[,14])
# fit model
fit <- knnreg(x, y, k=3)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, x)
# summarize accuracy
mse <- mean((BostonHousing$medv - predictions)^2)
print(mse)
```

knn 实现可以在 caret train()函数中用于分类，如下所示:

```py
# load libraries
library(caret)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# train
set.seed(7)
control <- trainControl(method="cv", number=5)
fit.knn <- train(diabetes~., data=PimaIndiansDiabetes, method="knn", metric="Accuracy", preProc=c("center", "scale"), trControl=control)
# summarize fit
print(fit.knn)
```

knn 实现可以在 caret train()函数中用于回归，如下所示:

```py
# load libraries
library(caret)
data(BostonHousing)
# Load the dataset
data(BostonHousing)
# train
set.seed(7)
control <- trainControl(method="cv", number=5)
fit.knn <- train(medv~., data=BostonHousing, method="knn", metric="RMSE", preProc=c("center", "scale"), trControl=control)
# summarize fit
print(fit.knn)
```

#### 2.朴素贝叶斯

天真贝叶斯函数在 e1071 库中，它独立地对每个属性到结果变量的概率进行建模。它可以用于分类问题。

```py
# load the libraries
library(e1071)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# fit model
fit <- naiveBayes(diabetes~., data=PimaIndiansDiabetes)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, PimaIndiansDiabetes[,1:8])
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)
```

一个非常相似的朴素贝叶斯实现(来自 klaR 库的朴素贝叶斯)可以如下使用 Caret:

```py
# load libraries
library(caret)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# train
set.seed(7)
control <- trainControl(method="cv", number=5)
fit.nb <- train(diabetes~., data=PimaIndiansDiabetes, method="nb", metric="Accuracy", trControl=control)
# summarize fit
print(fit.nb)
```

#### 3.支持向量机

ksvm 函数在 kernlab 包中，可用于分类或回归。它是 LIBSVM 库的包装器，并提供了一套内核类型和配置选项。

这些示例使用径向基核。

分类示例:

```py
 load the libraries
library(kernlab)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# fit model
fit <- ksvm(diabetes~., data=PimaIndiansDiabetes, kernel="rbfdot")
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, PimaIndiansDiabetes[,1:8], type="response")
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)
```

回归示例:

```py
# load the libraries
library(kernlab)
library(mlbench)
# load data
data(BostonHousing)
# fit model
fit <- ksvm(medv~., BostonHousing, kernel="rbfdot")
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, BostonHousing)
# summarize accuracy
mse <- mean((BostonHousing$medv - predictions)^2)
print(mse)
```

带有径向基核的 SVM 实现可以与 Caret 一起使用进行分类，如下所示:

```py
# load libraries
library(caret)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# train
set.seed(7)
control <- trainControl(method="cv", number=5)
fit.svmRadial <- train(diabetes~., data=PimaIndiansDiabetes, method="svmRadial", metric="Accuracy", trControl=control)
# summarize fit
print(fit.svmRadial)
```

带有径向基核的 SVM 实现可以与 Caret 一起用于回归，如下所示:

```py
# load libraries
library(caret)
library(mlbench)
# Load the dataset
data(BostonHousing)
# train
set.seed(7)
control <- trainControl(method="cv", number=5)
fit.svmRadial <- train(medv~., data=BostonHousing, method="svmRadial", metric="RMSE", trControl=control)
# summarize fit
print(fit.svmRadial)
```

#### 4.分类和回归树

rpart 库中的 rpart 函数为分类和回归提供了一个 CART 的实现。

分类示例:

```py
# load the libraries
library(rpart)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# fit model
fit <- rpart(diabetes~., data=PimaIndiansDiabetes)
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, PimaIndiansDiabetes[,1:8], type="class")
# summarize accuracy
table(predictions, PimaIndiansDiabetes$diabetes)
```

回归示例:

```py
# load the libraries
library(rpart)
library(mlbench)
# load data
data(BostonHousing)
# fit model
fit <- rpart(medv~., data=BostonHousing, control=rpart.control(minsplit=5))
# summarize the fit
print(fit)
# make predictions
predictions <- predict(fit, BostonHousing[,1:13])
# summarize accuracy
mse <- mean((BostonHousing$medv - predictions)^2)
print(mse)
```

rpart 实现可以与 caret 一起使用进行分类，如下所示:

```py
# load libraries
library(caret)
library(mlbench)
# Load the dataset
data(PimaIndiansDiabetes)
# train
set.seed(7)
control <- trainControl(method="cv", number=5)
fit.rpart <- train(diabetes~., data=PimaIndiansDiabetes, method="rpart", metric="Accuracy", trControl=control)
# summarize fit
print(fit.rpart)
```

rpart 实现可以与 Caret 一起用于回归，如下所示:

```py
# load libraries
library(caret)
library(mlbench)
# Load the dataset
data(BostonHousing)
# train
set.seed(7)
control <- trainControl(method="cv", number=2)
fit.rpart <- train(medv~., data=BostonHousing, method="rpart", metric="RMSE", trControl=control)
# summarize fit
print(fit.rpart)
```

### 其他算法

R 提供了许多其他算法，并且在 caret 中可用。

我建议你去探索它们，并在你的下一个机器学习项目中，在你自己的必须尝试算法的简短列表中添加更多的算法。

您可以在本页的 Caret 包中找到机器学习函数和包到它们名称的映射:

*   [Caret 模型列表](https://topepo.github.io/caret/modelList.html)

如果您在 caret 中使用一个算法，并且想知道它属于哪个包，以便您可以阅读参数并从中获得更多信息，此页面非常有用。

如果您直接在 R 中使用机器学习算法，并且想知道它如何在 Caret 中使用，此页面也很有用。

## 摘要

在这篇文章中，你发现了 8 种不同的算法，可以用来抽查你的数据集。具体来说:

*   线性回归
*   逻辑回归
*   线性判别分析
*   正则回归
*   k-最近邻
*   朴素贝叶斯
*   支持向量机
*   分类和回归树

您学习了每个算法使用哪些包和函数。您还学习了如何将每个算法与提供算法评估和调整功能的 caret 包一起使用。

您可以使用这些算法作为模板，对您当前或下一个机器学习项目进行抽查

## 你的下一步

你试过这些秘籍吗？

1.  开始你的互动环境。
2.  键入或复制粘贴上面的秘籍并试用。
3.  使用 R 中的内置帮助来了解有关所用函数的更多信息。

你有问题吗？在评论里问，我会尽力回答。