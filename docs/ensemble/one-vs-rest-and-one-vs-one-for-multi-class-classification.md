# 多类分类的一对一和一对剩余

> 原文：<https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/>

最后更新于 2021 年 4 月 27 日

并非所有分类预测模型都支持多类分类。

感知器、逻辑回归和支持向量机等算法是为二进制分类而设计的，并不支持两类以上的分类任务。

将二进制分类算法用于多分类问题的一种方法是将多类别数据集分割成多个二进制类别数据集，并在每个二进制类别数据集上拟合二进制分类模型。这种方法的两个不同的例子是“一对一”和“一对一”策略。

在本教程中，您将发现多类分类的一对一和一对一策略。

完成本教程后，您将知道:

*   像逻辑回归和 SVM 这样的二分类模型本身不支持多类分类，需要元策略。
*   One-vs-Rest 策略将多类分类拆分为每个类一个二分类问题。
*   一对一策略将多类分类分成每对类一个二进制分类问题。

**用我的新书[Python 集成学习算法](https://machinelearningmastery.com/ensemble-learning-algorithms-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Use One-vs-Rest and One-vs-One for Multi-Class Classification](img/45e27561a5be060193a7a4ddbc7d32bb.png)

如何使用一对一休息和一对一进行多类分类
图片由 [Espen Sundve](https://flickr.com/photos/sundve/3744159600/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  用于多类分类的二分类器
2.  多类分类中的一对多
3.  多类分类的一对一与一对一

## 用于多类分类的二分类器

分类是一个预测性的建模问题，包括给一个例子分配一个类标签。

二进制分类是那些例子被指定为两类中的一类的任务。多类分类是那些例子被指定为两个以上类中的一个的任务。

*   **二分类**:两类分类任务。
*   **多类分类**:两类以上的分类任务。

一些算法是为二进制分类问题设计的。例子包括:

*   逻辑回归
*   感知器
*   支持向量机

因此，它们不能用于多类分类任务，至少不能直接使用。

相反，启发式方法可以用于将多类分类问题分割成多个二进制类别数据集，并分别训练二进制分类模型。

这些启发式方法的两个例子包括:

*   一对一对多休息
*   一对一(OvO)

让我们仔细看看每一个。

## 多类分类中的一对多

One-vs-rest(简称 OvR，也称为 One-vs-All 或 OvA)是一种启发式方法，用于使用二进制分类算法进行多类分类。

它包括将多类数据集拆分成多个二进制分类问题。然后对每个二进制分类问题训练二进制分类器，并使用最有把握的模型进行预测。

例如，给定一个多类分类问题，其中每个类都有例子。这可以分为如下三个二元类别数据集:

*   **二分类问题 1** :红色 vs【蓝色，绿色】
*   **二分类问题 2** :蓝色 vs【红色，绿色】
*   **二进制分类问题 3** :绿色 vs【红色，蓝色】

这种方法的一个可能的缺点是，它需要为每个类创建一个模型。例如，三个类需要三个模型。对于大型数据集(例如，数百万行)、慢速模型(例如，神经网络)或非常大量的类(例如，数百个类)，这可能是一个问题。

> 显而易见的方法是使用一对其余的方法(也称为一对全部)，其中我们训练 C 二进制分类器 fc(x)，其中来自 C 类的数据被视为正，来自所有其他类的数据被视为负。

—第 503 页，[机器学习:概率视角](https://amzn.to/38wVdOd)，2012 年。

这种方法要求每个模型预测一个类成员概率或类似概率的分数。然后使用这些分数的 argmax(分数最大的班级指数)来预测班级。

这种方法通常用于自然预测数值类成员概率或分数的算法，例如:

*   逻辑回归
*   感知器

因此，当使用这些算法进行多类分类时，Sklearn 库中这些算法的实现默认实现了 OvR 策略。

我们可以用一个使用[物流分类](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)算法的 3 类分类问题的例子来证明这一点。处理多类分类的策略可以通过“*多类*参数设置，一对多策略可以设置为“ *ovr* ”。

下面列出了使用内置的一对多策略拟合多类分类的逻辑回归模型的完整示例。

```py
# logistic regression for multi-class classification using built-in one-vs-rest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = LogisticRegression(multi_class='ovr')
# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)
```

Sklearn 库还提供了一个单独的 [OneVsRestClassifier](https://Sklearn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) 类，该类允许 one-vs-rest 策略用于任何分类器。

此类可用于使用二进制分类器(如逻辑回归或感知器)进行多类分类，或者甚至使用其他本身支持多类分类的分类器。

它非常容易使用，并且要求将用于二进制分类的分类器作为参数提供给 *OneVsRestClassifier* 。

下面的例子演示了如何使用 *OneVsRestClassifier* 类和作为二进制分类模型的*物流分类*类。

```py
# logistic regression for multi-class classification using a one-vs-rest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = LogisticRegression()
# define the ovr strategy
ovr = OneVsRestClassifier(model)
# fit model
ovr.fit(X, y)
# make predictions
yhat = ovr.predict(X)
```

## 多类分类的一对一与一对一

One-vs-One(简称 OvO)是另一种使用二进制分类算法进行多类分类的启发式方法。

与 one-vs-rest 类似，one-vs-one 将多类类别数据集拆分为二分类问题。与 one-vs-rest 为每个类将数据集拆分为一个二进制数据集不同，one-vs-one 方法为每个类将数据集拆分为一个数据集，而不是其他类。

例如，考虑一个包含四个类的多类分类问题:“*红色*、“*蓝色*、”和“*绿色*、“*黄色*”这可以分为六个二元类别数据集，如下所示:

*   **二进制分类问题 1** :红色对蓝色
*   **二进制分类问题 2** :红色对绿色
*   **二进制分类问题 3** :红色对黄色
*   **二进制分类问题 4** :蓝色对绿色
*   **二进制分类问题 5** :蓝色对黄色
*   **二进制分类问题 6** :绿色对黄色

这比上一节描述的一对多策略要多得多的数据集和模型。

计算二进制数据集以及模型数量的公式如下:

*   (数字类别*(数字类别–1))/2

我们可以看到，对于四个类，这给了我们六个二分类问题的期望值:

*   (数字类别*(数字类别–1))/2
*   (4 * (4 – 1)) / 2
*   (4 * 3) / 2
*   12 / 2
*   six

每个二分类模型可以预测一个类别标签，并且通过一对一策略预测具有最多预测或投票的模型。

> 另一种方法是引入 K(k1)/2 二元判别函数，每一对可能的类对应一个。这就是所谓的一对一分类器。然后根据判别函数中的多数票对每个点进行分类。

—第 183 页，[模式识别与机器学习](https://amzn.to/2RFWf3N)，2006。

类似地，如果二进制分类模型预测数值类成员，例如概率，那么分数总和(具有最大总和分数的类)的 argmax 被预测为类标签。

经典地，这种方法被建议用于支持向量机(SVM)和相关的基于核的算法。这被认为是因为内核方法的表现与训练数据集的大小不成比例，使用训练数据的子集可能会抵消这种影响。

Sklearn 中的支持向量机实现由 [SVC](https://Sklearn.org/stable/modules/generated/sklearn.svm.SVC.html) 类提供，支持多类分类问题的一对一方法。这可以通过将“ *decision_function_shape* ”参数设置为“ *ovo* 来实现。

下面的例子演示了 SVM 使用一对一方法进行多类分类。

```py
# SVM for multi-class classification using built-in one-vs-one
from sklearn.datasets import make_classification
from sklearn.svm import SVC
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = SVC(decision_function_shape='ovo')
# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)
```

Sklearn 库还提供了一个单独的 [OneVsOneClassifier](https://Sklearn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html) 类，允许一对一策略用于任何分类器。

该类可以与二进制分类器(如 SVM、逻辑回归或感知器)一起用于多类分类，甚至可以与其他本身支持多类分类的分类器一起使用。

它非常容易使用，并且要求将用于二进制分类的分类器作为参数提供给 *OneVsOneClassifier* 。

下面的例子演示了如何使用 *OneVsOneClassifier* 类和一个用作二进制分类模型的 SVC 类。

```py
# SVM for multi-class classification using one-vs-one
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = SVC()
# define ovo strategy
ovo = OneVsOneClassifier(model)
# fit model
ovo.fit(X, y)
# make predictions
yhat = ovo.predict(X)
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [模式识别与机器学习](https://amzn.to/2RFWf3N)，2006。
*   [机器学习:概率视角](https://amzn.to/38wVdOd)，2012。

### 蜜蜂

*   [多类和多标签算法，Sklearn API](https://Sklearn.org/stable/modules/multiclass.html) 。
*   [硬化。多类。OneVsRestClassifier API](https://Sklearn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) 。
*   [硬化。多类。OneVsOneClassifier API](https://Sklearn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html) 。

### 文章

*   [多类分类，维基百科](https://en.wikipedia.org/wiki/Multiclass_classification)。

## 摘要

在本教程中，您发现了多类分类的一对一和一对一策略。

具体来说，您了解到:

*   像逻辑回归和 SVM 这样的二分类模型本身不支持多类分类，需要元策略。
*   One-vs-Rest 策略将多类分类拆分为每个类一个二分类问题。
*   一对一策略将多类分类分成每对类一个二进制分类问题。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。