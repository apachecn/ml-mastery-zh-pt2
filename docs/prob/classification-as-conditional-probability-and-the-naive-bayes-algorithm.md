# 如何在 Python 中从零开始开发朴素贝叶斯分类器

> 原文：<https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/>

最后更新于 2020 年 1 月 10 日

分类是一个预测建模问题，包括给给定的输入数据样本分配一个标签。

分类预测建模问题可以被框架化为计算给定数据样本的类标签的条件概率。[贝叶斯定理](https://machinelearningmastery.com/bayes-theorem-for-machine-learning/)为计算这种条件概率提供了一种有原则的方法，尽管实际上需要大量样本(非常大的数据集)并且计算成本很高。

相反，贝叶斯定理的计算可以通过做一些假设来简化，例如每个输入变量都独立于所有其他输入变量。虽然这是一个戏剧性的和不现实的假设，但它具有使条件概率的计算易于处理的效果，并产生了一个称为朴素贝叶斯的有效分类模型。

在本教程中，您将发现用于分类预测建模的朴素贝叶斯算法。

完成本教程后，您将知道:

*   如何将分类预测建模框架化为条件概率模型？
*   如何利用贝叶斯定理求解分类的条件概率模型？
*   如何实现简化的贝叶斯分类定理，称为朴素贝叶斯算法。

**用我的新书[机器学习概率](https://machinelearningmastery.com/probability-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2019 年 10 月更新**:修正了数学符号中的小的不一致问题。
*   **2020 年 1 月更新**:针对 Sklearn v0.22 API 的变化进行了更新。

![How to Develop a Naive Bayes Classifier from Scratch in Python](img/6aa80275375e2571b7e3fb503b63f08f.png)

如何在 Python 中从零开始开发朴素贝叶斯分类器
图片由 [Ryan Dickey](https://www.flickr.com/photos/meesterdickey/43549200532/) 提供，保留部分权利。

## 教程概述

本教程分为五个部分；它们是:

1.  分类的条件概率模型
2.  简化或朴素贝叶斯
3.  如何计算先验概率和条件概率
4.  朴素贝叶斯的工作示例
5.  使用朴素贝叶斯的 5 个技巧

## 分类的条件概率模型

在机器学习中，我们经常对一个预测建模问题感兴趣，在这个问题中，我们希望为给定的观察预测一个类标签。例如，根据花的尺寸对植物种类进行分类。

这类问题被称为分类预测建模问题，与涉及预测数值的回归问题相反。模型的观察或输入称为 *X* ，模型的类标签或输出称为 *y* 。

X 和 y 一起表示从域中收集的观察值，即用于拟合模型的训练数据的表或矩阵(列和行或特征和样本)。模型必须学会如何将具体的例子映射到类标签或 *y = f(X)* 上，从而将错误分类的误差降到最低。

解决这个问题的一种方法是开发一个概率模型。从概率的角度来看，我们感兴趣的是在给定观察的情况下，估计类标签的条件概率。

例如，一个分类问题可能有 k 个类标签 *y1，y2，…，yk* 和 n 个输入变量， *X1，X2，…，Xn* 。我们可以为每个列计算带有给定实例或一组输入值的类标签的条件概率 *x1，x2，…，xn* ，如下所示:

*   P(yi | x1、x2、…、xn)

然后，可以为问题中的每个类别标签计算条件概率，具有最高概率的标签可以作为最可能的分类返回。

条件概率可以使用联合概率来计算，尽管这很难处理。[贝叶斯定理](https://en.wikipedia.org/wiki/Bayes%27_theorem)提供了一种计算条件概率的原则性方法。

贝叶斯定理的简单计算形式如下:

*   P(A|B) = P(B|A) * P(A) / P(B)

其中我们对计算 P(A|B)感兴趣的概率称为后验概率，事件 P(A)的边缘概率称为先验概率。

我们可以用贝叶斯定理将分类框架为一个条件分类问题，如下所示:

*   P(yi | x1，x2，…，xn) = P(x1，x2，…，xn | yi) * P(yi) / P(x1，x2，…，xn)

先验的 *P(yi)* 很容易从数据集中估计，但是基于类 *P(x1，x2，…，xn | yi)* 的观测的条件概率是不可行的，除非例子的数量非常大，例如大到足以有效地估计所有不同可能值组合的概率分布。

因此，贝叶斯定理的直接应用也变得难以处理，尤其是当变量或特征的数量( *n* )增加时。

## 简化或朴素贝叶斯

将贝叶斯定理用于条件概率分类模型的解决方案是简化计算。

贝叶斯定理假设每个输入变量都依赖于所有其他变量。这是计算复杂的一个原因。我们可以去掉这个假设，将每个输入变量视为彼此独立的。

这将模型从依赖条件概率模型变为独立条件概率模型，并极大地简化了计算。

首先，从计算 *P(x1，x2，…，xn)* 中移除分母，因为它是用于计算给定实例的每个类的条件概率的常数，并且具有归一化结果的效果。

*   P(yi | x1，x2，…，xn) = P(x1，x2，…，xn | yi) * P(yi)

接下来，给定类别标签的所有变量的条件概率被改变为给定类别标签的每个变量值的单独的条件概率。然后将这些独立的条件变量相乘。例如:

*   P(yi | x1，x2，…，xn)= P(x1 | yi)* P(x2 | yi)*…P(xn | yi)* P(yi)

可以对每个类标签执行该计算，并且可以选择具有最大概率的标签作为给定实例的分类。这个决策规则被称为最大后验概率决策规则。

贝叶斯定理的这种简化是常见的，广泛用于分类预测建模问题，通常被称为[朴素贝叶斯](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)。

“天真”一词是法语，通常在“我”的前面加上一个“T2”分音符“T3”(umlaut)，为了简单起见，这个词通常被省略了，“贝叶斯”被大写，因为它是以“T4”牧师托马斯·贝叶斯的名字命名的。

## 如何计算先验概率和条件概率

现在我们知道了什么是朴素贝叶斯，我们可以更仔细地看看如何计算方程的元素。

先验概率的计算很简单。可以通过将训练数据集中具有类标签的观察频率除以训练数据集中的示例(行)总数来估计。例如:

*   P(yi) =有 yi 的例子/全部例子

给定类别标签的特征值的条件概率也可以从数据中估计。具体来说，属于给定类的那些数据示例，以及每个变量的一个数据分布。这意味着如果存在 *K* 类和 *n* 变量，那么 *k * n* 不同的概率分布必须被创建和维护。

根据每个功能的数据类型，需要不同的方法。具体而言，数据用于估计三个标准概率分布之一的参数。

对于分类变量，如计数或标签，可以使用多项式分布。如果变量是二进制的，如是/否或真/假，可以使用二项式分布。如果变量是数值型的，如测量值，通常使用高斯分布。

*   **二元**:二项式分布。
*   **分类**:多项式分布。
*   **数值**:高斯分布。

这三个分布非常常见，以至于朴素贝叶斯实现经常以分布命名。例如:

*   **二项朴素贝叶斯**:使用二项分布的朴素贝叶斯。
*   **多项式朴素贝叶斯**:使用多项式分布的朴素贝叶斯。
*   **高斯朴素贝叶斯**:使用高斯分布的朴素贝叶斯。

输入变量具有混合数据类型的数据集可能需要为每个变量选择不同类型的数据分布。

使用三种常见发行版中的一种并不是强制性的；例如，如果已知实值变量具有不同的特定分布，例如指数分布，则可以使用该特定分布来代替。如果实值变量没有定义明确的分布，如双峰或多峰，则可以使用核密度估计器来估计概率分布。

朴素贝叶斯算法被证明是有效的，因此在文本分类任务中很受欢迎。文档中的单词可以被编码为二进制(单词存在)、计数(单词出现)或频率(tf/idf)输入向量以及分别使用的二进制、多项式或高斯概率分布。

## 朴素贝叶斯的工作示例

在本节中，我们将通过一个机器学习数据集上的小例子来具体说明朴素贝叶斯计算。

我们可以使用 Sklearn API 中的 [make_blobs()函数](http://Sklearn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)生成一个小的人为二进制(2 类)分类问题。

下面的示例生成了 100 个带有两个数字输入变量的示例，每个变量被分配了两个类中的一个。

```py
# example of generating a small classification dataset
from sklearn.datasets import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# summarize
print(X.shape, y.shape)
print(X[:5])
print(y[:5])
```

运行该示例会生成数据集并汇总大小，从而确认数据集是按预期生成的。

“ *random_state* ”参数设置为 1，确保每次运行代码时生成相同的随机观察样本。

前五个示例的输入和输出元素也被打印出来，显示了两个输入变量实际上是数字，每个示例的类标签不是 0 就是 1。

```py
(100, 2) (100,)
[[-10.6105446    4.11045368]
 [  9.05798365   0.99701708]
 [  8.705727     1.36332954]
 [ -8.29324753   2.35371596]
 [  6.5954554    2.4247682 ]]
[0 1 1 0 1]
```

我们将使用高斯概率分布对数字输入变量建模。

这可以使用规范的 SciPy API 来实现。首先，可以通过指定分布的参数(例如，平均值和标准偏差)来构建分布，然后可以使用[范数. pdf()函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)对特定值的概率密度函数进行采样。

我们可以使用*均值()*和 *std()* NumPy 函数从数据集中估计分布参数。

下面的 *fit_distribution()* 函数获取一个变量的数据样本，并拟合一个数据分布。

```py
# fit a probability distribution to a univariate data sample
def fit_distribution(data):
	# estimate parameters
	mu = mean(data)
	sigma = std(data)
	print(mu, sigma)
	# fit distribution
	dist = norm(mu, sigma)
	return dist
```

回想一下，我们对每个输入变量的条件概率感兴趣。这意味着我们需要每个输入变量有一个分布，每个类标签有一组分布，或者总共有四个分布。

首先，我们必须为每个类标签将数据分成样本组。

```py
...
# sort data into classes
Xy0 = X[y == 0]
Xy1 = X[y == 1]
print(Xy0.shape, Xy1.shape)
```

然后，我们可以使用这些组来计算属于每个组的数据样本的先验概率。

这将是 50%，假设我们已经在两个类中创建了相同数量的例子；然而，我们将计算这些先验的完整性。

```py
...
# calculate priors
priory0 = len(Xy0) / len(X)
priory1 = len(Xy1) / len(X)
print(priory0, priory1)
```

最后，我们可以调用我们定义的 *fit_distribution()* 函数，为每个变量、每个类标签准备一个概率分布。

```py
...
# create PDFs for y==0
X1y0 = fit_distribution(Xy0[:, 0])
X2y0 = fit_distribution(Xy0[:, 1])
# create PDFs for y==1
X1y1 = fit_distribution(Xy1[:, 0])
X2y1 = fit_distribution(Xy1[:, 1])
```

将所有这些联系在一起，数据集的完整概率模型如下所示。

```py
# summarize probability distributions of the dataset
from sklearn.datasets import make_blobs
from scipy.stats import norm
from numpy import mean
from numpy import std

# fit a probability distribution to a univariate data sample
def fit_distribution(data):
	# estimate parameters
	mu = mean(data)
	sigma = std(data)
	print(mu, sigma)
	# fit distribution
	dist = norm(mu, sigma)
	return dist

# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# sort data into classes
Xy0 = X[y == 0]
Xy1 = X[y == 1]
print(Xy0.shape, Xy1.shape)
# calculate priors
priory0 = len(Xy0) / len(X)
priory1 = len(Xy1) / len(X)
print(priory0, priory1)
# create PDFs for y==0
X1y0 = fit_distribution(Xy0[:, 0])
X2y0 = fit_distribution(Xy0[:, 1])
# create PDFs for y==1
X1y1 = fit_distribution(Xy1[:, 0])
X2y1 = fit_distribution(Xy1[:, 1])
```

运行该示例首先将数据集分成两组用于两个类标签，并确认每组的大小是均匀的，优先级是 50%。

然后为每个类别标签的每个变量准备概率分布，并报告每个分布的平均值和标准偏差参数，确认分布不同。

```py
(50, 2) (50, 2)
0.5 0.5
-1.5632888906409914 0.787444265443213
4.426680361487157 0.958296071258367
-9.681177100524485 0.8943078901048118
-3.9713794295185845 0.9308177595208521
```

接下来，我们可以使用准备好的概率模型进行预测。

每个类别标签的独立条件概率可以使用该类别的先验(50%)和每个变量的值的条件概率来计算。

给定每个变量的先验和条件概率分布，下面的*概率()*函数对一个输入示例(两个值的数组)执行该计算。返回的值是一个分数，而不是一个概率，因为数量没有标准化，这是在实现朴素贝叶斯时经常执行的一个简化。

```py
# calculate the independent conditional probability
def probability(X, prior, dist1, dist2):
	return prior * dist1.pdf(X[0]) * dist2.pdf(X[1])
```

我们可以用这个函数来计算一个例子属于每个类的概率。

首先，我们可以选择一个例子进行分类；在这种情况下，数据集中的第一个示例。

```py
...
# classify one example
Xsample, ysample = X[0], y[0]
```

接下来，我们可以计算属于第一类的例子的分数，然后是第二类，然后报告结果。

```py
...
py0 = probability(Xsample, priory0, distX1y0, distX2y0)
py1 = probability(Xsample, priory1, distX1y1, distX2y1)
print('P(y=0 | %s) = %.3f' % (Xsample, py0*100))
print('P(y=1 | %s) = %.3f' % (Xsample, py1*100))
```

分数最大的班级将成为最终的分类。

将这些联系在一起，下面列出了拟合朴素贝叶斯模型并使用它进行预测的完整示例。

```py
# example of preparing and making a prediction with a naive bayes model
from sklearn.datasets import make_blobs
from scipy.stats import norm
from numpy import mean
from numpy import std

# fit a probability distribution to a univariate data sample
def fit_distribution(data):
	# estimate parameters
	mu = mean(data)
	sigma = std(data)
	print(mu, sigma)
	# fit distribution
	dist = norm(mu, sigma)
	return dist

# calculate the independent conditional probability
def probability(X, prior, dist1, dist2):
	return prior * dist1.pdf(X[0]) * dist2.pdf(X[1])

# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# sort data into classes
Xy0 = X[y == 0]
Xy1 = X[y == 1]
# calculate priors
priory0 = len(Xy0) / len(X)
priory1 = len(Xy1) / len(X)
# create PDFs for y==0
distX1y0 = fit_distribution(Xy0[:, 0])
distX2y0 = fit_distribution(Xy0[:, 1])
# create PDFs for y==1
distX1y1 = fit_distribution(Xy1[:, 0])
distX2y1 = fit_distribution(Xy1[:, 1])
# classify one example
Xsample, ysample = X[0], y[0]
py0 = probability(Xsample, priory0, distX1y0, distX2y0)
py1 = probability(Xsample, priory1, distX1y1, distX2y1)
print('P(y=0 | %s) = %.3f' % (Xsample, py0*100))
print('P(y=1 | %s) = %.3f' % (Xsample, py1*100))
print('Truth: y=%d' % ysample)
```

运行该示例首先像以前一样准备先验概率和条件概率，然后使用它们为一个示例进行预测。

属于 *y=0* 的例子的得分约为 0.3(回想一下这是一个非标准化的概率)，而属于 *y=1* 的例子的得分为 0.0。因此，我们将该示例归类为属于 *y=0* 。

在这种情况下，真实或实际的结果是已知的， *y=0* ，这与我们的朴素贝叶斯模型的预测相匹配。

```py
P(y=0 | [-0.79415228  2.10495117]) = 0.348
P(y=1 | [-0.79415228  2.10495117]) = 0.000
Truth: y=0
```

在实践中，使用[朴素贝叶斯算法](https://Sklearn.org/stable/modules/naive_bayes.html)的优化实现是一个好主意。Sklearn 库提供了三种实现，三种主要概率分布各一种；例如，二项式、多项式和高斯分布输入变量分别为 [BernoulliNB](https://Sklearn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html) 、[多项式 B](https://Sklearn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) 和[高斯分布输入变量。](https://Sklearn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)

为了使用 Sklearn 朴素贝叶斯模型，首先定义模型，然后将其拟合到训练数据集上。一旦拟合，概率可以通过 *predict_proba()* 函数预测，类标签可以通过 *predict()* 函数直接预测。

下面列出了将高斯朴素贝叶斯模型(GaussianNB)拟合到同一测试数据集的完整示例。

```py
# example of gaussian naive bayes
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# define the model
model = GaussianNB()
# fit the model
model.fit(X, y)
# select a single sample
Xsample, ysample = [X[0]], y[0]
# make a probabilistic prediction
yhat_prob = model.predict_proba(Xsample)
print('Predicted Probabilities: ', yhat_prob)
# make a classification prediction
yhat_class = model.predict(Xsample)
print('Predicted Class: ', yhat_class)
print('Truth: y=%d' % ysample)
```

运行该示例使模型适合训练数据集，然后对我们在前面示例中使用的第一个示例进行预测。

在这种情况下，该示例属于 *y=0* 的概率为 1.0 或确定性。 *y=1* 的概率是一个非常小的接近 0.0 的值。

最后，直接预测类标签，再次匹配示例的基本事实。

```py
Predicted Probabilities:  [[1.00000000e+00 5.52387327e-30]]
Predicted Class:  [0]
Truth: y=0
```

## 使用朴素贝叶斯的 5 个技巧

本节列出了使用朴素贝叶斯模型时的一些实用技巧。

### 1.对复杂分布使用 KDE

如果变量的概率分布复杂或未知，最好使用核密度估计器或 KDE 直接从数据样本中近似分布。

高斯 KDE 就是一个很好的例子。

### 2.随着变量依赖性的增加，表现下降

根据定义，朴素贝叶斯假设输入变量相互独立。

这在大多数时候都很有效，即使一些或大部分变量实际上是依赖的。然而，输入变量越依赖，算法的表现就越差。

### 3.用对数避免数字下溢

一个类标签的一个例子的独立条件概率的计算包括将多个概率相乘，一个概率用于该类，一个概率用于每个输入变量。因此，许多小数字相乘在数字上可能会变得不稳定，尤其是当输入变量的数量增加时。

为了克服这个问题，通常将计算从概率的乘积改为对数概率的和。例如:

*   P(yi | x1，x2，…，xn)= log(P(x1 | y1))+log(P(x2 | y1))+…log(P(xn | y1))+log(P(yi))

计算概率的自然对数具有产生更大(负)数字的效果，并且将这些数字相加将意味着更大的概率将更接近于零。仍然可以比较结果值并将其最大化，以给出最可能的类别标签。

当概率相乘时，这通常被称为对数技巧。

### 4.更新概率分布

随着新数据的出现，使用新数据和旧数据来更新每个变量概率分布的参数估计值会变得相对简单。

这允许模型容易地利用新数据或随着时间变化的数据分布。

### 5.用作生成模型

概率分布将总结每个类别标签的每个输入变量值的条件概率。

除了在分类模型中使用之外，这些概率分布可能更普遍地有用。

例如，可以对准备好的概率分布进行随机采样，以创建新的可信数据实例。假设的条件独立性假设可能意味着，基于数据集中输入变量之间实际存在的相互依赖程度，这些示例或多或少是可信的。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [机器学习朴素贝叶斯教程](https://machinelearningmastery.com/naive-bayes-tutorial-for-machine-learning/)
*   [机器学习的朴素贝叶斯](https://machinelearningmastery.com/naive-bayes-for-machine-learning/)
*   [更好的朴素贝叶斯:从朴素贝叶斯算法中获得最大收益的 12 个技巧](https://machinelearningmastery.com/better-naive-bayes/)

### 书

*   [机器学习](https://amzn.to/2jWd51p)，1997。
*   [机器学习:概率视角](https://amzn.to/2xKSTCP)，2012。
*   [模式识别与机器学习](https://amzn.to/2JwHE7I)，2006。
*   [数据挖掘:实用机器学习工具与技术](https://amzn.to/2lnW5S7)，第 4 版，2016。

### 应用程序接口

*   [sklearn . dataset . make _ blobs API](http://Sklearn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)。
*   [scipy . stat .规范 API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html) 。
*   [朴素贝叶斯，sci kit-学习文档](https://Sklearn.org/stable/modules/naive_bayes.html)。
*   [硬化。幼稚 _bayes。高斯乙 API〔t1〕](https://Sklearn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)

### 文章

*   [贝叶斯定理，维基百科](https://en.wikipedia.org/wiki/Bayes%27_theorem)。
*   [朴素贝叶斯分类器，维基百科](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)。
*   [最大后验估计，维基百科](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation)。

## 摘要

在本教程中，您发现了用于分类预测建模的朴素贝叶斯算法。

具体来说，您了解到:

*   如何将分类预测建模框架化为条件概率模型？
*   如何利用贝叶斯定理求解分类的条件概率模型？
*   如何实现简化的贝叶斯分类定理，称为朴素贝叶斯算法。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。