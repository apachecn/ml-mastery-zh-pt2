# 使用 AIC、BIC 和 MDL 的概率模型选择

> 原文：<https://machinelearningmastery.com/probabilistic-model-selection-measures/>

最后更新于 2020 年 8 月 28 日

模型选择是从一组候选模型中选择一个的问题。

通常选择在等待测试数据集上表现最好的模型，或者使用重采样技术来估计模型表现，例如 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)。

模型选择的另一种方法涉及使用概率统计测量，该测量试图量化训练数据集上的模型表现和模型的复杂性。示例包括阿卡克和贝叶斯信息标准以及最小描述长度。

这些信息标准统计的好处是，它们不需要等待测试集，尽管一个限制是，它们没有考虑模型的不确定性，并可能最终选择过于简单的模型。

在这篇文章中，你将发现机器学习模型选择的概率统计。

看完这篇文章，你会知道:

*   模型选择是从一组候选模型中选择一个的挑战。
*   Akaike 和 Bayesian Information Criterion 是基于模型的对数似然性和复杂性对模型进行评分的两种方法。
*   最小描述长度提供了另一种信息论的评分方法，可以证明等同于 BIC。

**用我的新书[机器学习概率](https://machinelearningmastery.com/probability-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Probabilistic Model Selection Measures AIC, BIC, and MDL](img/55215c0c8e43b6024b437b2ef20c0491.png)

概率模型选择测量 AIC、BIC 和 MDL
图片由 [Guilhem Vellut](https://www.flickr.com/photos/o_0/43121965672/) 提供，保留部分权利。

## 概观

本教程分为五个部分；它们是:

1.  模型选择的挑战
2.  概率模型选择
3.  阿卡克信息标准
4.  贝叶斯信息准则
5.  最小描述长度

## 模型选择的挑战

模型选择是在给定数据集上拟合多个模型并选择其中一个模型的过程。

> 模型选择:评估不同模型的表现，以选择最佳模型。

—第 222 页，[统计学习的要素](https://amzn.to/2YVqu8s)，2016。

这可以应用于无监督学习，例如选择聚类模型，或者有监督学习，例如为回归或分类任务选择预测模型。它也可能是建模的子任务，例如给定模型的特征选择。

有许多常见的方法可以用于模型选择。例如，在监督学习的情况下，三种最常见的方法是:

*   训练、验证和测试数据集。
*   重采样方法。
*   概率统计。

模型选择的最简单可靠的方法包括在训练集上拟合候选模型，在验证数据集中调整它们，并根据所选的度量(如准确性或误差)选择在测试数据集中表现最佳的模型。这种方法的一个问题是它需要大量的数据。

重采样技术试图实现与模型选择的训练/val/测试方法相同的效果，尽管使用的数据集很小。一个例子是 K 折交叉验证，其中一个训练集被分成许多训练/测试对，并对每个训练/测试对拟合和评估一个模型。对每个模型重复这一过程，并选择 k 倍平均得分最好的模型。这种方法和以前的方法的一个问题是只评估模型表现，而不考虑模型的复杂性。

模型选择的第三种方法试图将模型的复杂性和模型的表现结合成一个分数，然后选择最小化或最大化分数的模型。

我们可以将这种方法称为统计或概率模型选择，因为评分方法使用概率框架。

## 概率模型选择

概率模型选择(或“信息标准”)为候选模型之间的评分和选择提供了一种分析技术。

根据模型在训练数据集上的表现和模型的复杂性对模型进行评分。

*   **车型表现**。候选模型在训练数据集上的表现如何。
*   **模型复杂性**。训练后的候选模型有多复杂。

模型表现可以使用概率框架来评估，例如最大似然估计框架下的对数似然。模型复杂性可以被评估为模型中的自由度或参数的数量。

> 历史上已经提出了各种“信息标准”，试图通过增加惩罚项来纠正最大似然偏差，以补偿更复杂模型的过拟合。

—第 33 页，[模式识别与机器学习](https://amzn.to/2JwHE7I)，2006。

概率模型选择方法的一个好处是不需要测试数据集，这意味着所有的数据都可以用来拟合模型，并且将用于领域预测的最终模型可以直接评分。

概率模型选择方法的一个局限性是，不能在一系列不同类型的模型中计算相同的一般统计量。相反，必须为每个模型仔细推导度量。

> 应该注意的是，AIC 统计是为模型之间的预先计划的比较而设计的(与自动搜索期间许多模型的比较相反)。

—第 493 页，[应用预测建模](https://amzn.to/2Helnu5)，2013 年。

这些选择方法的另一个限制是它们没有考虑模型的不确定性。

> 然而，这种标准没有考虑模型参数的不确定性，实际上它们倾向于支持过于简单的模型。

—第 33 页，[模式识别与机器学习](https://amzn.to/2JwHE7I)，2006。

有三种统计方法来估计给定模型与数据集的匹配程度以及模型的复杂程度。每一个都可以被显示为彼此相等或成比例，尽管每一个都来自不同的框架或研究领域。

它们是:

*   阿卡克信息标准(AIC)。源于常客概率。
*   贝叶斯信息标准(BIC)。源自贝叶斯概率。
*   最小描述长度。源自信息论。

可以使用模型和数据的对数似然性来计算每个统计量。对数似然来自最大似然估计，这是一种根据训练数据集寻找或优化模型参数的技术。

在[最大似然估计](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)中，我们希望在给定特定概率分布及其参数(*θ*)的情况下，最大化观测数据的条件概率( *X* )，正式表述为:

*   p(X；θ)

其中 *X* 实际上是从问题域 1 到 n 的所有观测值的联合概率分布。

*   P(x1，x2，x3，…，xn；θ)

给定分布参数，联合概率分布可以重申为观察每个例子的条件概率的乘积。将许多小概率相乘可能不稳定；因此，通常将这个问题重述为自然对数条件概率的总和。

*   对数和(Xi；θ))

鉴于对数在似然函数中的频繁使用，它通常被称为对数似然函数。

常见预测建模问题的对数似然函数包括回归(如线性回归)的均方误差和二分类(如逻辑回归)的对数损失(二元交叉熵)。

在接下来的部分中，我们将仔细研究三个统计数据:AIC、BIC 和 MDL。

## 阿卡克信息标准

[阿卡克信息标准](https://en.wikipedia.org/wiki/Akaike_information_criterion)，简称 AIC，是一种评分和选择模型的方法。

它以该方法的开发者 [Hirotugu Akaike](https://en.wikipedia.org/wiki/Hirotugu_Akaike) 命名，并且可能被证明具有信息论基础和基于频率者的推断。

> 这是从一个经常性框架中推导出来的，不能被解释为边际可能性的近似值。

—第 162 页，[机器学习:概率视角](https://amzn.to/2xKSTCP)，2012。

逻辑回归的 AIC 统计定义如下(摘自“统计学习的要素”):

*   AIC = -2/N * LL + 2 * k/N

其中 *N* 为训练数据集中的样本数， *LL* 为模型在训练数据集中的对数似然， *k* 为模型中的参数数。

如上定义的分数被最小化，例如选择具有最低 AIC 的模型。

> 要使用 AIC 进行模型选择，我们只需在考虑的模型集中选择给出最小 AIC 的模型。

—第 231 页，[统计学习的要素](https://amzn.to/2YVqu8s)，2016。

与 BIC 方法(如下)相比，AIC 统计对复杂模型的惩罚更少，这意味着它可能会更加重视训练数据集的模型表现，并反过来选择更复杂的模型。

> 我们看到 AIC 的点球比 BIC 少。这导致 AIC 选择更复杂的模型。

—第 162 页，[机器学习:概率视角](https://amzn.to/2xKSTCP)，2012。

## 贝叶斯信息准则

[贝叶斯信息准则](https://en.wikipedia.org/wiki/Bayesian_information_criterion)，简称 BIC，是一种对模型进行评分和选择的方法。

它因其来源的研究领域而得名:贝叶斯概率和推理。像 AIC 一样，它适用于最大似然估计框架下的模型。

逻辑回归的 BIC 统计量计算如下(摘自“统计学习的要素”):

*   BIC = -2 * LL + log(N) * k

其中 *log()* 的底数 e 称为自然对数， *LL* 为模型的对数似然， *N* 为训练数据集中的样本数， *k* 为模型中的参数数。

如上定义的分数被最小化，例如选择具有最低 BIC 的模型。

计算的数量与 AIC 不同，尽管可以显示与 AIC 成正比。与 AIC 不同的是，BIC 对模型的处罚更多的是因为它的复杂性，这意味着更复杂的模型得分更低(更高)，反过来，被选中的可能性也更小。

> 请注意，与 AIC […]相比，这更加不利于模型的复杂性。

—第 217 页，[模式识别与机器学习](https://amzn.to/2JwHE7I)，2006。

重要的是，贝叶斯概率框架下的 BIC 推导意味着，如果候选模型的选择包括数据集的真实模型，那么 BIC 将选择真实模型的概率随着训练数据集的大小而增加。这不能说是 AIC 的分数。

> ……给定一系列模型，包括真实模型，当样本量 N ->无穷大时，BIC 选择正确模型的概率接近 1。

—第 235 页，[统计学习的要素](https://amzn.to/2YVqu8s)，2016。

BIC 的缺点是，对于较小的、代表性较低的训练数据集，更有可能选择过于简单的模型。

## 最小描述长度

[最小描述长度](https://en.wikipedia.org/wiki/Minimum_description_length)，简称 MDL，是一种评分和选择模型的方法。

它因其起源的研究领域而得名，即信息论。

信息论关注的是噪声信道上信息的表示和传输，因此，它测量的是像熵这样的量，熵是从随机变量或概率分布中表示一个事件所需的平均位数。

从信息论的角度来看，我们可能希望传递预测(或者更准确地说，它们的概率分布)和用于生成它们的模型。预测目标变量和模型都可以用在噪声信道上传输它们所需的比特数来描述。

最小描述长度是最小位数，或者表示数据和模型所需的位数总和的最小值。

> 最小描述长度原则建议选择最小化这两个描述长度之和的假设。

—第 173 页，[机器学习](https://amzn.to/2jWd51p)，1997。

MDL 统计计算如下(取自“[机器学习](https://amzn.to/2jWd51p)”):

*   CDM = l(h)+l(d | h)

其中 *h* 是模型， *D* 是模型做出的预测， *L(h)* 是表示模型所需的位数， *L(D | h)* 是表示训练数据集中模型的预测所需的位数。

如上定义的分数被最小化，例如选择具有最低 MDL 的模型。

编码所需的位数( *D | h* )和编码所需的位数( *h* )可以计算为负对数似然；例如(摘自“[统计学习的要素](https://amzn.to/2YVqu8s)”):

*   MDL =-log(P(θ))–log(P(y | X，θ))

或者给定输入值( *X* )和模型参数(*θ*)的模型参数的负对数似然( *y* )和目标值的负对数似然。

这种最小化模型及其预测编码的愿望与[奥卡姆剃刀](https://en.wikipedia.org/wiki/Occam%27s_razor)的概念有关，奥卡姆剃刀寻求最简单(最不复杂)的解释:在这种情况下，预测目标变量的最不复杂的模型。

> MDL 原则的立场是，对于一组数据来说，最好的理论是最小化理论的大小加上指定与理论相关的异常所需的信息量…

—第 198 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/2lnW5S7)，2016 年第 4 版。

MDL 计算与 BIC 非常相似，在某些情况下可以显示为等效。

> 因此，作为对数后验概率的近似值而导出的 BIC 准则，也可视为通过最小描述长度进行(近似)模型选择的一种手段。

—第 236 页，[统计学习的要素](https://amzn.to/2YVqu8s)，2016。

## 线性回归的实例

我们可以用一个算例来具体说明 AIC 和 BIC 的计算。

在本节中，我们将使用一个测试问题并拟合一个线性回归模型，然后使用 AIC 和 BIC 度量来评估该模型。

重要的是，AIC 和 BIC 对于线性回归模型的具体函数形式先前已经被导出，使得例子相对简单。在为您自己的算法调整这些示例时，为您的模型和预测问题找到合适的计算推导或者自己考虑推导计算是很重要的。

在本例中，我们将使用[make _ returnalism()sci kit-learn 函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_regression.html)提供的测试回归问题。该问题将有两个输入变量，需要预测目标数值。

```py
...
# generate dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
# define and fit the model on all data
```

我们将在整个数据集上直接拟合一个[线性回归()模型](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)。

```py
...
# define and fit the model on all data
model = LinearRegression()
model.fit(X, y)
```

一旦拟合，我们可以报告模型中的参数数量，给定问题的定义，我们期望是三个(两个系数和一个截距)。

```py
...
# number of parameters
num_params = len(model.coef_) + 1
print('Number of parameters: %d' % (num_params))
```

线性回归模型的似然函数可以显示为与最小二乘函数相同；因此，我们可以通过均方误差度量来估计模型的最大似然。

首先，模型可用于估计训练数据集中每个示例的结果，然后[均方误差()Sklearn 函数](https://Sklearn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)可用于计算模型的均方误差。

```py
...
# predict the training set
yhat = model.predict(X)
# calculate the error
mse = mean_squared_error(y, yhat)
print('MSE: %.3f' % mse)
```

将所有这些结合在一起，下面列出了定义数据集、拟合模型以及报告模型的参数数量和最大似然估计的完整示例。

```py
# generate a test dataset and fit a linear regression model
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# generate dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
# define and fit the model on all data
model = LinearRegression()
model.fit(X, y)
# number of parameters
num_params = len(model.coef_) + 1
print('Number of parameters: %d' % (num_params))
# predict the training set
yhat = model.predict(X)
# calculate the error
mse = mean_squared_error(y, yhat)
print('MSE: %.3f' % mse)
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

如我们所料，运行该示例首先报告模型中的参数数量为 3，然后报告 MSE 约为 0.01。

```py
Number of parameters: 3
MSE: 0.010
```

接下来，我们可以修改示例来计算模型的 AIC。

跳过推导，普通最小二乘线性回归模型的 AIC 计算可计算如下(摘自“[《统计识别模型新面貌》](https://ieeexplore.ieee.org/document/1100705)”，1974 年)。):

*   AIC = n * LL + 2 * k

其中 *n* 是训练数据集中的示例数量， *LL* 是使用自然对数的模型的对数似然性(例如，均方误差的对数)， *k* 是模型中的参数数量。

下面的 *calculate_aic()* 函数实现了这一点，以 *n* 、原始均方误差( *mse* )和 *k* 为参数。

```py
# calculate aic for regression
def calculate_aic(n, mse, num_params):
	aic = n * log(mse) + 2 * num_params
	return aic
```

然后可以更新该示例，以利用该新函数并计算模型的 AIC。

下面列出了完整的示例。

```py
# calculate akaike information criterion for a linear regression model
from math import log
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# calculate aic for regression
def calculate_aic(n, mse, num_params):
	aic = n * log(mse) + 2 * num_params
	return aic

# generate dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
# define and fit the model on all data
model = LinearRegression()
model.fit(X, y)
# number of parameters
num_params = len(model.coef_) + 1
print('Number of parameters: %d' % (num_params))
# predict the training set
yhat = model.predict(X)
# calculate the error
mse = mean_squared_error(y, yhat)
print('MSE: %.3f' % mse)
# calculate the aic
aic = calculate_aic(len(y), mse, num_params)
print('AIC: %.3f' % aic)
```

运行该示例会像以前一样报告参数数量和均方误差，然后报告 AIC。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，据报道 AIC 值约为-451.616。为了选择更好的型号，该值可以最小化。

```py
Number of parameters: 3
MSE: 0.010
AIC: -451.616
```

我们也可以用 BIC 而不是 AIC 的计算来探索同样的例子。

跳过推导，普通最小二乘线性回归模型的 BIC 计算可计算如下(此处取[):](https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case)

*   BIC = n * LL + k * log(n)

其中 n 是训练数据集中的示例数量， *LL* 是使用自然对数(例如均方误差的对数)的模型的对数似然性， *k* 是模型中的参数数量， *log()* 是自然对数。

下面的 *calculate_bic()* 函数实现了这一点，以 *n* 、原始均方误差( *mse* )和 *k* 为参数。

```py
# calculate bic for regression
def calculate_bic(n, mse, num_params):
	bic = n * log(mse) + num_params * log(n)
	return bic
```

然后可以更新该示例，以利用该新函数并计算模型的 BIC。

下面列出了完整的示例。

```py
# calculate bayesian information criterion for a linear regression model
from math import log
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# calculate bic for regression
def calculate_bic(n, mse, num_params):
	bic = n * log(mse) + num_params * log(n)
	return bic

# generate dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
# define and fit the model on all data
model = LinearRegression()
model.fit(X, y)
# number of parameters
num_params = len(model.coef_) + 1
print('Number of parameters: %d' % (num_params))
# predict the training set
yhat = model.predict(X)
# calculate the error
mse = mean_squared_error(y, yhat)
print('MSE: %.3f' % mse)
# calculate the bic
bic = calculate_bic(len(y), mse, num_params)
print('BIC: %.3f' % bic)
```

运行该示例会像以前一样报告参数数量和均方误差，然后报告 BIC。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，据报道 BIC 值约为-450.020，非常接近-451.616 的 AIC 值。同样，为了选择更好的型号，该值可以最小化。

```py
Number of parameters: 3
MSE: 0.010
BIC: -450.020
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   第七章模型评估与选择，[统计学习的要素](https://amzn.to/2YVqu8s)，2016。
*   第 1.3 节模型选择，[模式识别与机器学习](https://amzn.to/2JwHE7I)，2006。
*   第 4.4.1 节模型比较和 BIC，[模式识别和机器学习](https://amzn.to/2JwHE7I)，2006。
*   第 6.6 节最小描述长度原理，[机器学习](https://amzn.to/2jWd51p)，1997。
*   5.3.2.4·BIC 对对数边际似然的近似，[机器学习:概率观点](https://amzn.to/2xKSTCP)，2012。
*   [应用预测建模](https://amzn.to/2Helnu5)，2013。
*   第 28.3 节最小描述长度(MDL)，[信息论，推理和学习算法](https://amzn.to/2ZvZlJx)，2003。
*   第 5.10 节 MDL 原理，[数据挖掘:实用机器学习工具和技术](https://amzn.to/2lnW5S7)，第 4 版，2016。

### 报纸

*   [统计识别模型的新面貌](https://ieeexplore.ieee.org/document/1100705)，1974 年。

### 应用程序接口

*   [sklearn . datasets . make _ revolution API](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_regression.html)。
*   [硬化. linear_model .线性回归 API](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) 。
*   [sklearn . metrics . mean _ squared _ error API](https://Sklearn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)。

### 文章

*   [阿卡克信息准则，维基百科](https://en.wikipedia.org/wiki/Akaike_information_criterion)。
*   [贝叶斯信息准则，维基百科](https://en.wikipedia.org/wiki/Bayesian_information_criterion)。
*   [最小描述长度，维基百科](https://en.wikipedia.org/wiki/Minimum_description_length)。

## 摘要

在这篇文章中，你发现了机器学习模型选择的概率统计。

具体来说，您了解到:

*   模型选择是从一组候选模型中选择一个的挑战。
*   Akaike 和 Bayesian Information Criterion 是基于模型的对数似然性和复杂性对模型进行评分的两种方法。
*   最小描述长度提供了另一种信息论的评分方法，可以证明等同于 BIC。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。