# 6 种 Python 降维算法

> 原文：<https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/>

最后更新于 2020 年 8 月 17 日

**降维**是一种无监督学习技术。

然而，它可以用作机器学习算法在分类和回归预测建模数据集上的数据变换预处理步骤。

有许多降维算法可供选择，没有一个最佳算法适用于所有情况。相反，探索一系列降维算法和每个算法的不同配置是一个好主意。

在本教程中，您将发现如何在 Python 中拟合和评估顶级降维算法。

完成本教程后，您将知道:

*   降维寻求保留数据中显著关系的数字输入数据的低维表示。
*   有许多不同的降维算法，没有一种最佳方法适用于所有数据集。
*   如何使用 Sklearn 机器学习库在 Python 中实现、拟合和评估顶级降维。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Dimensionality Reduction Algorithms With Python](img/4bd445bee3ef2718371f8320d84579ce.png)

用 Python 进行降维算法。新西兰，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  降维
2.  降维算法
3.  降维的例子
    1.  sci kit-学习库安装
    2.  类别数据集
    3.  主成分分析
    4.  奇异值分解
    5.  线性判别分析
    6.  Isomap 嵌入
    7.  局部线性嵌入
    8.  改进的局部线性嵌入

## 降维

降维是指减少训练数据中输入变量数量的技术。

> 当处理高维数据时，通过将数据投影到捕捉数据“本质”的低维子空间来降低维数通常是有用的。这叫做降维。

—第 11 页，[机器学习:概率视角](https://amzn.to/2ucStHi)，2012。

高维可能意味着数百、数千甚至数百万个输入变量。

更少的输入维数通常意味着机器学习模型中相应更少的参数或更简单的结构，称为[自由度](https://machinelearningmastery.com/degrees-of-freedom-in-machine-learning/)。具有太多自由度的模型可能会过度训练数据集，并且可能在新数据上表现不佳。

希望有简单的模型能够很好地概括，进而输入数据中的输入变量很少。对于输入数量和模型自由度通常密切相关的线性模型尤其如此。

降维是在建模之前对数据执行的数据准备技术。它可以在数据清理和数据缩放之后、训练预测模型之前执行。

> …降维产生了目标概念的更紧凑、更容易解释的表示，将用户的注意力集中在最相关的变量上。

—第 289 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/2tlRP9V)，2016 年第 4 版。

因此，当使用[最终模型](https://machinelearningmastery.com/train-final-machine-learning-model/)进行预测时，对训练数据执行的任何降维也必须对新数据执行，例如测试数据集、验证数据集和数据。

## 降维算法

有许多算法可以用于降维。

两大类方法是从线性代数和流形学习中得出的。

### 线性代数方法

来自线性代数领域的矩阵分解方法可以用于维度。

有关矩阵分解的更多信息，请参见教程:

*   [机器学习矩阵分解的简单介绍](https://machinelearningmastery.com/introduction-to-matrix-decompositions-for-machine-learning/)

一些比较流行的方法包括:

*   主成分分析
*   奇异值分解
*   非负矩阵分解

### 流形学习方法

流形学习方法寻求高维输入的低维投影，该投影捕捉输入数据的显著属性。

一些比较流行的方法包括:

*   Isomap 嵌入
*   局部线性嵌入
*   多维排列
*   频谱嵌入
*   t 分布随机邻居嵌入

每种算法都提供了一种不同的方法来应对在低维数据中发现自然关系的挑战。

没有最佳的降维算法，也没有简单的方法可以在不使用受控实验的情况下为您的数据找到最佳算法。

在本教程中，我们将回顾如何使用 Sklearn 库中这些流行的降维算法的每个子集。

这些示例将为您复制粘贴示例和在自己的数据上测试方法提供基础。

我们不会深入研究算法背后的理论，也不会直接比较它们。有关此主题的良好起点，请参见:

*   [分解组件中的信号，Sklearn API](https://Sklearn.org/stable/modules/decomposition.html) 。
*   [歧管学习，sci kit-学习 API](https://Sklearn.org/stable/modules/manifold.html) 。

让我们开始吧。

## 降维的例子

在本节中，我们将回顾如何在 Sklearn 中使用流行的降维算法。

这包括使用降维技术作为建模管道中的数据转换并评估数据上的模型拟合的示例。

这些示例旨在让您复制粘贴到自己的项目中，并将这些方法应用到自己的数据中。Sklearn 库中有一些算法没有演示，因为鉴于算法的性质，它们不能直接用作数据转换。

因此，我们将在每个示例中使用合成类别数据集。

### sci kit-学习库安装

首先，让我们安装库。

不要跳过这一步，因为您需要确保安装了最新版本。

您可以使用 pip Python 安装程序安装 Sklearn 库，如下所示:

```py
sudo pip install Sklearn
```

有关特定于您的平台的其他安装说明，请参见:

*   [安装 Sklearn](https://Sklearn.org/stable/install.html)

接下来，让我们确认库已安装，并且您使用的是现代版本。

运行以下脚本打印库版本号。

```py
# check Sklearn version
import sklearn
print(sklearn.__version__)
```

运行该示例时，您应该会看到以下版本号或更高版本号。

```py
0.23.0
```

### 类别数据集

我们将使用 [make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建一个测试二进制类别数据集。

数据集将有 1，000 个包含 20 个输入要素的示例，其中 10 个是信息性的，10 个是冗余的。这为每种技术提供了识别和移除冗余输入特征的机会。

伪随机数发生器的固定随机种子确保我们在每次代码运行时生成相同的合成数据集。

下面列出了创建和汇总综合类别数据集的示例。

```py
# synthetic classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=7)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例会创建数据集，并报告符合我们预期的行数和列数。

```py
(1000, 20) (1000,)
```

这是一个二分类任务，我们将在每次降维变换后评估一个[物流分类](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)模型。

模型将使用[重复分层 10 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)的金标准进行评估。将报告所有折叠和重复的平均和标准偏差分类准确率。

以下示例将原始数据集上的模型作为比较点进行评估。

```py
# evaluate logistic regression model on raw data
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=7)
# define the model
model = LogisticRegression()
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例会对包含所有 20 列的原始数据集进行逻辑回归评估，分类准确率约为 82.4%。

对该数据进行成功的降维变换应该会产生比该基线具有更好准确率的模型，尽管这可能不是所有技术都可以做到的。

注意:我们并不是试图“*求解*”这个数据集，只是提供可以作为起点的工作示例。

```py
Accuracy: 0.824 (0.034)
```

接下来，我们可以开始查看应用于该数据集的降维算法的示例。

我已经做了一些最小的尝试来调整每个方法到数据集。每个降维方法将被配置为尽可能将 20 个输入列减少到 10 个。

我们将使用[管道](https://Sklearn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)将数据转换和模型组合成一个原子单元，可以使用交叉验证程序进行评估；例如:

```py
...
# define the pipeline
steps = [('pca', PCA(n_components=10)), ('m', LogisticRegression())]
model = Pipeline(steps=steps)
```

我们开始吧。

**其中一个算法能得到更好的结果吗？**
在下面的评论里告诉我。

### 主成分分析

主成分分析可能是最流行的密集数据降维技术。

有关主成分分析如何工作的更多信息，请参见教程:

*   [如何在 Python 中从头计算主成分分析](https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/)

Sklearn 库提供了主成分分析的[主成分分析类](https://Sklearn.org/stable/modules/generated/sklearn.decomposition.PCA.html)实现，可用作降维数据转换。可以设置“ *n_components* ”参数来配置变换输出中所需的维数。

下面列出了使用主成分分析降维评估模型的完整示例。

```py
# evaluate pca with logistic regression algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=7)
# define the pipeline
steps = [('pca', PCA(n_components=10)), ('m', LogisticRegression())]
model = Pipeline(steps=steps)
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例评估具有降维和逻辑回归预测模型的建模管道。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们看不到使用主成分分析变换对模型表现有任何提升。

```py
Accuracy: 0.824 (0.034)
```

### 奇异值分解

奇异值分解(SVD)是最流行的用于[稀疏数据](https://machinelearningmastery.com/sparse-matrices-for-machine-learning/)(具有许多零值的数据)的降维技术之一。

有关 SVD 如何工作的更多信息，请参见教程:

*   [如何用 Python 从头计算奇异值分解](https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/)

Sklearn 库提供了奇异值分解的[截断的 VD 类](https://Sklearn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)实现，可用作降维数据转换。可以设置“ *n_components* ”参数来配置变换输出中所需的维数。

下面列出了使用奇异值分解降维评估模型的完整示例。

```py
# evaluate svd with logistic regression algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=7)
# define the pipeline
steps = [('svd', TruncatedSVD(n_components=10)), ('m', LogisticRegression())]
model = Pipeline(steps=steps)
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例评估具有降维和逻辑回归预测模型的建模管道。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们看不到使用奇异值分解变换对模型表现有任何提升。

```py
Accuracy: 0.824 (0.034)
```

### 线性判别分析

线性判别分析(LDA)是一种多类分类算法，可用于降维。

投影的维数限于 1 和 C-1，其中 C 是类的数目。在这种情况下，我们的数据集是一个二分类问题(两个类)，将维数限制为 1。

有关降维 LDA 的更多信息，请参见教程:

*   [Python 中降维的线性判别分析](https://machinelearningmastery.com/linear-discriminant-analysis-for-dimensionality-reduction-in-python/)

Sklearn 库提供了[线性判别分析类](https://Sklearn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)的线性判别分析实现，可用作降维数据转换。可以设置“ *n_components* ”参数来配置变换输出中所需的维数。

下面列出了使用 LDA 降维评估模型的完整示例。

```py
# evaluate lda with logistic regression algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=7)
# define the pipeline
steps = [('lda', LinearDiscriminantAnalysis(n_components=1)), ('m', LogisticRegression())]
model = Pipeline(steps=steps)
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例评估具有降维和逻辑回归预测模型的建模管道。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，与原始数据的基线拟合相比，我们可以看到表现略有提升。

```py
Accuracy: 0.825 (0.034)
```

### Isomap 嵌入

Isomap 嵌入(Isomap)创建数据集的嵌入，并尝试保留数据集中的关系。

Sklearn 库提供了 Isomap 嵌入的 [Isomap 类](https://Sklearn.org/stable/modules/generated/sklearn.manifold.Isomap.html)实现，可用作降维数据转换。可以设置“ *n_components* ”参数来配置变换输出中所需的维数。

下面列出了使用奇异值分解降维评估模型的完整示例。

```py
# evaluate isomap with logistic regression algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.manifold import Isomap
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=7)
# define the pipeline
steps = [('iso', Isomap(n_components=10)), ('m', LogisticRegression())]
model = Pipeline(steps=steps)
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例评估具有降维和逻辑回归预测模型的建模管道。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，与原始数据的基线拟合相比，我们可以看到 Isomap 数据转换的表现有所提升。

```py
Accuracy: 0.888 (0.029)
```

### 局部线性嵌入

局部线性嵌入(LLE)创建数据集的嵌入，并试图保留数据集中邻域之间的关系。

Sklearn 库提供了局部线性嵌入的[局部线性嵌入类](https://Sklearn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html)实现，可用作降维数据转换。可以设置“ *n_components* ”参数来配置变换输出中所需的维数

下面列出了使用 LLE 降维评估模型的完整示例。

```py
# evaluate lle and logistic regression for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=7)
# define the pipeline
steps = [('lle', LocallyLinearEmbedding(n_components=10)), ('m', LogisticRegression())]
model = Pipeline(steps=steps)
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例评估具有降维和逻辑回归预测模型的建模管道。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，与原始数据的基线拟合相比，我们可以看到 LLE 数据转换的表现提升。

```py
Accuracy: 0.886 (0.028)
```

### 改进的局部线性嵌入

修改的局部线性嵌入，或修改的 LLE，是局部线性嵌入的扩展，为每个邻域创建多个加权向量。

Sklearn 库提供了修改的局部线性嵌入的[局部线性嵌入类](https://Sklearn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html)实现，可用作降维数据转换。“*方法*”参数必须设置为“已修改”，并且“ *n_components* ”参数可以设置为配置转换输出中所需的维数，该维数必须小于“ *n_neighbors* ”参数。

下面列出了使用修正 LLE 降维评估模型的完整示例。

```py
# evaluate modified lle and logistic regression for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=7)
# define the pipeline
steps = [('lle', LocallyLinearEmbedding(n_components=5, method='modified', n_neighbors=10)), ('m', LogisticRegression())]
model = Pipeline(steps=steps)
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例评估具有降维和逻辑回归预测模型的建模管道。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，与原始数据的基线拟合相比，我们可以看到修改后的 LLE 数据转换的表现有所提升。

```py
Accuracy: 0.846 (0.036)
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [机器学习降维介绍](https://machinelearningmastery.com/dimensionality-reduction-for-machine-learning/)
*   [Python 中降维的主成分分析](https://machinelearningmastery.com/principal-components-analysis-for-dimensionality-reduction-in-python/)
*   [Python 中降维的奇异值分解](https://machinelearningmastery.com/singular-value-decomposition-for-dimensionality-reduction-in-python/)
*   [Python 中降维的线性判别分析](https://machinelearningmastery.com/linear-discriminant-analysis-for-dimensionality-reduction-in-python/)

### 蜜蜂

*   [分解组件中的信号，Sklearn API](https://Sklearn.org/stable/modules/decomposition.html) 。
*   [歧管学习，sci kit-学习 API](https://Sklearn.org/stable/modules/manifold.html) 。

## 摘要

在本教程中，您发现了如何在 Python 中拟合和评估顶级降维算法。

具体来说，您了解到:

*   降维寻求保留数据中显著关系的数字输入数据的低维表示。
*   有许多不同的降维算法，没有一种最佳方法适用于所有数据集。
*   如何使用 Sklearn 机器学习库在 Python 中实现、拟合和评估顶级降维。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。