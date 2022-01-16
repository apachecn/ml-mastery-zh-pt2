# Python 集成机器学习（7 天迷你课程）

> 原文：<https://machinelearningmastery.com/ensemble-machine-learning-with-python-7-day-mini-course/>

#### Python 速成班的集成学习算法。
用 Python 在 7 天内掌握集成学习。

集成学习指的是结合两个或更多模型的预测的机器学习模型。

集成是一种先进的机器学习方法，通常在预测的能力和技巧比使用简单易懂的模型更重要时使用。因此，它们经常被机器学习竞赛的顶尖和获奖参与者使用，如[百万美元网飞奖](https://en.wikipedia.org/wiki/Netflix_Prize)和[卡格尔竞赛](https://www.kaggle.com/)。

像 Sklearn Python 这样的现代机器学习库提供了一套高级集成学习方法，这些方法易于配置和正确使用，不会出现数据泄漏，这是使用集成算法时的一个常见问题。

在这个速成课程中，您将发现如何在七天内开始并自信地将集成学习算法带到您的 Python 预测建模项目中。

这是一个又大又重要的岗位。你可能想把它做成书签。

**用我的新书[Python 集成学习算法](https://machinelearningmastery.com/ensemble-learning-algorithms-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Ensemble Machine Learning With Python (7-Day Mini-Course)](img/0a2ff3688115acd8523ad9176ea62d79.png)

Python 集成机器学习(7 天迷你课程)
图片由[和](https://www.flickr.com/photos/anoldent/2260091725/)提供，保留部分权利。

## 这个速成班是给谁的？

在我们开始之前，让我们确保你在正确的地方。

本课程面向可能了解一些应用机器学习的开发人员。也许你知道如何用流行的工具来从头到尾解决一个预测建模问题，或者至少是大部分的主要步骤。

本课程中的课程假设了您的一些情况，例如:

*   你对编程的基本 Python 很熟悉。
*   您可能知道一些用于数组操作的基本 NumPy。
*   你可能知道一些基本的 sci kit-学习建模。

你不需要:

*   数学天才！
*   机器学习专家！

这门速成课程将把你从一个懂一点机器学习的开发人员带到一个能在预测建模项目中有效和胜任地应用集成学习算法的开发人员。

注意:本速成课程假设您有一个至少安装了 NumPy 的工作 Python 3 SciPy 环境。如果您需要环境方面的帮助，可以遵循这里的逐步教程:

*   [如何用 Anaconda](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/) 设置机器学习的 Python 环境

## 速成班概述

这门速成课分为七节课。

您可以每天完成一节课(推荐)或一天内完成所有课程(硬核)。这真的取决于你有多少时间和你的热情程度。

下面是用 Python 进行数据准备的七个经验教训:

*   **第 01 课**:什么是集成学习？
*   **第 02 课**:装袋集成
*   **第 03 课**:随机森林集成
*   **第 04 课** : AdaBoost 集成
*   **第 05 课**:梯度提升集成
*   **第 06 课**:投票团
*   **第 07 课**:堆叠集成

每节课可能需要你 60 秒或 30 分钟。慢慢来，按照自己的节奏完成课程。提问，甚至在下面的评论中发布结果。

这些课程可能期望你去发现如何做事。我会给你一些提示，但是每节课的部分要点是迫使你学习去哪里寻找关于 Python 中的算法和最佳工具的帮助。(**提示** : *我有这个博客所有的答案；使用搜索框*。)

**在评论**中发布你的结果；我会为你加油的！

坚持住。不要放弃。

## 第一课:什么是集成学习？

在本课中，您将发现什么是集成学习，以及为什么它很重要。

应用机器学习通常涉及在数据集上拟合和评估模型。

鉴于我们无法事先知道哪个模型在数据集上表现最佳，这可能需要大量的反复试验，直到我们找到一个表现良好或最适合我们项目的模型。

另一种方法是准备多个不同的模型，然后组合它们的预测。

这被称为集成机器学习模型，或简称为集成，而寻找表现良好的集成模型的过程被称为“*集成学习*”

尽管实现这一点的方法几乎是无限的，但在实践中最常讨论和使用的集成学习技术可能有三类。

它们之所以受欢迎，很大程度上是因为它们易于实现，并且在广泛的预测建模问题上取得了成功。

它们是:

*   **装袋**，如装袋决策树、随机林。
*   **升压**，例如 adaboost 和梯度升压
*   **堆叠**，例如投票和使用元模型。

在单个模型上使用集合有两个主要原因，它们是相关的；它们是:

*   **可靠性**:集成可以减少预测的方差。
*   **技能**:集成可以达到比单一模式更好的表现。

这些都是机器学习项目的重要关注点，有时我们可能更喜欢模型的一个或两个属性。

### 你的任务

这节课，你必须列出集合学习的三个应用。

这些可能是著名的例子，比如机器学习竞赛，或者你在教程、书籍或研究论文中遇到的例子。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何开发和评估 bagging 集成。

## 第二课:装袋集成

在本课中，您将发现**引导聚合**，或装袋，集成。

[装袋](https://machinelearningmastery.com/bagging-ensemble-with-python/)通过创建训练数据集的样本并在每个样本上拟合决策树来工作。

训练数据集的差异导致拟合决策树的差异，进而导致这些树做出的预测的差异。然后使用简单的统计数据，如投票或平均，将全体成员做出的预测结合起来。

该方法的关键是数据集的每个样本准备训练集合成员的方式。示例(行)是从数据集中随机抽取的，尽管进行了替换。替换意味着如果选择了一行，它将返回到训练数据集中，以便在同一训练数据集中进行可能的重新选择。

这被称为[引导样本](https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/)，给这项技术命名。

装袋在 Sklearn 中通过[装袋分类器](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)和[装袋分类器](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)类提供，默认情况下，它们使用决策树作为基本模型，您可以通过“*n _ estimates*”参数指定要创建的树的数量。

下面列出了评估装袋集合进行分类的完整示例。

```py
# example of evaluating a bagging ensemble for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
# create the synthetic classification dataset
X, y = make_classification(random_state=1)
# configure the ensemble model
model = BaggingClassifier(n_estimators=50)
# configure the resampling method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the ensemble on the dataset using the resampling method
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

### 你的任务

在本课中，您必须运行示例并查看评估模型的结果。

对于加分，评估在集合中使用更多决策树的效果，甚至改变使用的基础学习器。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何开发和评估随机森林集合。

## 第 03 课:随机森林集合

在本课中，您将发现随机森林集合。

[随机森林](https://machinelearningmastery.com/random-forest-ensemble-in-python/)是装袋集成的延伸。

像装袋一样，随机森林集成在训练数据集的不同引导样本上拟合决策树。

与装袋不同，随机森林还将对每个数据集的要素(列)进行采样。

具体来说，在构建每个决策树时，在数据中选择分割点。随机森林不会在选择分割点时考虑所有要素，而是将要素限制为要素的随机子集，例如，如果有 10 个要素，则为 3 个。

随机森林集成在 Sklearn 中通过[随机森林分类器](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)和[随机森林回归器](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)类提供。您可以通过“*n _ estimates*”参数指定要创建的树的数量，并通过“ *max_features* ”参数指定要在每个分割点考虑的随机选择的要素的数量，默认情况下，该参数设置为数据集中要素数量的平方根。

下面列出了评估随机森林集合进行分类的完整示例。

```py
# example of evaluating a random forest ensemble for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# create the synthetic classification dataset
X, y = make_classification(random_state=1)
# configure the ensemble model
model = RandomForestClassifier(n_estimators=50)
# configure the resampling method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the ensemble on the dataset using the resampling method
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

### 你的任务

在本课中，您必须运行示例并查看评估模型的结果。

对于加分项，评估在集合中使用更多决策树或调整每个分割点要考虑的特征数量的效果。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何开发和评估 AdaBoost 集成。

## 第 04 课:AdaBoost 集成

在本课中，您将发现自适应增强或 AdaBoost 集成。

增强包括将模型按顺序添加到集合中，其中新模型试图纠正已经添加到集合中的先前模型所产生的错误。这样，添加的集成成员越多，集成预计产生的错误就越少，至少在数据支持的限度内，并且在过拟合训练数据集之前。

boosting 的思想最初是作为一种理论思想发展起来的，AdaBoost 算法是实现基于 boosting 的集成算法的第一种成功方法。

[AdaBoost](https://machinelearningmastery.com/adaboost-ensemble-in-python/) 的工作原理是在经过加权的训练数据集版本上拟合决策树，使得该树更多地关注先前成员出错的示例(行)，而较少关注先前模型正确的示例(行)。

AdaBoost 使用非常简单的树，在进行预测之前，对一个输入变量进行单一决策，而不是完整的决策树。这些矮树被称为决策树桩。

AdaBoost 在 Sklearn 中通过[adaboosttclassifier](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)和[adaboosttregressor](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)类提供，默认情况下，它们使用决策树(决策树桩)作为基本模型，您可以通过“*n _ estimates*”参数指定要创建的树的数量。

下面列出了评估 AdaBoost 集成进行分类的完整示例。

```py
# example of evaluating an adaboost ensemble for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
# create the synthetic classification dataset
X, y = make_classification(random_state=1)
# configure the ensemble model
model = AdaBoostClassifier(n_estimators=50)
# configure the resampling method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the ensemble on the dataset using the resampling method
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

### 你的任务

在本课中，您必须运行示例并查看评估模型的结果。

对于加分，评估在集合中使用更多决策树的效果，甚至改变使用的基础学习器(注意，它必须支持加权训练数据)。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何开发和评估梯度提升集成。

## 第五课:梯度提升集成

在本课中，您将发现梯度提升集成。

[梯度提升](https://machinelearningmastery.com/gradient-boosting-machine-ensemble-in-python/)是提升集成算法的框架，是 AdaBoost 的扩展。

它将 boosting 重新构建为统计框架下的加法模型，并允许使用任意损失函数使其更加灵活，以及损失惩罚(收缩)以减少过拟合。

梯度提升还向集成成员引入了装袋的思想，例如对训练数据集的行和列进行采样，称为随机梯度提升。

对于结构化或表格数据，这是一种非常成功的集成技术，尽管考虑到模型是按顺序添加的，拟合模型可能会很慢。已经开发了更有效的实现，例如流行的极限梯度提升(XGBoost)和光梯度提升机(LightGBM)。

梯度提升在 Sklearn 中通过[梯度提升分类器](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)和[梯度提升回归器](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)类提供，默认情况下，它们使用决策树作为基本模型。您可以通过“*n _ estimates*”参数指定要创建的树的数量，并通过默认为 0.1 的“ *learning_rate* ”参数指定控制每个树的贡献的学习率。

下面列出了评估用于分类的梯度提升集成的完整示例。

```py
# example of evaluating a gradient boosting ensemble for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
# create the synthetic classification dataset
X, y = make_classification(random_state=1)
# configure the ensemble model
model = GradientBoostingClassifier(n_estimators=50)
# configure the resampling method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the ensemble on the dataset using the resampling method
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

### 你的任务

在本课中，您必须运行示例并查看评估模型的结果。

对于加分，评估在集成中使用更多决策树的效果，或者尝试不同的学习率值。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何开发和评估投票组合。

## 第 06 课:投票团

在本课中，您将发现投票组合。

[投票集合](https://machinelearningmastery.com/voting-ensembles-with-python/)使用简单的统计数据来组合来自多个模型的预测。

通常，这包括在同一训练数据集上拟合多个不同的模型类型，然后在回归或分类投票最多的类标签的情况下计算平均预测，称为硬投票。

当通过合计预测概率并选择具有最大合计概率的标签来预测分类问题上类别标签的概率时，也可以使用投票。这被称为软投票，当集成中使用的基本模型本身支持预测类概率时，它是首选的，因为它可以产生更好的表现。

Sklearn 中的投票组合可通过 [VotingClassifier](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) 和[voting retriever](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)课程获得。基本模型的列表可以作为模型的参数提供，列表中的每个模型必须是一个带有名称和模型的元组，例如*(“lr”，logisticreduce())*。用于分类的投票类型可以通过“*投票*参数指定，并设置为“*软*或“*硬*”。

下面列出了评估投票集合进行分类的完整示例。

```py
# example of evaluating a voting ensemble for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
# create the synthetic classification dataset
X, y = make_classification(random_state=1)
# configure the models to use in the ensemble
models = [('lr', LogisticRegression()), ('nb', GaussianNB())]
# configure the ensemble model
model = VotingClassifier(models, voting='soft')
# configure the resampling method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the ensemble on the dataset using the resampling method
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

### 你的任务

在本课中，您必须运行示例并查看评估模型的结果。

对于加分，评估在集成中尝试不同类型模型的效果，甚至将投票类型从软投票改为硬投票。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何开发和评估堆叠集合。

## 第 07 课:堆叠集成

在本课中，您将发现堆叠概括或堆叠集合。

[叠加](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)包括组合多个不同类型的基础模型的预测，很像投票。

与投票的重要区别在于，另一个机器学习模型用于学习如何最好地组合基础模型的预测。这通常是一个线性模型，例如用于回归问题的线性回归或用于分类的逻辑回归，但也可以是您喜欢的任何机器学习模型。

元模型基于基础模型对样本外数据的预测进行训练。

这包括对每个基础模型使用 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)并存储所有超出范围的预测。然后在整个训练数据集上训练基本模型，元模型在不一致的预测上训练，并学习信任哪个模型、信任它们的程度以及在什么情况下。

虽然内部堆叠使用 k-fold 交叉验证来训练元模型，但是您可以以任何您喜欢的方式评估堆叠模型，例如通过 train-test 拆分或 k-fold 交叉验证。模型的评估与内部的重采样训练过程是分开的。

堆叠集成在 Sklearn 中通过[堆叠分类器](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)和[堆叠回归器](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)课程提供。基本模型的列表可以作为模型的参数提供，列表中的每个模型必须是一个带有名称和模型的元组，例如*(“lr”，logisticreduce())*。元学习器可以通过“ *final_estimator* 参数指定，重采样策略可以通过“ *cv* 参数指定，并且可以简单地设置为指示交叉验证折叠次数的整数。

下面列出了评估堆叠集合进行分类的完整示例。

```py
# example of evaluating a stacking ensemble for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# create the synthetic classification dataset
X, y = make_classification(random_state=1)
# configure the models to use in the ensemble
models = [('knn', KNeighborsClassifier()), ('tree', DecisionTreeClassifier())]
# configure the ensemble model
model = StackingClassifier(models, final_estimator=LogisticRegression(), cv=3)
# configure the resampling method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the ensemble on the dataset using the resampling method
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report ensemble performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

### 你的任务

在本课中，您必须运行示例并查看评估模型的结果。

对于加分，评估在集合中尝试不同类型的模型和不同元模型来组合预测的效果。

在下面的评论中发表你的答案。我想看看你有什么想法。

这是最后一课。

## 末日！
( *看你走了多远*

你成功了。干得好！

花一点时间，回头看看你已经走了多远。

你发现了:

*   什么是集成学习，为什么要在预测建模项目中使用它。
*   如何使用引导聚合或装袋集成。
*   如何使用随机森林集合作为装袋的延伸？
*   如何使用自适应增强或 adaboost 集成？
*   如何使用梯度提升集成？
*   如何使用投票集合组合模型的预测。
*   如何学习如何使用叠加集合组合模型的预测。

## 摘要

**你觉得迷你课程怎么样？**
你喜欢这个速成班吗？

**你有什么问题吗？有什么症结吗？**
让我知道。请在下面留言。