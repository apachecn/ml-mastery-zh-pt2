# 如何在 Python 中开发特征选择子空间集成

> 原文：<https://machinelearningmastery.com/feature-selection-subspace-ensemble-in-python/>

最后更新于 2021 年 4 月 27 日

随机子空间集成由适合训练数据集中不同随机选择的输入特征组(列)的相同模型组成。

在训练数据集中选择特征组的方法有很多，而[特征选择](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/)是专门为此目的设计的一类流行的数据准备技术。相同特征选择方法和不同特征选择方法的不同配置所选择的特征完全可以作为集成学习的基础。

在本教程中，您将发现如何使用 Python 开发**特征选择子空间集成**。

完成本教程后，您将知道:

*   特征选择为选择输入特征组提供了随机子空间的替代方案。
*   如何开发和评估由单个特征选择技术选择的特征组成的集成？
*   如何开发和评估由多种不同特征选择技术选择的特征组成的集成。

**用我的新书[Python 集成学习算法](https://machinelearningmastery.com/ensemble-learning-algorithms-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Develop a Feature Selection Subspace Ensemble in Python](img/79404f4be03343282eaa34437bd8080e.png)

如何在 Python 中开发特征选择子空间集成。新西兰，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  特征选择子空间集成
2.  单特征选择方法集成
    1.  方差分析 f 统计总体
    2.  互信息集成
    3.  递归特征选择集成
3.  组合特征选择集成
    1.  具有固定数量特征的集合
    2.  具有连续数量特征的集合

## 特征选择子空间集成

**随机子空间方法**或随机子空间集成是一种集成学习方法，它在训练数据集中随机选择的不同列组上拟合模型。

用于训练集合中每个模型的列的选择的差异导致模型及其预测的多样性。每个模型都表现良好，尽管每个模型的表现不同，犯的错误也不同。

> 训练数据通常由一组特征描述。不同的特征子集，或称为子空间，提供不同的数据视图。因此，从不同子空间训练的个体学习器通常是不同的。

—第 116 页，[集合方法](https://amzn.to/2XZzrjG)，2012。

随机子空间方法通常与决策树一起使用，然后使用简单的统计信息组合每棵树做出的预测，例如计算用于分类的模式类别标签或用于回归的平均预测。

特征选择是一种数据准备技术，它试图选择数据集中与目标变量最相关的列子集。流行的方法包括使用统计方法，如[互信息](https://machinelearningmastery.com/information-gain-and-mutual-information/)，评估特征子集上的模型，并选择产生最佳表现模型的子集，称为[递归特征消除](https://machinelearningmastery.com/rfe-feature-selection-in-python/)，简称 RFE。

每种特征选择方法对于哪些特征与目标变量最相关会有不同的想法或明智的猜测。此外，可以定制特征选择方法，从 1 到数据集中的列总数中选择特定数量的特征，这是一个可以作为模型选择的一部分进行调整的超参数。

每组选择的特征可以被认为是输入特征空间的子集，很像随机子空间集合，尽管是使用度量而不是随机选择的。我们可以使用由特征选择方法选择的特征作为一种集成模型。

可能有许多方法可以实现这一点，但可能有两种自然的方法包括:

*   **一种方法**:为数据集中从 1 到列数的每个特征生成一个特征子空间，在每个特征子空间上拟合一个模型，并组合它们的预测。
*   **多种方法**:使用多种不同的特征选择方法生成特征子空间，在每种方法上拟合一个模型，并组合它们的预测。

由于没有更好的名字，我们可以称之为“**特征选择子空间集成**”

我们将在本教程中探讨这个想法。

让我们定义一个测试问题作为这种探索的基础，并建立一个表现基线，看看它是否比单个模型有好处。

首先，我们可以使用 [make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建一个包含 1000 个示例和 20 个输入特征的合成二进制分类问题，其中 5 个是冗余的。

下面列出了完整的示例。

```py
# synthetic classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例将创建数据集并总结输入和输出组件的形状。

```py
(1000, 20) (1000,)
```

接下来，我们可以建立一个表现基线。我们将为数据集开发一个决策树，并使用三次重复和 10 次重复的重复分层 k-fold 交叉验证对其进行评估。

结果将报告为所有重复和折叠的分类准确度的平均值和标准偏差。

下面列出了完整的示例。

```py
# evaluate a decision tree on the classification dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
# define the random subspace ensemble model
model = DecisionTreeClassifier()
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例会报告平均和标准偏差分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到单个决策树模型实现了大约 79.4%的分类准确率。我们可以将此作为表现的基准，看看我们的功能选择集成是否能够实现更好的表现。

```py
Mean Accuracy: 0.794 (0.046)
```

接下来，让我们探索使用不同的特征选择方法作为集成的基础。

## 单特征选择方法集成

在本节中，我们将探索如何从单个特征选择方法选择的特征中创建一个集合。

对于给定的特征选择方法，我们将对不同数量的所选特征重复应用它，以创建多个特征子空间。然后，我们将在每一个模型上训练一个模型，在这个例子中是一个决策树，并组合预测。

有许多方法可以组合预测，但为了简单起见，我们将使用投票集合，该集合可以配置为使用硬或软投票进行分类，或者使用平均进行回归。为了保持例子简单，我们将集中在分类和使用硬投票，因为决策树不能预测校准概率，使得软投票不太合适。

要了解有关投票集成的更多信息，请参阅教程:

*   [如何用 Python 开发投票集成](https://machinelearningmastery.com/voting-ensembles-with-python/)

投票集成中的每个模型将是[管道](https://Sklearn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)，其中第一步是特征选择方法，被配置为选择特定数量的特征，随后是决策树分类器模型。

我们将为输入数据集中从 1 到列数的每个列数创建一个特征选择子空间。为了简单起见，这是任意选择的，您可能希望在集合中尝试不同数量的特征，例如奇数个特征，或者更复杂的方法。

因此，我们可以定义一个名为 *get_ensemble()* 的辅助函数，为给定数量的输入特征创建一个具有基于特征选择的成员的投票集成。然后，我们可以使用这个函数作为模板来探索使用不同的特征选择方法。

```py
# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# enumerate the features in the training dataset
	for i in range(1, n_features+1):
		# create the feature selection transform
		fs = ...
		# create the model
		model = DecisionTreeClassifier()
		# create the pipeline
		pipe = Pipeline([('fs',fs), ('m', model)])
		# add as a tuple to the list of models for voting
		models.append((str(i),pipe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble
```

假设我们正在使用类别数据集，我们将探索三种不同的特征选择方法:

*   方差分析 f 统计量。
*   相互信息。
*   递归特征选择。

让我们仔细看看每一个。

### 方差分析 f 统计总体

ANOVA 是“方差分析”的缩写，是一种参数统计假设检验，用于确定两个或多个数据样本(通常是三个或更多)的平均值是否来自同一分布。

f 统计或 f 检验是一类统计检验，用于计算方差值之间的比率，如两个不同样本的方差或通过统计检验解释和解释的方差，如[方差分析](https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/)。方差分析方法是一种 f 统计，这里称为[方差分析 f 检验](https://machinelearningmastery.com/feature-selection-with-numerical-input-data/)。

Sklearn 机器库提供了 [f_classif()函数](https://Sklearn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)中方差分析 F-检验的实现。该功能可用于特征选择策略，例如通过[选择最相关的特征(最大值)选择最相关的 *k* 类](https://Sklearn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)。

```py
# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# enumerate the features in the training dataset
	for i in range(1, n_features+1):
		# create the feature selection transform
		fs = SelectKBest(score_func=f_classif, k=i)
		# create the model
		model = DecisionTreeClassifier()
		# create the pipeline
		pipe = Pipeline([('fs',fs), ('m', model)])
		# add as a tuple to the list of models for voting
		models.append((str(i),pipe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble
```

将这些联系在一起，下面的例子评估了一个由模型组成的投票集合，这些模型适合于由方差分析统计选择的特征子空间。

```py
# example of an ensemble created from features selected with the anova f-statistic
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from matplotlib import pyplot

# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# enumerate the features in the training dataset
	for i in range(1, n_features+1):
		# create the feature selection transform
		fs = SelectKBest(score_func=f_classif, k=i)
		# create the model
		model = DecisionTreeClassifier()
		# create the pipeline
		pipe = Pipeline([('fs',fs), ('m', model)])
		# add as a tuple to the list of models for voting
		models.append((str(i),pipe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
# get the ensemble model
ensemble = get_ensemble(X.shape[1])
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例会报告平均和标准偏差分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到单个模型的表现有所提升，使用基于方差分析统计选择的特征的模型集合，该模型的准确率达到了约 79.4%至约 83.2%。

```py
Mean Accuracy: 0.832 (0.043)
```

接下来，让我们探索使用相互信息。

### 互信息集成

[互信息](https://machinelearningmastery.com/information-gain-and-mutual-information/)来自信息论领域的是信息增益(通常用于决策树的构建)对特征选择的应用。

计算两个变量之间的互信息，并在已知另一个变量的值的情况下，测量一个变量不确定性的减少。当考虑两个离散(分类或序数)变量的分布时，例如分类输入和分类输出数据，这很简单。然而，它可以适用于数字输入和分类输出。

Sklearn 机器学习库通过 [mutual_info_classif()函数](https://Sklearn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)为特征选择提供了数字输入和分类输出变量的互信息实现。和 *f_classif()* 一样，可以在 *SelectKBest* 特征选择策略(和其他策略)中使用。

```py
# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# enumerate the features in the training dataset
	for i in range(1, n_features+1):
		# create the feature selection transform
		fs = SelectKBest(score_func=mutual_info_classif, k=i)
		# create the model
		model = DecisionTreeClassifier()
		# create the pipeline
		pipe = Pipeline([('fs',fs), ('m', model)])
		# add as a tuple to the list of models for voting
		models.append((str(i),pipe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble
```

将这些联系在一起，下面的例子评估了一个由模型组成的投票集合，这些模型适合由互信息选择的特征子空间。

```py
# example of an ensemble created from features selected with mutual information
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from matplotlib import pyplot

# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# enumerate the features in the training dataset
	for i in range(1, n_features+1):
		# create the feature selection transform
		fs = SelectKBest(score_func=mutual_info_classif, k=i)
		# create the model
		model = DecisionTreeClassifier()
		# create the pipeline
		pipe = Pipeline([('fs',fs), ('m', model)])
		# add as a tuple to the list of models for voting
		models.append((str(i),pipe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
# get the ensemble model
ensemble = get_ensemble(X.shape[1])
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例会报告平均和标准偏差分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到表现比使用单一模型有所提升，尽管比选择的特征子空间略少，方差分析的平均准确率约为 82.7%。

```py
Mean Accuracy: 0.827 (0.048)
```

接下来，让我们探索使用 RFE 选择的子空间。

### 递归特征选择集成

递归特征消除(简称 RFE)的工作原理是，从训练数据集中的所有特征开始搜索特征子集，并成功移除特征，直到保留所需数量。

这是通过拟合模型核心中使用的给定机器学习算法、按重要性排列特征、丢弃最不重要的特征以及重新拟合模型来实现的。重复此过程，直到保留指定数量的特征。

有关 RFE 的更多信息，请参见教程:

*   [Python 中特征选择的递归特征消除(RFE)](https://machinelearningmastery.com/rfe-feature-selection-in-python/)

RFE 方法可通过 Sklearn 中的 [RFE 类](https://Sklearn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)获得，并可直接用于特征选择。不需要和*选择测试*类结合。

```py
# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# enumerate the features in the training dataset
	for i in range(1, n_features+1):
		# create the feature selection transform
		fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
		# create the model
		model = DecisionTreeClassifier()
		# create the pipeline
		pipe = Pipeline([('fs',fs), ('m', model)])
		# add as a tuple to the list of models for voting
		models.append((str(i),pipe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble
```

将这些联系在一起，下面的例子评估了由适合 RFE 选择的特征子空间的模型组成的投票集合。

```py
# example of an ensemble created from features selected with RFE
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from matplotlib import pyplot

# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# enumerate the features in the training dataset
	for i in range(1, n_features+1):
		# create the feature selection transform
		fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
		# create the model
		model = DecisionTreeClassifier()
		# create the pipeline
		pipe = Pipeline([('fs',fs), ('m', model)])
		# add as a tuple to the list of models for voting
		models.append((str(i),pipe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
# get the ensemble model
ensemble = get_ensemble(X.shape[1])
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例会报告平均和标准偏差分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到平均准确率与互信息特征选择相似，得分约为 82.3%。

```py
Mean Accuracy: 0.823 (0.045)
```

这是一个好的开始，看看使用由更少成员组成的集成(例如，每隔一秒、三分之一或五分之一数量的选定特征)是否可以获得更好的结果可能会很有趣。

接下来，让我们看看是否可以通过组合适合不同特征选择方法选择的特征子空间的模型来提高结果。

## 组合特征选择集成

在前一节中，我们看到，通过使用单个特征选择方法作为数据集集成预测的基础，我们可以提升单个模型的表现。

我们预计，许多集成成员之间的预测是相关的。这可以通过使用不同数量的所选输入特征作为集合的基础来解决，而不是从 1 到列数的连续数量的特征。

引入多样性的另一种方法是使用不同的特征选择方法来选择特征子空间。

我们将探讨这种方法的两个版本。对于第一种方法，我们将从每个方法中选择相同数量的特征，对于第二种方法，我们将从 1 到多个方法的列数中选择连续数量的特征。

### 具有固定数量特征的集合

在本节中，我们将首次尝试使用由多种特征选择技术选择的特征来设计一个集合。

我们将从数据集中选择任意数量的特征，然后使用三种特征选择方法中的每一种来选择特征子空间，拟合每一种的模型，并将它们用作投票集成的基础。

下面的 *get_ensemble()* 函数实现了这一点，将每个方法要选择的指定数量的特征作为参数。希望通过每种方法选择的特征足够不同，足够熟练，以产生有效的集成。

```py
# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# anova
	fs = SelectKBest(score_func=f_classif, k=n_features)
	anova = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('anova', anova))
	# mutual information
	fs = SelectKBest(score_func=mutual_info_classif, k=n_features)
	mutinfo = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('mutinfo', mutinfo))
	# rfe
	fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=n_features)
	rfe = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('rfe', rfe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble
```

将这些联系在一起，下面的例子评估了使用不同特征选择方法选择的固定数量的特征的集合。

```py
# ensemble of a fixed number features selected by different feature selection methods
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from matplotlib import pyplot

# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# anova
	fs = SelectKBest(score_func=f_classif, k=n_features)
	anova = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('anova', anova))
	# mutual information
	fs = SelectKBest(score_func=mutual_info_classif, k=n_features)
	mutinfo = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('mutinfo', mutinfo))
	# rfe
	fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=n_features)
	rfe = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('rfe', rfe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# get the ensemble model
ensemble = get_ensemble(15)
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例会报告平均和标准偏差分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到表现相对于上一节中考虑的技术有适度的提升，导致平均分类准确率约为 83.9%。

```py
Mean Accuracy: 0.839 (0.044)
```

更公平的比较可能是将这个结果与构成整体的每个单独模型进行比较。

更新后的示例恰好执行了这种比较。

```py
# comparison of ensemble of a fixed number features to single models fit on each set of features
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from matplotlib import pyplot

# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models, names = list(), list()
	# anova
	fs = SelectKBest(score_func=f_classif, k=n_features)
	anova = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('anova', anova))
	names.append('anova')
	# mutual information
	fs = SelectKBest(score_func=mutual_info_classif, k=n_features)
	mutinfo = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('mutinfo', mutinfo))
	names.append('mutinfo')
	# rfe
	fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=n_features)
	rfe = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('rfe', rfe))
	names.append('rfe')
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	names.append('ensemble')
	return names, [anova, mutinfo, rfe, ensemble]

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# get the ensemble model
names, models = get_ensemble(15)
# evaluate each model
results = list()
for model,name in zip(models,names):
	# define the evaluation method
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model on the dataset
	n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# report performance
	print('>%s: %.3f (%.3f)' % (name, mean(n_scores), std(n_scores)))
	results.append(n_scores)
# plot the results for comparison
pyplot.boxplot(results, labels=names)
pyplot.show()
```

运行该示例会报告每个单个模型在选定特征上的平均表现，并以组合所有三个模型的集合的表现结束。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，结果表明，如我们所希望的那样，适合所选特征的模型集合比集合中的任何单个模型表现得更好。

```py
>anova: 0.811 (0.048)
>mutinfo: 0.807 (0.041)
>rfe: 0.825 (0.043)
>ensemble: 0.837 (0.040)
```

创建一个图形来显示每组结果的方框图和触须图，允许直接比较分布准确度分数。

我们可以看到，集合的分布倾斜得更高，并且具有更大的中值分类准确率(橙色线)，直观地证实了这一发现。

![Box and Whisker Plots of Accuracy of Singles Model Fit On Selected Features vs. Ensemble](img/a3fb8a2f288b680fc9338967ee808582.png)

单个模型拟合选定特征与整体的准确度的方框图和触须图

接下来，让我们探索为每个特征选择方法添加多个成员。

### 具有连续数量特征的集合

我们可以把上一节的实验和上面的实验结合起来。

具体来说，我们可以使用每个特征选择方法选择多个特征子空间，在每个子空间上拟合一个模型，并将所有模型添加到单个集成中。

在这种情况下，我们将选择子空间，就像我们在上一节中从 1 到数据集中的列数一样，尽管在这种情况下，用每个特征选择方法重复这个过程。

```py
# get a voting ensemble of models
def get_ensemble(n_features_start, n_features_end):
	# define the base models
	models = list()
	for i in range(n_features_start, n_features_end+1):
		# anova
		fs = SelectKBest(score_func=f_classif, k=i)
		anova = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
		models.append(('anova'+str(i), anova))
		# mutual information
		fs = SelectKBest(score_func=mutual_info_classif, k=i)
		mutinfo = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
		models.append(('mutinfo'+str(i), mutinfo))
		# rfe
		fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
		rfe = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
		models.append(('rfe'+str(i), rfe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble
```

希望通过特征选择方法选择的特征的多样性导致集成表现的进一步提升。

将这些联系在一起，完整的示例如下所示。

```py
# ensemble of many subsets of features selected by multiple feature selection methods
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from matplotlib import pyplot

# get a voting ensemble of models
def get_ensemble(n_features_start, n_features_end):
	# define the base models
	models = list()
	for i in range(n_features_start, n_features_end+1):
		# anova
		fs = SelectKBest(score_func=f_classif, k=i)
		anova = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
		models.append(('anova'+str(i), anova))
		# mutual information
		fs = SelectKBest(score_func=mutual_info_classif, k=i)
		mutinfo = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
		models.append(('mutinfo'+str(i), mutinfo))
		# rfe
		fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
		rfe = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
		models.append(('rfe'+str(i), rfe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# get the ensemble model
ensemble = get_ensemble(1, 20)
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例会报告集合的均值和标准差分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到表现的进一步提升，正如我们所希望的那样，组合集成导致平均分类准确率约为 86.0%。

```py
Mean Accuracy: 0.860 (0.036)
```

使用特征选择来选择输入特征的子空间可以为选择随机子空间提供一个有趣的替代或者补充。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [如何用 Python 开发投票集成](https://machinelearningmastery.com/voting-ensembles-with-python/)
*   [Python 中特征选择的递归特征消除(RFE)](https://machinelearningmastery.com/rfe-feature-selection-in-python/)
*   [如何用数值输入数据进行特征选择](https://machinelearningmastery.com/feature-selection-with-numerical-input-data/)

### 书

*   [使用集成方法的模式分类](https://amzn.to/2zxc0F7)，2010。
*   [集成方法](https://amzn.to/2XZzrjG)，2012。
*   [集成机器学习](https://amzn.to/2C7syo5)，2012。

### 蜜蜂

*   [硬化. feature _ selection . f _ classic API](https://Sklearn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)。
*   [sklearn . feature _ selection . mutual _ info _ class if API](https://Sklearn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)。
*   [硬化. feature_selection。SelectKBest API](https://Sklearn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) 。
*   [sklearn.feature_selection。RFE 原料药](https://Sklearn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)。

### 文章

*   [随机子空间法，维基百科](https://en.wikipedia.org/wiki/Random_subspace_method)。

## 摘要

在本教程中，您发现了如何使用 Python 开发特征选择子空间集成。

具体来说，您了解到:

*   特征选择为选择输入特征组提供了随机子空间的替代方案。
*   如何开发和评估由单个特征选择技术选择的特征组成的集成？
*   如何开发和评估由多种不同特征选择技术选择的特征组成的集成。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。