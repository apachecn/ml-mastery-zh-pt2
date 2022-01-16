# 使用 Python 的混合集成机器学习

> 原文：<https://machinelearningmastery.com/blending-ensemble-machine-learning-with-python/>

最后更新于 2021 年 4 月 27 日

**融合**是一种集成机器学习算法。

这是一个口语化的名称**堆叠泛化**或堆叠集成，其中不是将元模型拟合到基础模型做出的折外预测上，而是将其拟合到保持数据集上做出的预测上。

混合被用来描述堆叠模型，在 100 万美元的网飞机器学习竞赛中，竞争对手将数百个预测模型组合在一起，因此，在竞争激烈的机器学习圈子(如 Kaggle 社区)中，混合仍然是一种流行的堆叠技术和名称。

在本教程中，您将发现如何在 python 中开发和评估混合集成。

完成本教程后，您将知道:

*   混合集成是一种堆叠类型，其中元模型使用保持验证数据集上的预测来拟合，而不是折叠预测。
*   如何开发混合集成，包括用于训练模型和对新数据进行预测的功能。
*   如何评估分类和回归预测建模问题的混合集成？

**用我的新书[Python 集成学习算法](https://machinelearningmastery.com/ensemble-learning-algorithms-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Blending Ensemble Machine Learning With Python](img/26fe0007cf21fbac2106e6e672337841.png)

将集成机器学习与 Python 融合
图片由 [Nathalie](https://www.flickr.com/photos/nathalie-photos/37421605474/) 提供，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  混合集成
2.  开发混合集成
3.  用于分类的混合集成
4.  回归的混合集成

## 混合集成

混合是一种集成机器学习技术，它使用机器学习模型来学习如何最好地组合来自多个贡献集成成员模型的预测。

因此，混合与[堆叠概括](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)相同，称为堆叠，广义上来说。通常，在同一篇论文或模型描述中，混合和堆叠可以互换使用。

> 许多机器学习实践者已经成功地使用堆叠和相关技术来提高预测准确率，使其超过任何单个模型获得的水平。在某些情况下，堆叠也被称为混合，我们将在这里互换使用这些术语。

——[特征加权线性叠加](https://arxiv.org/abs/0911.0460)，2009。

堆叠模型的体系结构包括两个或多个基础模型，通常称为 0 级模型，以及一个组合基础模型预测的元模型，称为 1 级模型。元模型基于基础模型对样本外数据的预测进行训练。

*   **0 级模型** ( *基础模型*):模型适合训练数据，并且其预测被编译。
*   **一级模型** ( *元模型*):学习如何最好地组合基础模型预测的模型。

然而，混合对于如何构建堆叠集成模型有特定的含义。

混合可能建议开发一个堆叠集成，其中基础模型是任何类型的机器学习模型，元模型是一个线性模型，它将基础模型的预测“混合”*。*

 *例如，当预测数值时的线性回归模型或当预测类别标签时的逻辑回归模型将计算由基础模型做出的预测的加权和，并将被认为是预测的混合。

*   **混合集成**:使用线性模型，如线性回归或逻辑回归，作为堆叠集成中的元模型。

2009 年网飞奖期间，混合是堆叠集成常用的术语。该奖项涉及寻求比原生网飞算法表现更好的电影推荐预测的团队，获得 100 万美元奖金的团队获得了 10%的业绩提升。

> 我们的 RMSE=0.8643^2 解决方案是 100 多个结果的线性混合。[……]在整个方法描述中，我们强调了参与最终混合解决方案的具体预测因素。

——[BellKor 2008 年 Netflix 大奖解决方案](https://netflixprize.com/assets/ProgressPrize2008_BellKor.pdf)，2008 年。

因此，混合是一个口语术语，指的是具有堆叠型架构模型的集成学习。除了与竞争性机器学习相关的内容之外，它很少被用于教科书或学术论文。

最常见的是，混合用于描述堆叠的具体应用，其中元模型是基于基础模型在搁置验证数据集上做出的预测来训练的。在这种情况下，堆叠是为元模型保留的，元模型是在交叉验证过程中对折外预测进行训练的。

*   **混合**:堆叠式集成，其中元模型基于在保持数据集上做出的预测进行训练。
*   **叠加**:叠加型集合，其中元模型基于 k 倍交叉验证期间做出的超折叠预测进行训练。

这种区别在 Kaggle 竞争机器学习社区中很常见。

> 混合是网飞获奖者引入的一个词。它非常接近于堆叠概括，但更简单一点，信息泄露的风险也更小。[……]通过混合，而不是为列车组创建不一致的预测，您创建了一个小的保持组，比如说 10%的列车组。然后，堆垛机模型仅在该保持装置上运行。

——[kag gle 联合指南](https://mlwave.com/kaggle-ensembling-guide/)，MLWave，2015 年。

我们将使用混合的后一种定义。

接下来，让我们看看如何实现混合。

## 开发混合集成

Sklearn 库在编写本文时并不支持混合。

相反，我们可以使用 Sklearn 模型自己实现它。

首先，我们需要创建一些基础模型。对于回归或分类问题，这些可以是我们喜欢的任何模型。我们可以定义一个函数 *get_models()* ，该函数返回一个模型列表，其中每个模型被定义为一个元组，该元组有一个名称和配置的分类器或回归对象。

例如，对于分类问题，我们可以使用逻辑回归、知识网络、决策树、SVM 和朴素贝叶斯模型。

```py
# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('cart', DecisionTreeClassifier()))
	models.append(('svm', SVC(probability=True)))
	models.append(('bayes', GaussianNB()))
	return models
```

接下来，我们需要拟合混合模型。

回想一下，基础模型适合训练数据集。元模型适合于保持数据集上每个基础模型所做的预测。

首先，我们可以枚举模型列表，并在训练数据集上依次拟合每个模型。同样在这个循环中，我们可以使用拟合模型对保持(验证)数据集进行预测，并存储预测供以后使用。

```py
...
# fit all models on the training set and predict on hold out set
meta_X = list()
for name, model in models:
	# fit in training set
	model.fit(X_train, y_train)
	# predict on hold out set
	yhat = model.predict(X_val)
	# reshape predictions into a matrix with one column
	yhat = yhat.reshape(len(yhat), 1)
	# store predictions as input for blending
	meta_X.append(yhat)
```

我们现在有了“ *meta_X* ”，它代表了可以用来训练元模型的输入数据。每个列或特征代表一个基本模型的输出。

每行代表保持数据集中的一个样本。我们可以使用 [hstack()函数](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html)来确保该数据集是机器学习模型所期望的 2D numpy 数组。

```py
...
# create 2d array from predictions, each set is an input feature
meta_X = hstack(meta_X)
```

我们现在可以训练我们的元模型了。这可以是我们喜欢的任何机器学习模型，例如用于分类的逻辑回归。

```py
...
# define blending model
blender = LogisticRegression()
# fit on predictions from base models
blender.fit(meta_X, y_val)
```

我们可以将所有这些结合到一个名为 *fit_ensemble()* 的函数中，该函数使用训练数据集和保持验证数据集来训练混合模型。

```py
# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, y_train, y_val):
	# fit all models on the training set and predict on hold out set
	meta_X = list()
	for name, model in models:
		# fit in training set
		model.fit(X_train, y_train)
		# predict on hold out set
		yhat = model.predict(X_val)
		# reshape predictions into a matrix with one column
		yhat = yhat.reshape(len(yhat), 1)
		# store predictions as input for blending
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# define blending model
	blender = LogisticRegression()
	# fit on predictions from base models
	blender.fit(meta_X, y_val)
	return blender
```

下一步是使用混合集成对新数据进行预测。

这是一个分两步走的过程。第一步是使用每个基本模型进行预测。然后，这些预测被收集在一起，并用作混合模型的输入，以做出最终预测。

我们可以使用与训练模型时相同的循环结构。也就是说，我们可以将每个基础模型的预测收集到一个训练数据集中，将预测堆叠在一起，并使用这个元级数据集在 blender 模型上调用 *predict()* 。

下面的 *predict_ensemble()* 函数实现了这一点。给定拟合基础模型、拟合混合器集合和数据集(如测试数据集或新数据)的列表，它将返回数据集的一组预测。

```py
# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for name, model in models:
		# predict with base model
		yhat = model.predict(X_test)
		# reshape predictions into a matrix with one column
		yhat = yhat.reshape(len(yhat), 1)
		# store prediction
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict(meta_X)
```

我们现在已经拥有了为分类或回归预测建模问题实现混合集成所需的所有元素

## 用于分类的混合集成

在这一节中，我们将研究混合在分类问题中的应用。

首先，我们可以使用 [make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建一个包含 10，000 个示例和 20 个输入特征的合成二进制分类问题。

下面列出了完整的示例。

```py
# test classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例将创建数据集并总结输入和输出组件的形状。

```py
(10000, 20) (10000,)
```

接下来，我们需要将数据集分成训练集和测试集，然后将训练集分成用于训练基本模型的子集和用于训练元模型的子集。

在这种情况下，我们将对训练集和测试集使用 50-50 的分割，然后对训练集和验证集使用 67-33 的分割。

```py
...
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# split training set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# summarize data split
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))
```

然后，我们可以使用上一节中的 *get_models()* 函数来创建集成中使用的分类模型。

然后可以调用*拟合集合()*函数来拟合训练和验证数据集上的混合集合，并且可以使用*预测集合()*函数来对保持数据集进行预测。

```py
...
# create the base models
models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make predictions on test set
yhat = predict_ensemble(models, blender, X_test)
```

最后，我们可以通过在测试数据集上报告分类准确率来评估混合模型的表现。

```py
...
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Blending Accuracy: %.3f' % score)
```

将所有这些联系在一起，下面列出了在合成二分类问题上评估混合集成的完整示例。

```py
# blending ensemble for classification using hard voting
from numpy import hstack
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('cart', DecisionTreeClassifier()))
	models.append(('svm', SVC()))
	models.append(('bayes', GaussianNB()))
	return models

# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, y_train, y_val):
	# fit all models on the training set and predict on hold out set
	meta_X = list()
	for name, model in models:
		# fit in training set
		model.fit(X_train, y_train)
		# predict on hold out set
		yhat = model.predict(X_val)
		# reshape predictions into a matrix with one column
		yhat = yhat.reshape(len(yhat), 1)
		# store predictions as input for blending
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# define blending model
	blender = LogisticRegression()
	# fit on predictions from base models
	blender.fit(meta_X, y_val)
	return blender

# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for name, model in models:
		# predict with base model
		yhat = model.predict(X_test)
		# reshape predictions into a matrix with one column
		yhat = yhat.reshape(len(yhat), 1)
		# store prediction
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict(meta_X)

# define dataset
X, y = get_dataset()
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# split training set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# summarize data split
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))
# create the base models
models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make predictions on test set
yhat = predict_ensemble(models, blender, X_test)
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Blending Accuracy: %.3f' % (score*100))
```

运行该示例首先报告训练、验证和测试数据集的形状，然后报告测试数据集上集成的准确性。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到混合集成实现了大约 97.900%的分类准确率。

```py
Train: (3350, 20), Val: (1650, 20), Test: (5000, 20)
Blending Accuracy: 97.900
```

在前面的示例中，使用混合模型组合了清晰的类标签预测。这是一种[硬投票](https://machinelearningmastery.com/voting-ensembles-with-python/)的类型。

另一种方法是让每个模型预测类概率，并使用元模型来混合概率。这是一种软投票，在某些情况下可以产生更好的表现。

首先，我们必须将模型配置为返回概率，例如 SVM 模型。

```py
# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('cart', DecisionTreeClassifier()))
	models.append(('svm', SVC(probability=True)))
	models.append(('bayes', GaussianNB()))
	return models
```

接下来，我们必须改变基本模型来预测概率，而不是简单的类标签。

这可以通过在拟合基础模型时调用*拟合 _ 集合()*函数中的*预测 _proba()* 函数来实现。

```py
...
# fit all models on the training set and predict on hold out set
meta_X = list()
for name, model in models:
	# fit in training set
	model.fit(X_train, y_train)
	# predict on hold out set
	yhat = model.predict_proba(X_val)
	# store predictions as input for blending
	meta_X.append(yhat)
```

这意味着用于训练元模型的元数据集每个分类器将有 n 列，其中 n 是预测问题中的类的数量，在我们的例子中是两个。

当使用混合模型对新数据进行预测时，我们还需要更改基础模型所做的预测。

```py
...
# make predictions with base models
meta_X = list()
for name, model in models:
	# predict with base model
	yhat = model.predict_proba(X_test)
	# store prediction
	meta_X.append(yhat)
```

将这些联系在一起，下面列出了对合成二进制分类问题的预测类概率使用混合的完整示例。

```py
# blending ensemble for classification using soft voting
from numpy import hstack
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('cart', DecisionTreeClassifier()))
	models.append(('svm', SVC(probability=True)))
	models.append(('bayes', GaussianNB()))
	return models

# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, y_train, y_val):
	# fit all models on the training set and predict on hold out set
	meta_X = list()
	for name, model in models:
		# fit in training set
		model.fit(X_train, y_train)
		# predict on hold out set
		yhat = model.predict_proba(X_val)
		# store predictions as input for blending
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# define blending model
	blender = LogisticRegression()
	# fit on predictions from base models
	blender.fit(meta_X, y_val)
	return blender

# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for name, model in models:
		# predict with base model
		yhat = model.predict_proba(X_test)
		# store prediction
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict(meta_X)

# define dataset
X, y = get_dataset()
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# split training set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# summarize data split
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))
# create the base models
models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make predictions on test set
yhat = predict_ensemble(models, blender, X_test)
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Blending Accuracy: %.3f' % (score*100))
```

运行该示例首先报告训练、验证和测试数据集的形状，然后报告测试数据集上集成的准确性。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到混合类概率导致分类准确率提升到大约 98.240%。

```py
Train: (3350, 20), Val: (1650, 20), Test: (5000, 20)
Blending Accuracy: 98.240
```

混合集成只有在能够超越任何单个贡献模型时才有效。

我们可以通过单独评估每个基础模型来证实这一点。每个基础模型可以适合整个训练数据集(不同于混合集成)，并在测试数据集上进行评估(就像混合集成一样)。

下面的示例演示了这一点，单独评估每个基础模型。

```py
# evaluate base models on the entire training dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('cart', DecisionTreeClassifier()))
	models.append(('svm', SVC(probability=True)))
	models.append(('bayes', GaussianNB()))
	return models

# define dataset
X, y = get_dataset()
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# summarize data split
print('Train: %s, Test: %s' % (X_train_full.shape, X_test.shape))
# create the base models
models = get_models()
# evaluate standalone model
for name, model in models:
	# fit the model on the training dataset
	model.fit(X_train_full, y_train_full)
	# make a prediction on the test dataset
	yhat = model.predict(X_test)
	# evaluate the predictions
	score = accuracy_score(y_test, yhat)
	# report the score
	print('>%s Accuracy: %.3f' % (name, score*100))
```

运行该示例首先报告完整训练和测试数据集的形状，然后报告测试数据集上每个基础模型的准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到所有模型的表现都比混合集成差。

有趣的是，我们可以看到 SVM 非常接近达到 98.200%的准确率，相比之下混合集成达到 98.240%。

```py
Train: (5000, 20), Test: (5000, 20)
>lr Accuracy: 87.800
>knn Accuracy: 97.380
>cart Accuracy: 88.200
>svm Accuracy: 98.200
>bayes Accuracy: 87.300
```

我们可以选择使用混合集成作为我们的最终模型。

这包括在整个训练数据集上拟合集合，并对新的例子进行预测。具体来说，将整个训练数据集分割成训练集和验证集，分别训练基本模型和元模型，然后可以使用集成进行预测。

下面列出了使用混合集成对新数据进行分类预测的完整示例。

```py
# example of making a prediction with a blending ensemble for classification
from numpy import hstack
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('cart', DecisionTreeClassifier()))
	models.append(('svm', SVC(probability=True)))
	models.append(('bayes', GaussianNB()))
	return models

# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, y_train, y_val):
	# fit all models on the training set and predict on hold out set
	meta_X = list()
	for _, model in models:
		# fit in training set
		model.fit(X_train, y_train)
		# predict on hold out set
		yhat = model.predict_proba(X_val)
		# store predictions as input for blending
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# define blending model
	blender = LogisticRegression()
	# fit on predictions from base models
	blender.fit(meta_X, y_val)
	return blender

# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for _, model in models:
		# predict with base model
		yhat = model.predict_proba(X_test)
		# store prediction
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict(meta_X)

# define dataset
X, y = get_dataset()
# split dataset set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize data split
print('Train: %s, Val: %s' % (X_train.shape, X_val.shape))
# create the base models
models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make a prediction on a new row of data
row = [-0.30335011, 2.68066314, 2.07794281, 1.15253537, -2.0583897, -2.51936601, 0.67513028, -3.20651939, -1.60345385, 3.68820714, 0.05370913, 1.35804433, 0.42011397, 1.4732839, 2.89997622, 1.61119399, 7.72630965, -2.84089477, -1.83977415, 1.34381989]
yhat = predict_ensemble(models, blender, [row])
# summarize prediction
print('Predicted Class: %d' % (yhat))
```

运行该示例适合数据集上的混合集成模型，然后用于对新的数据行进行预测，就像我们在应用程序中使用该模型时可能做的那样。

```py
Train: (6700, 20), Val: (3300, 20)
Predicted Class: 1
```

接下来，让我们探索如何评估回归的混合集合。

## 回归的混合集成

在这一节中，我们将研究使用堆叠来解决回归问题。

首先，我们可以使用[make _ revolution()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_regression.html)创建一个包含 10，000 个示例和 20 个输入特征的合成回归问题。

下面列出了完整的示例。

```py
# test regression dataset
from sklearn.datasets import make_regression
# define dataset
X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例将创建数据集并总结输入和输出组件的形状。

```py
(10000, 20) (10000,)
```

接下来，我们可以定义用作基础模型的回归模型列表。在这种情况下，我们将使用线性回归、kNN、决策树和 SVM 模型。

```py
# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LinearRegression()))
	models.append(('knn', KNeighborsRegressor()))
	models.append(('cart', DecisionTreeRegressor()))
	models.append(('svm', SVR()))
	return models
```

用于训练混合集成的 *fit_ensemble()* 函数与分类没有变化，除了用于混合的模型必须改为回归模型。

在这种情况下，我们将使用线性回归模型。

```py
...
# define blending model
blender = LinearRegression()
```

假设这是一个回归问题，我们将使用误差度量来评估模型的表现，在这种情况下，是平均绝对误差，简称 MAE。

```py
...
# evaluate predictions
score = mean_absolute_error(y_test, yhat)
print('Blending MAE: %.3f' % score)
```

将这些联系在一起，下面列出了合成回归预测建模问题的混合集成的完整示例。

```py
# evaluate blending ensemble for regression
from numpy import hstack
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# get the dataset
def get_dataset():
	X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
	return X, y

# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LinearRegression()))
	models.append(('knn', KNeighborsRegressor()))
	models.append(('cart', DecisionTreeRegressor()))
	models.append(('svm', SVR()))
	return models

# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, y_train, y_val):
	# fit all models on the training set and predict on hold out set
	meta_X = list()
	for name, model in models:
		# fit in training set
		model.fit(X_train, y_train)
		# predict on hold out set
		yhat = model.predict(X_val)
		# reshape predictions into a matrix with one column
		yhat = yhat.reshape(len(yhat), 1)
		# store predictions as input for blending
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# define blending model
	blender = LinearRegression()
	# fit on predictions from base models
	blender.fit(meta_X, y_val)
	return blender

# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for name, model in models:
		# predict with base model
		yhat = model.predict(X_test)
		# reshape predictions into a matrix with one column
		yhat = yhat.reshape(len(yhat), 1)
		# store prediction
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict(meta_X)

# define dataset
X, y = get_dataset()
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# split training set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# summarize data split
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))
# create the base models
models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make predictions on test set
yhat = predict_ensemble(models, blender, X_test)
# evaluate predictions
score = mean_absolute_error(y_test, yhat)
print('Blending MAE: %.3f' % score)
```

运行该示例首先报告训练、验证和测试数据集的形状，然后报告测试数据集上集合的 MAE。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到混合集成在测试数据集上实现了大约 0.237 的 MAE。

```py
Train: (3350, 20), Val: (1650, 20), Test: (5000, 20)
Blending MAE: 0.237
```

与分类一样，混合集成只有在表现优于任何有助于集成的基础模型时才是有用的。

我们可以通过单独评估每个基础模型来检查这一点，方法是首先在整个训练数据集上拟合它(不同于混合集成)，然后在测试数据集上进行预测(类似于混合集成)。

下面的示例在合成回归预测建模数据集上单独评估每个基本模型。

```py
# evaluate base models in isolation on the regression dataset
from numpy import hstack
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# get the dataset
def get_dataset():
	X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
	return X, y

# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LinearRegression()))
	models.append(('knn', KNeighborsRegressor()))
	models.append(('cart', DecisionTreeRegressor()))
	models.append(('svm', SVR()))
	return models

# define dataset
X, y = get_dataset()
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# summarize data split
print('Train: %s, Test: %s' % (X_train_full.shape, X_test.shape))
# create the base models
models = get_models()
# evaluate standalone model
for name, model in models:
	# fit the model on the training dataset
	model.fit(X_train_full, y_train_full)
	# make a prediction on the test dataset
	yhat = model.predict(X_test)
	# evaluate the predictions
	score = mean_absolute_error(y_test, yhat)
	# report the score
	print('>%s MAE: %.3f' % (name, score))
```

运行该示例首先报告完整的训练和测试数据集的形状，然后报告测试数据集上每个基础模型的 MAE。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到线性回归模型确实比混合集成表现得稍好，实现了 0.236 的 MAE，而集成为 0.237。这可能是因为合成数据集的构建方式。

然而，在这种情况下，我们会选择使用线性回归模型直接解决这个问题。这突出了在采用集合模型作为最终模型之前检查贡献模型的表现的重要性。

```py
Train: (5000, 20), Test: (5000, 20)
>lr MAE: 0.236
>knn MAE: 100.169
>cart MAE: 133.744
>svm MAE: 138.195
```

同样，我们可以选择使用混合集合作为回归的最终模型。

这包括拟合将整个数据集分割成训练集和验证集，以分别拟合基本模型和元模型，然后集成可以用于预测新的数据行。

下面列出了使用混合集合对新数据进行回归预测的完整示例。

```py
# example of making a prediction with a blending ensemble for regression
from numpy import hstack
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# get the dataset
def get_dataset():
	X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
	return X, y

# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LinearRegression()))
	models.append(('knn', KNeighborsRegressor()))
	models.append(('cart', DecisionTreeRegressor()))
	models.append(('svm', SVR()))
	return models

# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, y_train, y_val):
	# fit all models on the training set and predict on hold out set
	meta_X = list()
	for _, model in models:
		# fit in training set
		model.fit(X_train, y_train)
		# predict on hold out set
		yhat = model.predict(X_val)
		# reshape predictions into a matrix with one column
		yhat = yhat.reshape(len(yhat), 1)
		# store predictions as input for blending
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# define blending model
	blender = LinearRegression()
	# fit on predictions from base models
	blender.fit(meta_X, y_val)
	return blender

# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for _, model in models:
		# predict with base model
		yhat = model.predict(X_test)
		# reshape predictions into a matrix with one column
		yhat = yhat.reshape(len(yhat), 1)
		# store prediction
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict(meta_X)

# define dataset
X, y = get_dataset()
# split dataset set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize data split
print('Train: %s, Val: %s' % (X_train.shape, X_val.shape))
# create the base models
models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make a prediction on a new row of data
row = [-0.24038754, 0.55423865, -0.48979221, 1.56074459, -1.16007611, 1.10049103, 1.18385406, -1.57344162, 0.97862519, -0.03166643, 1.77099821, 1.98645499, 0.86780193, 2.01534177, 2.51509494, -1.04609004, -0.19428148, -0.05967386, -2.67168985, 1.07182911]
yhat = predict_ensemble(models, blender, [row])
# summarize prediction
print('Predicted: %.3f' % (yhat[0]))
```

运行该示例适合数据集上的混合集成模型，然后用于对新的数据行进行预测，就像我们在应用程序中使用该模型时可能做的那样。

```py
Train: (6700, 20), Val: (3300, 20)
Predicted: 359.986
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 相关教程

*   [用 Python 堆叠集成机器学习](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)
*   [如何用 Python 从零开始实现堆叠泛化(堆叠)](https://machinelearningmastery.com/implementing-stacking-scratch-python/)

### 报纸

*   [特征加权线性叠加](https://arxiv.org/abs/0911.0460)，2009。
*   [BellKor 2008 年 Netflix 奖的解决方案](https://netflixprize.com/assets/ProgressPrize2008_BellKor.pdf)，2008 年。
*   [鹿儿岛联合指南，mlwave](https://mlwave.com/kaggle-ensembling-guide/)2015 年。

### 文章

*   [网飞奖，维基百科](https://en.wikipedia.org/wiki/Netflix_Prize)。

## 摘要

在本教程中，您发现了如何在 python 中开发和评估混合集成。

具体来说，您了解到:

*   混合集成是一种堆叠类型，其中元模型使用保持验证数据集上的预测来拟合，而不是折叠预测。
*   如何开发混合集成，包括用于训练模型和对新数据进行预测的功能。
*   如何评估分类和回归预测建模问题的混合集成？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。*