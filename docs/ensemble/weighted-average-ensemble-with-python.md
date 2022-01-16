# 如何用 Python 开发加权平均集成

> 原文：<https://machinelearningmastery.com/weighted-average-ensemble-with-python/>

最后更新于 2021 年 5 月 8 日

**加权平均集成**假设集成中的一些模型比其他模型具有更多的技能，并在进行预测时给予它们更多的贡献。

加权平均或加权和集成是对投票集成的扩展，投票集成假设所有模型都同样熟练，并且对集成做出的预测做出相同的比例贡献。

每个模型都分配有一个固定的权重，该权重乘以模型所做的预测，并用于总和或平均预测计算。这种集成的挑战是如何计算、分配或搜索模型权重，从而获得比任何贡献模型和使用相同模型权重的集成更好的表现。

在本教程中，您将发现如何开发用于分类和回归的加权平均集成。

完成本教程后，您将知道:

*   加权平均集成是投票集成的扩展，其中模型投票与模型表现成比例。
*   如何使用 Sklearn 的投票集成开发加权平均集成？
*   如何评估加权平均集成进行分类和回归，并确认模型是熟练的。

**用我的新书[Python 集成学习算法](https://machinelearningmastery.com/ensemble-learning-algorithms-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2021 年 5 月更新**:加权平均的固定定义。

![How to Develop a Weighted Average Ensemble With Python](img/1b3c7ee05a4146eb2aabec670b02f56e.png)

如何用 Python 开发加权平均集成
图片由 [Alaina McDavid](https://www.flickr.com/photos/allie444/7694667194/) 提供，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  加权平均集成
2.  开发加权平均集成
3.  分类的加权平均集成
4.  回归的加权平均集成

## 加权平均集成

加权平均或加权和集成是一种集成机器学习方法，它结合了来自多个模型的预测，其中每个模型的贡献与其能力或技能成比例地加权。

[加权平均](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean)集合与投票集合相关。

[投票集成](https://machinelearningmastery.com/voting-ensembles-with-python/)由多个机器学习模型组成，其中来自每个模型的预测被直接平均。对于回归，这包括计算集成成员所做预测的[算术平均值](https://machinelearningmastery.com/arithmetic-geometric-and-harmonic-means-for-machine-learning/)。对于分类，这可能涉及计算统计模式(最常见的类别标签)或类似的投票方案，或者对每个类别的预测概率进行求和，并选择具有最大求和概率的类别。

有关投票集成的更多信息，请参见教程:

*   [如何用 Python 开发投票集成](https://machinelearningmastery.com/voting-ensembles-with-python/)

投票集成技术的一个限制是，它假设集成中的所有模型都同等有效。情况可能并非如此，因为有些模型可能比其他模型更好，尤其是如果使用不同的机器学习算法来训练每个模型集成成员。

投票的另一种选择是假设全体成员的能力并不完全相同，相反，一些模特比其他模特更优秀，在做预测时应该得到更多的选票或更多的席位。这为加权和或加权平均集成方法提供了动机。

在回归中，使用算术平均值计算平均预测，例如预测之和除以总预测。例如，如果一个集成有三个集成成员，则缩减可能是:

*   **型号 1** : 97.2
*   **型号 2** : 100.0
*   **型号 3** : 95.8

平均预测计算如下:

*   yhat = (97.2 + 100.0 + 95.8) / 3
*   yhat = 293 / 3
*   yhat = 97，666

加权平均预测包括首先给每个集成成员分配一个固定的权重系数。这可能是一个介于 0 和 1 之间的浮点值，代表权重的百分比。它也可以是从 1 开始的整数，代表给每个模型的投票数。

例如，集成成员的固定权重可能为 0.84、0.87、0.75。这些权重可用于计算加权平均值，方法是将每个预测乘以模型的权重，得出加权总和，然后将该值除以权重总和。例如:

*   yhat =((97.2 * 0.84)+(100.0 * 0.87)+(95.8 * 0.75))/(0.84+0.87+0.75)
*   yhat =(81 648+87+71.85)/(0.84+0.87+0.75)
*   yhat = 240 498/2.46
*   yhat = 97，763

我们可以看到，只要分数有相同的尺度，权重有相同的尺度并且是最大化的(意味着权重越大越好)，加权和就会产生一个可感知的值，反过来，加权平均也是可感知的，意味着结果的尺度与分数的尺度相匹配。

这种相同的方法可以用于计算每个清晰类别标签的加权投票和或分类问题上每个类别标签的加权概率和。

使用加权平均集成的挑战性方面是如何为每个集成成员选择相对权重。

有许多方法可以使用。例如，可以基于每个模型的技能来选择权重，例如分类准确率或负误差，其中大的权重意味着表现更好的模型。表现可以在用于训练的数据集或保持数据集上计算，后者可能更相关。

每个模型的分数可以直接使用，也可以转换成不同的值，例如每个模型的相对排名。另一种方法可能是使用搜索算法来测试不同的权重组合。

现在我们已经熟悉了加权平均集成方法，让我们看看如何开发和评估它们。

## 开发加权平均集成

在本节中，我们将开发、评估和使用加权平均或加权和集合模型。

我们可以手动实现加权平均集成，尽管这不是必需的，因为我们可以使用 Sklearn 库中的投票集成来实现期望的效果。具体来说，[voting revolutionor](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)和 [VotingClassifier](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) 类可以分别用于回归和分类，并且两者都提供了“权重”参数，该参数指定了每个集成成员在进行预测时的相对贡献。

基本模型列表通过“*估计量*参数提供。这是一个 Python 列表，其中列表中的每个元素都是一个元组，具有模型的名称和配置的模型实例。列表中的每个模型必须有唯一的名称。

例如，我们可以用两个集成成员定义一个加权平均集成进行分类，如下所示:

```py
...
# define the models in the ensemble
models = [('lr',LogisticRegression()),('svm',SVC())]
# define the weight of each model in the ensemble
weights = [0.7, 0.9]
# create a weighted sum ensemble
ensemble = VotingClassifier(estimators=models, weights=weights)
```

此外，用于分类的投票集成提供了“投票”参数，该参数在计算用于预测的加权和时，支持用于组合清晰类标签的硬投票(‘T0’硬【T1’)和用于组合类概率的软投票(‘T2’软【T3’)；例如:

```py
...
# define the models in the ensemble
models = [('lr',LogisticRegression()),('svm',SVC())]
# define the weight of each model in the ensemble
weights = [0.7, 0.9]
# create a weighted sum ensemble
ensemble = VotingClassifier(estimators=models, weights=weights, voting='soft')
```

如果贡献模型支持预测类概率，软投票通常是首选，因为它通常会带来更好的表现。预测概率的加权和也是如此。

现在我们已经熟悉了如何使用投票集成 API 来开发加权平均集成，让我们来看看一些工作示例。

## 分类的加权平均集成

在这一节中，我们将研究使用加权平均集成来解决分类问题。

首先，我们可以使用 [make_classification()函数](http://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建一个包含 10，000 个示例和 20 个输入特征的合成二进制分类问题。

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

接下来，我们可以在这个数据集上评估加权平均集成算法。

首先，我们将数据集分割成 50-50 分割的训练集和测试集。然后，我们将整个训练集分成训练模型的子集和验证的子集。

```py
...
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
# split the full train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
```

接下来，我们将定义一个函数来创建一个在集合中使用的模型列表。在这种情况下，我们将使用不同的分类模型集合，包括逻辑回归、决策树和[朴素贝叶斯](https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/)。

```py
# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('cart', DecisionTreeClassifier()))
	models.append(('bayes', GaussianNB()))
	return models
```

接下来，我们需要权衡每个集成成员。

在这种情况下，我们将使用训练数据集上每个集成模型的表现作为模型在进行预测时的相对权重。表现将使用分类准确率作为 0 到 1 之间正确预测的百分比来计算，值越大意味着模型越好，进而对预测的贡献越大。

每个集成模型将首先适合训练集，然后在验证集上进行评估。验证集的准确性将用作模型权重。

下面的 *evaluate_models()* 函数实现了这一点，返回每个模型的表现。

```py
# evaluate each base model
def evaluate_models(models, X_train, X_val, y_train, y_val):
	# fit and evaluate the models
	scores = list()
	for name, model in models:
		# fit the model
		model.fit(X_train, y_train)
		# evaluate the model
		yhat = model.predict(X_val)
		acc = accuracy_score(y_val, yhat)
		# store the performance
		scores.append(acc)
		# report model performance
	return scores
```

然后，我们可以调用这个函数来获得分数，并将它们用作集合的权重。

```py
...
# fit and evaluate each model
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
# create the ensemble
ensemble = VotingClassifier(estimators=models, voting='soft', weights=scores)
```

然后，我们可以在完整的训练数据集上拟合集合，并在保持测试集上对其进行评估。

```py
...
# fit the ensemble on the training dataset
ensemble.fit(X_train, y_train)
# make predictions on test set
yhat = ensemble.predict(X_test)
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Weighted Avg Accuracy: %.3f' % (score*100))
```

将这些联系在一起，完整的示例如下所示。

```py
# evaluate a weighted average ensemble for classification
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('cart', DecisionTreeClassifier()))
	models.append(('bayes', GaussianNB()))
	return models

# evaluate each base model
def evaluate_models(models, X_train, X_val, y_train, y_val):
	# fit and evaluate the models
	scores = list()
	for name, model in models:
		# fit the model
		model.fit(X_train, y_train)
		# evaluate the model
		yhat = model.predict(X_val)
		acc = accuracy_score(y_val, yhat)
		# store the performance
		scores.append(acc)
		# report model performance
	return scores

# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
# split the full train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# create the base models
models = get_models()
# fit and evaluate each model
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
# create the ensemble
ensemble = VotingClassifier(estimators=models, voting='soft', weights=scores)
# fit the ensemble on the training dataset
ensemble.fit(X_train_full, y_train_full)
# make predictions on test set
yhat = ensemble.predict(X_test)
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Weighted Avg Accuracy: %.3f' % (score*100))
```

运行该示例首先评估每个独立模型，并报告将用作模型权重的准确度分数。最后，在报告表现的测试上拟合和评估加权平均集成。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到投票集成实现了大约 90.960%的分类准确率。

```py
[0.8896969696969697, 0.8575757575757575, 0.8812121212121212]
Weighted Avg Accuracy: 90.960
```

我们的期望是，这个乐团将会比任何一个有贡献的乐团成员表现得更好。问题是用作权重的模型的准确度分数不能直接与集成的表现进行比较，因为成员是在训练子集上评估的，而集成是在测试数据集上评估的。

我们可以更新示例，并添加对每个独立模型的评估，以便进行比较。

```py
...
# evaluate each standalone model
scores = evaluate_models(models, X_train_full, X_test, y_train_full, y_test)
for i in range(len(models)):
	print('>%s: %.3f' % (models[i][0], scores[i]*100))
```

我们还期望加权平均集成比同等加权的投票集成表现得更好。

这也可以通过显式评估投票集合来检查。

```py
...
# evaluate equal weighting
ensemble = VotingClassifier(estimators=models, voting='soft')
ensemble.fit(X_train_full, y_train_full)
yhat = ensemble.predict(X_test)
score = accuracy_score(y_test, yhat)
print('Voting Accuracy: %.3f' % (score*100))
```

将这些联系在一起，完整的示例如下所示。

```py
# evaluate a weighted average ensemble for classification compared to base model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('cart', DecisionTreeClassifier()))
	models.append(('bayes', GaussianNB()))
	return models

# evaluate each base model
def evaluate_models(models, X_train, X_val, y_train, y_val):
	# fit and evaluate the models
	scores = list()
	for name, model in models:
		# fit the model
		model.fit(X_train, y_train)
		# evaluate the model
		yhat = model.predict(X_val)
		acc = accuracy_score(y_val, yhat)
		# store the performance
		scores.append(acc)
		# report model performance
	return scores

# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
# split the full train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# create the base models
models = get_models()
# fit and evaluate each model
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
# create the ensemble
ensemble = VotingClassifier(estimators=models, voting='soft', weights=scores)
# fit the ensemble on the training dataset
ensemble.fit(X_train_full, y_train_full)
# make predictions on test set
yhat = ensemble.predict(X_test)
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Weighted Avg Accuracy: %.3f' % (score*100))
# evaluate each standalone model
scores = evaluate_models(models, X_train_full, X_test, y_train_full, y_test)
for i in range(len(models)):
	print('>%s: %.3f' % (models[i][0], scores[i]*100))
# evaluate equal weighting
ensemble = VotingClassifier(estimators=models, voting='soft')
ensemble.fit(X_train_full, y_train_full)
yhat = ensemble.predict(X_test)
score = accuracy_score(y_test, yhat)
print('Voting Accuracy: %.3f' % (score*100))
```

运行该示例首先像以前一样准备和评估加权平均集成，然后报告单独评估的每个贡献模型的表现，最后是对贡献模型使用相同权重的投票集成。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到加权平均集成比任何贡献的集成成员表现得更好。

我们还可以看到，等权重集成(投票)获得了大约 90.620%的准确率，这低于获得略高的 90.760%准确率的加权集成。

```py
[0.8896969696969697, 0.8703030303030304, 0.8812121212121212]
Weighted Avg Accuracy: 90.760
>lr: 87.800
>cart: 88.180
>bayes: 87.300
Voting Accuracy: 90.620
```

接下来，让我们看看如何开发和评估用于回归的加权平均集成。

## 回归的加权平均集成

在这一节中，我们将研究使用加权平均集成来解决回归问题。

首先，我们可以使用[make _ revolution()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_regression.html)创建一个包含 1000 个示例和 20 个输入特征的合成回归问题。

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

接下来，我们可以在这个数据集上评估加权平均集成模型。

首先，我们可以将数据集分割成训练集和测试集，然后进一步将训练集分割成训练集和验证集，这样我们就可以估计每个贡献模型的表现。

```py
...
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
# split the full train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
```

我们可以定义在集合中使用的模型列表。在这种情况下，我们将使用 k 近邻、决策树和支持向量回归。

```py
# get a list of base models
def get_models():
	models = list()
	models.append(('knn', KNeighborsRegressor()))
	models.append(('cart', DecisionTreeRegressor()))
	models.append(('svm', SVR()))
	return models
```

接下来，我们可以更新 *evaluate_models()* 函数来计算等待验证数据集上每个集成成员的平均绝对误差(MAE)。

我们将使用负的 MAE 分数作为权重，其中接近零的大误差值表示更好的模型表现。

```py
# evaluate each base model
def evaluate_models(models, X_train, X_val, y_train, y_val):
	# fit and evaluate the models
	scores = list()
	for name, model in models:
		# fit the model
		model.fit(X_train, y_train)
		# evaluate the model
		yhat = model.predict(X_val)
		mae = mean_absolute_error(y_val, yhat)
		# store the performance
		scores.append(-mae)
		# report model performance
	return scores
```

然后，我们可以调用这个函数来获得分数，并使用它们来定义回归的加权平均集合。

```py
...
# fit and evaluate each model
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
# create the ensemble
ensemble = VotingRegressor(estimators=models, weights=scores)
```

然后，我们可以在整个训练数据集上拟合集合，并在保持测试数据集上评估表现。

```py
...
# fit the ensemble on the training dataset
ensemble.fit(X_train_full, y_train_full)
# make predictions on test set
yhat = ensemble.predict(X_test)
# evaluate predictions
score = mean_absolute_error(y_test, yhat)
print('Weighted Avg MAE: %.3f' % (score))
```

我们期望集成比任何贡献的集成成员表现得更好，这可以通过独立评估完整列车和测试集上的每个成员模型来直接检查。

```py
...
# evaluate each standalone model
scores = evaluate_models(models, X_train_full, X_test, y_train_full, y_test)
for i in range(len(models)):
	print('>%s: %.3f' % (models[i][0], scores[i]))
```

最后，我们还期望加权平均集成比具有相同权重的相同集成表现得更好。这也是可以证实的。

```py
...
# evaluate equal weighting
ensemble = VotingRegressor(estimators=models)
ensemble.fit(X_train_full, y_train_full)
yhat = ensemble.predict(X_test)
score = mean_absolute_error(y_test, yhat)
print('Voting MAE: %.3f' % (score))
```

将这些联系在一起，下面列出了评估回归的加权平均集合的完整示例。

```py
# evaluate a weighted average ensemble for regression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor

# get a list of base models
def get_models():
	models = list()
	models.append(('knn', KNeighborsRegressor()))
	models.append(('cart', DecisionTreeRegressor()))
	models.append(('svm', SVR()))
	return models

# evaluate each base model
def evaluate_models(models, X_train, X_val, y_train, y_val):
	# fit and evaluate the models
	scores = list()
	for name, model in models:
		# fit the model
		model.fit(X_train, y_train)
		# evaluate the model
		yhat = model.predict(X_val)
		mae = mean_absolute_error(y_val, yhat)
		# store the performance
		scores.append(-mae)
		# report model performance
	return scores

# define dataset
X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
# split the full train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# create the base models
models = get_models()
# fit and evaluate each model
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
# create the ensemble
ensemble = VotingRegressor(estimators=models, weights=scores)
# fit the ensemble on the training dataset
ensemble.fit(X_train_full, y_train_full)
# make predictions on test set
yhat = ensemble.predict(X_test)
# evaluate predictions
score = mean_absolute_error(y_test, yhat)
print('Weighted Avg MAE: %.3f' % (score))
# evaluate each standalone model
scores = evaluate_models(models, X_train_full, X_test, y_train_full, y_test)
for i in range(len(models)):
	print('>%s: %.3f' % (models[i][0], scores[i]))
# evaluate equal weighting
ensemble = VotingRegressor(estimators=models)
ensemble.fit(X_train_full, y_train_full)
yhat = ensemble.predict(X_test)
score = mean_absolute_error(y_test, yhat)
print('Voting MAE: %.3f' % (score))
```

运行该示例首先报告将用作分数的每个集成成员的负 MAE，然后是加权平均集成的表现。最后，报告每个独立模型的表现以及具有相同权重的集合的表现。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到加权平均集成实现了大约 105.158 的平均绝对误差，这比实现了大约 100.169 的误差的独立 kNN 模型更差(大误差)。我们还可以看到，假设每个模型权重相等的投票集成也比加权平均集成表现更好，误差约为 102.706。

```py
[-101.97000126284476, -142.26014983127837, -153.9765827528269]
Weighted Avg MAE: 105.158
>knn: -100.169
>cart: -134.487
>svm: -138.195
Voting MAE: 102.706
```

加权平均集成的表现比预期的差可能与模型加权方式的选择有关。

另一种加权策略是使用排名来表示每个集合在加权平均值中的票数。

例如，在三个集成成员的情况下，表现最差的模型有 1 票，第二差的有 2 票，最佳模型有 3 票。

这可以使用 [argsort() numpy 函数](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html)来实现。

argsort 函数返回数组中已排序的值的索引。所以，如果我们有数组[300，100，200]，最小值的索引是 1，下一个最大值的索引是 2，下一个最大值的索引是 0。

因此，[300，100，200]的 argsort 为[1，2，0]。

然后，我们可以对 argsort 的结果进行 argsort，以给出原始数组中数据的排名。要了解如何操作，argsort，2，0]将指示索引 2 是最小值，后跟索引 0 并以索引 1 结束。

因此，[1，2，0]的 argsort 为[2，0，1]。换句话说，[300，100，200]的 argsort 的 argsort 是[2，0，1]，这是如果值按升序排序，数组中每个值的相对排名。那就是:

*   300:排名第二
*   100:等级为 0
*   200:排名第 1

我们可以用下面列出的一个小例子来说明这一点。

```py
# demonstrate argsort
from numpy import argsort
# data
x = [300, 100, 200]
print(x)
# argsort of data
print(argsort(x))
# arg sort of argsort of data
print(argsort(argsort(x)))
```

运行该示例首先报告原始数据，然后报告原始数据的 argsort 和原始数据的 argsort。

结果与我们的手动计算相匹配。

```py
[300, 100, 200]
[1 2 0]
[2 0 1]
```

我们可以使用模型得分的 argsort 的 argsort 来计算每个集成成员的相对排名。如果负平均绝对误差按升序排序，那么最佳模型将具有最大的负误差，进而具有最高的等级。表现最差的模型将具有最小的负误差，并且依次具有最低的等级。

同样，我们可以用一个工作实例来证实这一点。

```py
# demonstrate argsort with negative scores
from numpy import argsort
# data
x = [-10, -100, -80]
print(x)
# argsort of data
print(argsort(x))
# arg sort of argsort of data
print(argsort(argsort(x)))
```

运行示例，我们可以看到第一个模型的得分最好(-10)，第二个模型的得分最差(-100)。

得分的 argsort 的 argsort 显示最佳模型获得最高等级(最多票数)值为 2，最差模型获得最低等级(最少票数)值为 0。

```py
[-10, -100, -80]
[1 2 0]
[2 0 1]
```

实际上，我们不希望任何一个模特获得零票，因为它会被排除在集成之外。因此，我们可以给所有排名加 1。

在计算分数之后，我们可以计算模型分数的 argsort 的 argsort 来给出排名。然后使用模型排名作为加权平均集成的模型权重。

```py
...
# fit and evaluate each model
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
ranking = 1 + argsort(argsort(scores))
print(ranking)
# create the ensemble
ensemble = VotingRegressor(estimators=models, weights=ranking)
```

将这些联系在一起，下面列出了用于回归的加权平均集合的完整示例，模型排名用作模型权重。

```py
# evaluate a weighted average ensemble for regression with rankings for model weights
from numpy import argsort
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor

# get a list of base models
def get_models():
	models = list()
	models.append(('knn', KNeighborsRegressor()))
	models.append(('cart', DecisionTreeRegressor()))
	models.append(('svm', SVR()))
	return models

# evaluate each base model
def evaluate_models(models, X_train, X_val, y_train, y_val):
	# fit and evaluate the models
	scores = list()
	for name, model in models:
		# fit the model
		model.fit(X_train, y_train)
		# evaluate the model
		yhat = model.predict(X_val)
		mae = mean_absolute_error(y_val, yhat)
		# store the performance
		scores.append(-mae)
		# report model performance
	return scores

# define dataset
X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
# split the full train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# create the base models
models = get_models()
# fit and evaluate each model
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
ranking = 1 + argsort(argsort(scores))
print(ranking)
# create the ensemble
ensemble = VotingRegressor(estimators=models, weights=ranking)
# fit the ensemble on the training dataset
ensemble.fit(X_train_full, y_train_full)
# make predictions on test set
yhat = ensemble.predict(X_test)
# evaluate predictions
score = mean_absolute_error(y_test, yhat)
print('Weighted Avg MAE: %.3f' % (score))
# evaluate each standalone model
scores = evaluate_models(models, X_train_full, X_test, y_train_full, y_test)
for i in range(len(models)):
	print('>%s: %.3f' % (models[i][0], scores[i]))
# evaluate equal weighting
ensemble = VotingRegressor(estimators=models)
ensemble.fit(X_train_full, y_train_full)
yhat = ensemble.predict(X_test)
score = mean_absolute_error(y_test, yhat)
print('Voting MAE: %.3f' % (score))
```

运行该示例首先对每个模型进行评分，然后将评分转换为排名。然后评估使用排序的加权平均集成，并与每个独立模型和具有同等加权模型的集成的表现进行比较。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到排名如预期的那样进行，得分为 101 的表现最好的成员 kNN 被分配了 3 的排名，其他模型也相应地进行了排名。我们可以看到，加权平均集成实现了约 96.692 的 MAE，优于任何单个模型和未加权投票集成。

这突出了探索在集合中选择模型权重的替代方法的重要性。

```py
[-101.97000126284476, -141.51998518020065, -153.9765827528269]
[3 2 1]
Weighted Avg MAE: 96.692
>knn: -100.169
>cart: -132.976
>svm: -138.195
Voting MAE: 102.832
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 相关教程

*   [如何为深度学习神经网络开发加权平均集成](https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/)
*   [如何用 Python 开发投票集成](https://machinelearningmastery.com/voting-ensembles-with-python/)

### 蜜蜂

*   num py . argsort API。
*   [硬化。一起。投票分类器 API](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) 。
*   [硬化。一起。投票输入 API](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html) 。

### 文章

*   [加权算术平均值，维基百科](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean)。
*   [集成平均(机器学习)，维基百科](https://en.wikipedia.org/wiki/Ensemble_averaging_(machine_learning))。

## 摘要

在本教程中，您发现了如何开发用于分类和回归的加权平均集成。

具体来说，您了解到:

*   加权平均集成是投票集成的扩展，其中模型投票与模型表现成比例。
*   如何使用 Sklearn 的投票集成开发加权平均集成？
*   如何评估加权平均集成进行分类和回归，并确认模型是熟练的。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。