# 如何在 Sklearn 中保存和重用数据准备对象

> 原文：<https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-Sklearn-for-later-use/>

最后更新于 2020 年 6 月 30 日

将来，在训练数据集上执行的任何数据准备都必须在新的数据集上执行，这一点至关重要。

这可能包括评估模型时的测试数据集，或者使用模型进行预测时来自域的新数据。

通常，训练数据集中的模型拟合被保存以备后用。将来为模型准备新数据的正确解决方案是将任何数据准备对象(如数据缩放方法)与模型一起保存到文件中。

在本教程中，您将了解如何将模型和数据准备对象保存到文件中供以后使用。

完成本教程后，您将知道:

*   为机器学习模型正确准备测试数据和新数据的挑战。
*   将模型和数据准备对象保存到文件中以备后用的解决方案。
*   如何在新数据上保存和后期加载并使用机器学习模型和数据准备模型。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 1 月更新**:针对 Sklearn v0.22 API 的变化进行了更新。
*   **2020 年 5 月更新**:改进代码示例，打印输出。

![How to Save and Load Models and Data Preparation in Sklearn for Later Use](img/0cba995eae757a1620c499d20d6f898a.png)

如何在 Scikit 中保存和加载模型和数据准备-学习以备后用
图片由 [Dennis Jarvis](https://www.flickr.com/photos/archer10/21730827905/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  为模型准备新数据的挑战
2.  保存数据准备对象
3.  如何保存和以后使用数据准备对象

## 为模型准备新数据的挑战

数据集中的每个输入变量可能有不同的单位。

例如，一个变量可能以英寸为单位，另一个以英里为单位，另一个以天为单位，等等。

因此，在拟合模型之前缩放数据通常很重要。

这对于使用输入或距离度量的加权和的模型尤其重要，如逻辑回归、神经网络和 k 近邻。这是因为具有较大值或范围的变量可能会支配或抵消具有较小值或范围的变量的影响。

缩放技术，如标准化或规范化，具有将每个输入变量的分布转换为相同的效果，如标准化情况下的相同最小值和最大值，或标准化情况下的相同平均值和标准偏差。

缩放技术必须适合，这意味着它只需要从数据中计算系数，例如观察到的最小值和最大值，或者观察到的平均值和标准偏差。这些值也可以由领域专家设置。

在使用缩放技术评估模型时，最佳实践是将模型拟合到训练数据集中，然后将它们应用到训练和测试数据集中。

或者，在使用最终模型时，在训练数据集上调整缩放方法，并将转换应用于训练数据集和将来的任何新数据集。

应用于训练数据集的任何数据准备或转换在将来也应用于测试或其他数据集是至关重要的。

当所有的数据和模型都在内存中时，这很简单。

当模型被保存并在以后使用时，这是一个挑战。

保存拟合模型供以后使用(如最终模型)时，缩放数据的最佳做法是什么？

## 保存数据准备对象

解决方案是将数据准备对象与模型一起保存到文件中。

例如，通常使用 pickle 框架(内置于 Python 中)来保存机器学习模型供以后使用，例如保存最终模型。

这个相同的框架可以用来保存用于数据准备的对象。

之后，可以加载和使用模型和数据准备对象。

将整个对象保存到文件中很方便，例如模型对象和数据准备对象。然而，专家可能更喜欢只将模型参数保存到文件中，然后稍后加载它们，并将它们设置到新的模型对象中。这种方法也可以用于缩放数据的系数，例如每个变量的最小值和最大值，或者每个变量的平均值和标准偏差。

选择哪种方法适合您的项目由您决定，但我建议将模型和数据准备对象(或多个对象)直接保存到文件中以备后用。

为了使将对象和数据转换对象保存到文件的想法具体化，让我们看一个工作示例。

## 如何保存和以后使用数据准备对象

在本节中，我们将演示准备数据集、在数据集上拟合模型、将模型和数据转换对象保存到文件中，以及稍后加载模型和转换并在新数据上使用它们。

### 1.定义数据集

首先，我们需要一个数据集。

我们将使用 Sklearn 数据集的测试数据集，特别是一个二分类问题，通过 [make_blobs()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)随机创建两个输入变量。

下面的示例创建了一个包含 100 个示例、两个输入要素和两个类标签(0 和 1)的测试数据集。然后将数据集分成训练集和测试集，并报告每个变量的最小值和最大值。

重要的是， *random_state* 是在创建数据集和拆分数据时设置的，这样每次运行代码时都会创建相同的数据集并执行相同的数据拆分。

```py
# example of creating a test dataset and splitting it into train and test sets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the scale of each input variable
for i in range(X_test.shape[1]):
	print('>%d, train: min=%.3f, max=%.3f, test: min=%.3f, max=%.3f' %
		(i, X_train[:, i].min(), X_train[:, i].max(),
			X_test[:, i].min(), X_test[:, i].max()))
```

运行该示例会报告训练和测试数据集中每个变量的最小值和最大值。

我们可以看到，每个变量都有不同的尺度，并且训练数据集和测试数据集之间的尺度不同。这是一个现实的场景，我们可能会遇到真实的数据集。

```py
>0, train: min=-11.856, max=0.526, test: min=-11.270, max=0.085
>1, train: min=-6.388, max=6.507, test: min=-5.581, max=5.926
```

### 2.缩放数据集

接下来，我们可以缩放数据集。

我们将使用[最小最大缩放器](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)将每个输入变量缩放到范围[0，1]。使用此定标器的最佳实践是将其放在训练数据集上，然后将转换应用于训练数据集和其他数据集:在本例中是测试数据集。

下面列出了缩放数据和总结效果的完整示例。

```py
# example of scaling the dataset
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# define scaler
scaler = MinMaxScaler()
# fit scaler on the training dataset
scaler.fit(X_train)
# transform both datasets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# summarize the scale of each input variable
for i in range(X_test.shape[1]):
	print('>%d, train: min=%.3f, max=%.3f, test: min=%.3f, max=%.3f' %
		(i, X_train_scaled[:, i].min(), X_train_scaled[:, i].max(),
			X_test_scaled[:, i].min(), X_test_scaled[:, i].max()))
```

运行该示例将打印缩放数据的效果，显示训练和测试数据集中每个变量的最小值和最大值。

我们可以看到，两个数据集中的所有变量现在的值都在 0 到 1 的期望范围内。

```py
>0, train: min=0.000, max=1.000, test: min=0.047, max=0.964
>1, train: min=0.000, max=1.000, test: min=0.063, max=0.955
```

### 3.保存模型和数据缩放器

接下来，我们可以在训练数据集上拟合模型，并将模型和缩放器对象保存到文件中。

我们将使用[物流配送](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)模型，因为这个问题是一个简单的二分类任务。

训练数据集像以前一样缩放，在这种情况下，我们将假设测试数据集当前不可用。缩放后，数据集用于拟合逻辑回归模型。

我们将使用 pickle 框架将*后勤导出*模型保存到一个文件中，并将*最小最大缩放器*保存到另一个文件中。

下面列出了完整的示例。

```py
# example of fitting a model on the scaled dataset
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from pickle import dump
# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# split data into train and test sets
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.33, random_state=1)
# define scaler
scaler = MinMaxScaler()
# fit scaler on the training dataset
scaler.fit(X_train)
# transform the training dataset
X_train_scaled = scaler.transform(X_train)
# define model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train_scaled, y_train)
# save the model
dump(model, open('model.pkl', 'wb'))
# save the scaler
dump(scaler, open('scaler.pkl', 'wb'))
```

运行该示例会缩放数据，拟合模型，并使用 pickle 将模型和缩放器保存到文件中。

您当前的工作目录中应该有两个文件:

*   *型号*
*   *scaler . PK*

### 4.负载模型和数据缩放器

最后，我们可以加载模型和缩放器对象并使用它们。

在这种情况下，我们将假设训练数据集不可用，并且只有新数据或测试数据集可用。

我们将加载模型和定标器，然后使用定标器准备新数据并使用模型进行预测。因为它是一个测试数据集，所以我们有预期的目标值，所以我们将把预测与预期的目标值进行比较，并计算模型的准确性。

下面列出了完整的示例。

```py
# load model and scaler and make predictions on new data
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pickle import load
# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# split data into train and test sets
_, X_test, _, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# load the model
model = load(open('model.pkl', 'rb'))
# load the scaler
scaler = load(open('scaler.pkl', 'rb'))
# check scale of the test set before scaling
print('Raw test set range')
for i in range(X_test.shape[1]):
	print('>%d, min=%.3f, max=%.3f' % (i, X_test[:, i].min(), X_test[:, i].max()))
# transform the test dataset
X_test_scaled = scaler.transform(X_test)
print('Scaled test set range')
for i in range(X_test_scaled.shape[1]):
	print('>%d, min=%.3f, max=%.3f' % (i, X_test_scaled[:, i].min(), X_test_scaled[:, i].max()))
# make predictions on the test set
yhat = model.predict(X_test_scaled)
# evaluate accuracy
acc = accuracy_score(y_test, yhat)
print('Test Accuracy:', acc)
```

运行该示例加载模型和定标器，然后使用定标器为模型正确准备测试数据集，满足模型在训练时的期望。

为了确认缩放器具有所需的效果，我们报告了应用缩放之前和之后每个输入特征的最小值和最大值。然后，该模型对测试集中的示例进行预测，并计算分类准确率。

在这种情况下，正如预期的那样，数据集正确地规范化了模型，在测试集上实现了 100%的准确性，因为测试问题微不足道。

```py
Raw test set range
>0, min=-11.270, max=0.085
>1, min=-5.581, max=5.926

Scaled test set range
>0, min=0.047, max=0.964
>1, min=0.063, max=0.955

Test Accuracy: 1.0
```

这提供了一个模板，您可以使用它将模型和缩放器对象(或多个对象)保存到自己的项目中。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 邮件

*   [用 Sklearn 保存并加载 Python 中的机器学习模型](https://machinelearningmastery.com/save-load-machine-learning-models-python-Sklearn/)
*   [如何训练最终的机器学习模型](https://machinelearningmastery.com/train-final-machine-learning-model/)

### 蜜蜂

*   [sklearn . dataset . make _ blobs API](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)。
*   [sklearn . model _ selection . train _ test _ split API](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)。
*   [硬化。预处理。MinMaxScaler API](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) 。
*   [sklearn . metrics . accuracy _ score API](https://Sklearn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)。
*   [sklearn.linear_model。物流配送应用编程接口](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。
*   [泡菜 API](https://docs.python.org/3/library/pickle.html) 。

## 摘要

在本教程中，您发现了如何将模型和数据准备对象保存到文件中供以后使用。

具体来说，您了解到:

*   为机器学习模型正确准备测试数据和新数据的挑战。
*   将模型和数据准备对象保存到文件中以备后用的解决方案。
*   如何在新数据上保存和后期加载并使用机器学习模型和数据准备模型。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。