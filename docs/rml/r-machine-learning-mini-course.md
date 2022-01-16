# R 机器学习迷你课程

> 原文：<https://machinelearningmastery.com/r-machine-learning-mini-course/>

最后更新于 2019 年 8 月 22 日

### *14 天从开发者到机器学习实践者*

在这个迷你课程中，您将发现如何在 14 天内使用 R 开始、构建准确的模型并自信地完成预测建模机器学习项目。

这是一个又大又重要的岗位。你可能想把它做成书签。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

## 这个迷你课程是给谁的？

在我们开始之前，让我们确保你在正确的地方。下面的列表提供了一些关于本课程是为谁设计的一般指南。

不要惊慌，如果你没有完全匹配这些点，你可能只需要在一个或另一个领域刷起来就能跟上。

*   **懂得写一点代码的开发者**。这意味着一旦你知道了基本的语法，再去学一门像 R 这样的新编程语言也不是什么大不了的事情。这并不意味着你是一个向导式的程序员，只是意味着你可以不费吹灰之力就能掌握一门基本的 C 语言。
*   **懂一点机器学习的开发者**。这意味着你知道机器学习的基础知识，比如交叉验证、一些算法和偏差方差权衡。这并不意味着你是一个机器学习博士，只是你知道地标或知道在哪里可以找到它们。

这门迷你课程既不是关于 R 的教科书，也不是关于机器学习的教科书。

它会带你从一个懂一点机器学习的开发者，变成一个能用 R 这个最强大、最受欢迎的机器学习平台得到结果的开发者。

## 迷你课程概述(期待什么)

这门迷你课程分为 14 节课，我称之为“日子”。

你可以每天完成一节课(推荐)或者一天内完成所有的课(硬核！).这真的取决于你有多少时间和你的热情程度。

以下是 14 节课，将让你开始在 R:

*   **第 1 天**:下载并安装 r
*   **第二天**:用基本语法在 R 中走动。
*   **第 3 天**:加载数据和标准机器学习数据集。
*   **第 4 天**:用描述性统计理解数据。
*   **第 5 天**:用可视化理解数据。
*   **第 6 天**:通过预处理数据为建模做准备。
*   **第 7 天**:采用重采样方法的算法评估。
*   **第 8 天**:算法评估指标。
*   **第 9 天**:抽查算法。
*   **第 10 天**:车型对比选择。
*   **第 11 天**:通过算法调整提高准确率。
*   **第 12 天**:通过集合预测提高准确性。
*   **第 13 天**:完成并保存您的模型。
*   **第 14 天**:你好世界端到端项目。

每节课可能需要你 60 秒或 30 分钟。慢慢来，按照自己的节奏完成课程。提问，甚至在下面的评论中发布结果。

这些课程期望你去发现如何做事。我会给你提示，但每节课的一部分要点是强迫你学会去哪里寻找关于 R 平台的帮助(提示，我所有的答案都直接在这个博客上，使用搜索)。

我确实在早期课程中提供了更多的帮助，因为我想让你建立一些信心和惰性。坚持住，不要放弃！

## 第一天:下载并安装 R

在进入该平台之前，您无法在 R 中开始机器学习。

今天的课很简单，你必须在你的电脑上下载并安装 R 平台。

1.  访问 [R 主页](https://www.r-project.org/)下载你的操作系统(Linux、OS X 或 Windows)的 R。
2.  在计算机上安装 R。
3.  通过键入“R”，第一次从命令行启动 R。

如果您需要安装 R 的帮助，请查看帖子:

*   [使用 R 进行机器学习](https://machinelearningmastery.com/use-r-for-machine-learning/)

## 第二天:用基本语法在休息时间四处走动

你需要能够读写基本的 R 脚本。

作为一名开发人员，你可以很快学会新的编程语言。r 区分大小写，对注释使用哈希(#)，对赋值使用箭头运算符(

今天的任务是在 R 交互环境中练习 R 编程语言的基本语法。

1.  用箭头运算符(
2.  练习使用基本数据结构，如向量、列表和数据框。
3.  练习使用流控制结构，如 If-Then-Else 和循环。
4.  练习调用函数、安装和加载包。

例如，下面是一个创建数字列表并计算平均值的示例。

```py
numbers <- c(1, 2, 3, 4, 5, 6)
mean(numbers)
```

如果你需要基本的 R 语法方面的帮助，请看帖子:

*   [R](https://machinelearningmastery.com/r-crash-course-for-developers/)中的超快速成班。

## 第 3 天:加载数据和标准机器学习数据集

机器学习算法需要数据。您可以从 CSV 文件中加载自己的数据，但是当您在 R 中开始机器学习时，您应该在标准机器学习数据集上练习。

今天课程的任务是轻松地将数据加载到 R 中，并查找和加载标准机器学习数据集。

R 自带的*数据集*包有很多标准数据集，包括著名的鸢尾花数据集。 *mlbench* 包还包含标准的机器学习数据集。

1.  练习使用 *read.csv()* 功能将 CSV 文件加载到 R 中。
2.  练习从数据集和 *mlbench* 包加载标准机器学习数据集。

**帮助**:输入*可以获得某个功能的帮助？函数名*或者调用*帮助()*函数，并传递需要帮助的函数名作为参数。

为了让您开始，下面的代码片段将安装并加载 *mlbench* 包，列出它提供的所有数据集，并将 PimaIndiansDiabetes 数据集附加到您的环境中供您使用。

```py
install.packages("mlbench")
library(mlbench)
data(package="mlbench")
data(PimaIndiansDiabetes)
head(PimaIndiansDiabetes)
```

干得好，走到这一步！**坚持住**。

到目前为止还有问题吗？在评论中提问。

## 第 4 天:用描述性统计数据理解数据

一旦你把数据加载到 R 中，你需要能够理解它。

你越能理解你的数据，你能建立的模型就越好、越准确。理解数据的第一步是使用描述性统计。

今天你的课程是学习如何使用描述性统计来理解你的数据。

1.  使用 *head()* 函数查看前几行，了解您的数据。
2.  使用 *dim()* 功能查看数据的尺寸。
3.  使用*汇总()*功能查看您的数据分布。
4.  使用 *cor()* 函数计算变量之间的成对相关性。

以下示例加载 iris 数据集并总结每个属性的分布。

```py
data(iris)
summary(iris)
```

试试看！

## 第 5 天:通过可视化理解数据

从昨天的课程继续，你必须花时间更好地理解你的数据。

提高对数据理解的第二种方法是使用数据可视化技术(例如绘图)。

今天，您的课程是学习如何使用 R 中的绘图来单独理解属性及其交互。

1.  使用 *hist()* 函数创建每个属性的直方图。
2.  使用*方框图()*功能创建每个属性的方框图和触须图。
3.  使用 *pairs()* 功能创建所有属性的成对散点图。

例如，下面的代码片段将加载 iris 数据集，并创建数据集的散点图矩阵。

```py
data(iris)
pairs(iris)
```

## 第 6 天:通过预处理数据为建模做准备

您的原始数据可能没有设置为建模的最佳状态。

有时，您需要预处理数据，以便向建模算法最好地呈现数据中问题的固有结构。在今天的课程中，您将使用 caret 包提供的预处理功能。

Caret 包提供了*预处理()*函数，该函数使用方法参数来指示要执行的预处理类型。从数据集中准备好预处理参数后，可以对您可能拥有的每个数据集应用相同的预处理步骤。

请记住，您可以按如下方式安装和加载 Caret 包:

```py
install.packages("caret")
library(caret)
```

1.  使用*刻度*和*中心*选项标准化数值数据(例如，0 的平均值和 1 的标准偏差)。
2.  使用*范围*选项标准化数值数据(例如 0-1 的范围)。
3.  使用 *BoxCox* 选项探索更高级的电源转换，如 Box-Cox 电源转换。

例如，下面的代码片段加载 iris 数据集，计算标准化数据所需的参数，然后创建数据的标准化副本。

```py
# load caret package
library(caret)
# load the dataset
data(iris)
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(iris[,1:4], method=c("range"))
# transform the dataset using the pre-processing parameters
transformed <- predict(preprocessParams, iris[,1:4])
# summarize the transformed dataset
summary(transformed)
```

## 第 7 天:使用重采样方法的算法评估

用于训练机器学习算法的数据集称为训练数据集。用于训练算法的数据集不能用于为您提供新数据模型准确性的可靠估计。这是一个大问题，因为创建模型的整个想法是对新数据进行预测。

您可以使用称为重采样方法的统计方法将训练数据集划分为子集，一些子集用于训练模型，另一些子集被保留下来，用于根据看不见的数据估计模型的准确性。

今天课程的目标是练习使用 caret 包中可用的不同重采样方法。在 r 中查找 *createDataPartition()* 、 *trainControl()* 和 *train()* 函数的帮助

1.  将数据集拆分为训练集和测试集。
2.  使用 k 倍交叉验证估计算法的准确性。
3.  使用重复的 k 倍交叉验证来估计算法的准确性。

下面的代码片段使用 Caret 包，使用 10 倍交叉验证来估计朴素贝叶斯算法在虹膜数据集上的准确性。

```py
# load the library
library(caret)
# load the iris dataset
data(iris)
# define training control
trainControl <- trainControl(method="cv", number=10)
# estimate the accuracy of Naive Bayes on the dataset
fit <- train(Species~., data=iris, trControl=trainControl, method="nb")
# summarize the estimated accuracy
print(fit)
```

这一步需要更多帮助吗？

看看关于重采样方法的帖子:

*   [如何使用 caret 包](https://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/)估计 R 中的模型准确率。

你意识到这是中途点了吗？干得好！

## 第 8 天:算法评估指标

有许多不同的度量可以用来评估数据集上机器学习算法的技能。

您可以在 *train()* 函数中的 Caret 中指定用于测试线束的度量，默认值可用于回归和分类问题。今天课程的目标是练习使用 caret 包中可用的不同算法表现指标。

1.  练习在分类问题(例如虹膜数据集)上使用*准确度*和*卡帕*度量。
2.  练习在回归问题上使用 *RMSE* 和 *RSquared* 度量(例如 longley 数据集)。
3.  练习在二进制分类问题上使用 *ROC* 度量(例如， *mlbench* 包中的 PimaIndiansDiabetes 数据集)。

下面的代码片段演示了在 iris 数据集上计算 LogLoss 度量。

```py
# load caret library
library(caret)
# load the iris dataset
data(iris)
# prepare 5-fold cross validation and keep the class probabilities
control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=mnLogLoss)
# estimate accuracy using LogLoss of the CART algorithm
fit <- train(Species~., data=iris, method="rpart", metric="logLoss", trControl=control)
# display results
print(fit)
```

## 第 9 天:抽查算法

你不可能事先知道哪种算法对你的数据效果最好。

你必须通过反复试验来发现它。我称之为抽查算法。caret 包提供了许多机器学习算法和工具的接口，以比较这些算法的估计准确率。

在本课中，您必须练习抽查不同的机器学习算法。

1.  抽查数据集上的线性算法(例如线性回归、逻辑回归和线性判别分析)。
2.  抽查数据集上的一些非线性算法(例如 KNN、SVM 和 CART)。
3.  抽查数据集上的一些复杂集成算法(例如随机森林和随机梯度提升)。

**帮助**:您可以通过键入:*名称(getmodeinfo())*获得可以在 Caret 中使用的模型列表

例如，下面的片段抽查了来自 *mlbench* 包的 Pima Indians 糖尿病数据集上的两个线性算法。

```py
# load libraries
library(caret)
library(mlbench)
# load the Pima Indians Diabetes dataset
data(PimaIndiansDiabetes)
# prepare 10-fold cross validation
trainControl <- trainControl(method="cv", number=10)
# estimate accuracy of logistic regression
set.seed(7)
fit.lr <- train(diabetes~., data=PimaIndiansDiabetes, method="glm", trControl=trainControl)
# estimate accuracy of linear discriminate analysis
set.seed(7)
fit.lda <- train(diabetes~., data=PimaIndiansDiabetes, method="lda", trControl=trainControl)
# collect resampling statistics
results <- resamples(list(LR=fit.lr, LDA=fit.lda))
# summarize results
summary(results)
```

## 第 10 天:模型比较和选择

现在，您已经知道如何在数据集上抽查机器学习算法，您需要知道如何比较不同算法的估计表现，并选择最佳模型。

谢天谢地，caret 包提供了一套工具来绘制和总结模型之间的表现差异。

在今天的课程中，您将练习在 r

1.  使用 *summary()* caret 函数创建结果表(提示，上一课有一个例子)
2.  使用*点绘图()*Caret 函数比较结果。
3.  使用 *bwplot()* Caret 函数比较结果。
4.  使用 *diff()* Caret 函数计算结果之间的统计显著性。

下面的代码片段扩展了昨天的示例，并创建了抽查结果的图表。

```py
# load libraries
library(caret)
library(mlbench)
# load the Pima Indians Diabetes dataset
data(PimaIndiansDiabetes)
# prepare 10-fold cross validation
trainControl <- trainControl(method="cv", number=10)
# estimate accuracy of logistic regression
set.seed(7)
fit.lr <- train(diabetes~., data=PimaIndiansDiabetes, method="glm", trControl=trainControl)
# estimate accuracy of linear discriminate analysis
set.seed(7)
fit.lda <- train(diabetes~., data=PimaIndiansDiabetes, method="lda", trControl=trainControl)
# collect resampling statistics
results <- resamples(list(LR=fit.lr, LDA=fit.lda))
# plot the results
dotplot(results)
```

## 第 11 天:通过算法调整提高准确性

一旦您找到一两个在数据集上表现良好的算法，您可能希望提高这些模型的表现。

提高算法表现的一种方法是根据您的特定数据集调整其参数。

caret 包提供了三种方法来搜索机器学习算法的参数组合。你今天课的目标是练习每一个。

1.  自动调整算法的参数(例如参见*列车()*的*调整长度*参数)。
2.  使用您指定的网格搜索来调整算法的参数。
3.  使用随机搜索调整算法的参数。

查看*列车控制()*和*列车()*功能的帮助，注意*方法*和*调谐网格*参数。

下面使用的代码片段是在 iris 数据集上使用网格搜索随机森林算法的示例。

```py
# load the library
library(caret)
# load the iris dataset
data(iris)
# define training control
trainControl <- trainControl(method="cv", number=10)
# define a grid of parameters to search for random forest
grid <- expand.grid(.mtry=c(1,2,3,4,5,6,7,8,10))
# estimate the accuracy of Random Forest on the dataset
fit <- train(Species~., data=iris, trControl=trainControl, tuneGrid=grid, method="rf")
# summarize the estimated accuracy
print(fit)
```

你快完了！再来几堂课。

## 第 12 天:提高集合预测的准确性

提高模型表现的另一种方法是组合多个模型的预测。

一些模型内置了这种功能，例如随机森林用于装袋，随机梯度提升用于提升。另一种称为堆叠(或混合)的集合类型可以学习如何最好地组合来自多个模型的预测，并在软件包*Carestensemble*中提供。

在今天的课程中，你将练习使用集成法。

1.  用 caret 中的随机森林和装袋 CART 算法练习装袋集成。
2.  在 caret 中使用梯度提升机和 C5.0 算法练习提升集成。
3.  使用*carestensemble*包和 *caretStack()* 功能练习堆叠集成。

下面的代码片段演示了如何使用堆叠来组合来自多个模型的预测。

```py
# Load packages
library(mlbench)
library(caret)
library(caretEnsemble)
# load the Pima Indians Diabetes dataset
data(PimaIndiansDiabetes)
# create sub-models
trainControl <- trainControl(method="cv", number=5, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('knn', 'glm')
set.seed(7)
models <- caretList(diabetes~., data=PimaIndiansDiabetes, trControl=trainControl, methodList=algorithmList)
print(models)
# learn how to best combine the predictions
stackControl <- trainControl(method="cv", number=5, savePredictions=TRUE, classProbs=TRUE)
set.seed(7)
stack.glm <- caretStack(models, method="glm", trControl=stackControl)
print(stack.glm)
```

## 第 13 天:完成并保存您的模型

一旦你在你的机器学习问题上找到了一个表现良好的模型，你就需要完成它。

在今天的课程中，您将练习与最终确定模型相关的任务。

1.  练习使用 *predict()* 函数，通过使用 Caret 训练的模型进行预测。
2.  练习训练表现良好的模型的独立版本。
3.  使用 *saveRDS()* 和 *readRDS()* 功能练习将训练好的模型保存到文件中并再次加载。

例如，下面的代码片段显示了如何在整个数据集上创建一个随机的森林算法。

```py
# load package
library(randomForest)
# load iris data
data(iris)
# train random forest model
finalModel <- randomForest(Species~., iris, mtry=2, ntree=2000)
# display the details of the final model
print(finalModel)
```

## 第 14 天:你好世界端到端项目

您现在知道如何完成预测建模机器学习问题的每一项任务。

在今天的课程中，您需要练习将各个部分组合在一起，并通过标准的机器学习数据集进行端到端的工作。

1.  端到端地处理 iris 数据集(机器学习的 hello 世界)

这包括以下步骤:

1.  使用描述性统计和可视化来理解您的数据。
2.  对数据进行预处理，以最好地揭示问题的结构。
3.  使用您自己的测试工具抽查一些算法。
4.  使用算法参数调整改善结果。
5.  使用集成方法改进结果。
6.  最终确定模型以备将来使用。

## 末日！(看看你已经走了多远)

你成功了。干得好！

花一点时间，回头看看你已经走了多远。

*   你开始时对机器学习很感兴趣，并强烈希望能够使用 r 练习和应用机器学习。
*   你下载、安装并启动了 R，也许是第一次，并开始熟悉语言的语法。
*   在大量的课程中，您慢慢地、稳定地了解了预测建模机器学习项目的标准任务是如何映射到 R 平台上的。
*   根据常见机器学习任务的秘籍，您使用 r 端到端地解决了第一个机器学习问题
*   使用一个标准模板，你已经收集的秘籍和经验，你现在能够自己解决新的和不同的预测建模机器学习问题。

不要轻视这一点，你在短时间内已经取得了很大的进步。

这只是你用 r 进行机器学习旅程的开始，继续练习和发展你的技能。

### 你觉得迷你课程怎么样？

你喜欢这个迷你课程吗？

你有什么问题吗？有什么症结吗？

让我知道。请在下面留言。