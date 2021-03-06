# 如何在 R 中开始机器学习（一个周末内获得结果）

> 原文：<https://machinelearningmastery.com/get-started-in-machine-learning-with-r/>

最后更新于 2019 年 12 月 13 日

如何在 R 中开始机器学习？

r 是一个庞大而复杂的平台。它也是世界上最好的数据科学家最受欢迎的平台。

在这篇文章中，您将发现一步一步的过程，您可以使用该过程开始在 R 平台上使用机器学习进行预测建模。

这些步骤既实用又简单，你可以在一个周末后建立准确的预测模型。

这个过程确实假设你是一个开发人员，知道一点机器学习，并且实际上会做工作，但是这个过程确实交付了结果。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

![How To Get Started With Machine Learning in R](img/f66ab222b56f4ec84107e33f593a9713.png)

如何在 R
中开始机器学习图片由[sebastian ter Burg](https://www.flickr.com/photos/ter-burg/17315216959/)提供，保留部分权利。

## 以错误的方式学习

以下是我认为你不应该在 r 学习机器学习的方法

*   **第一步**:真正精通 R 编程和 R 语法。
*   **第二步**:了解你可以在 r 中使用的每一种可能算法的深层理论。
*   **第三步**:详细学习如何在 r 中使用各个机器学习算法。
*   **第四步**:只轻轻一碰如何评估车型。

我认为这是错误的方式。

*   它告诉你，你需要花所有的时间学习如何使用单独的机器学习算法。
*   它没有教你在 R 中建立预测性机器学习模型的过程，你实际上可以在实践中使用它来进行预测。

可悲的是，这是我在几乎所有关于这个主题的书籍和在线课程中看到的用 R 教授机器学习的方法。

你不想成为 R 中的坏蛋，甚至不想成为 R 中的机器学习算法的坏蛋。你想成为使用 R 构建精确预测模型的坏蛋。这就是背景。

你可以花时间非常详细地学习单个机器学习算法，只要它能帮助你更可靠地建立更精确的预测模型。

## 良好的机器学习背景

你可以直接进入休息区，去吧。

不过在我看来，我认为如果你有一些背景，你会从中获得更多。

r 是一个高级平台，作为初学者你可以从中获得很多。但是，如果你有一点机器学习和一点编程作为基础，R 将很快成为建立精确预测模型的超级大国。

### 一般建议

这里有一些建议，可以让你在 r 中最大限度地开始机器学习。我认为这些对一个对机器学习感兴趣的现代开发人员来说是合理的。

**懂得编程的开发者**。这很有帮助，因为学习 R 的语法并不是什么大不了的事情，有时候会有点奇怪。知道由谁来创建脚本或脚本小程序(迷你脚本)来完成这个或那个任务也很有帮助。r 毕竟是一种编程语言。

**对预测建模机器学习感兴趣**。机器学习是一个涵盖各种有趣算法的大领域。预测建模是一个子集，它只关注构建对新数据进行预测的模型。没有解释数据之间的关系，也没有从一般数据中学习。预测建模是 R 作为机器学习平台真正闪耀的地方。

**熟悉机器学习基础知识**。你把机器学习理解为一个归纳问题，在这个问题中，所有的算法实际上只是试图估计一个输入空间到一个输出空间的底层映射函数。通过这个镜头，所有预测性机器学习都是有意义的，搜索好的和最好的机器学习算法、算法参数和数据转换的策略也是如此。

### 具体建议

我在下一节布局的方法也对你的背景做了一些假设。

**你不是机器学习的绝对初学者**。你可以，这种方法可能对你有用，但是如果你有一些额外的建议背景，你会从中获得更多。

**你想用自上而下的方法来学习机器学习**。这是我教的方法，不是从理论和原理开始，如果有时间的话，最终接触实际的机器学习，而是从端到端的项目和研究细节的目标开始，因为你需要它们来提供更好的结果。

**您熟悉预测建模机器学习项目**中的步骤。具体来说:

1.  定义问题
2.  准备数据
3.  评估计法
4.  改善结果
5.  呈现结果

您可以在此了解有关此过程和这些步骤的更多信息:

*   [如何使用机器学习清单可靠地获得准确的预测(即使你是初学者)](https://machinelearningmastery.com/machine-learning-checklist/)
*   [处理机器学习问题的过程](https://machinelearningmastery.com/process-for-working-through-machine-learning-problems/)

**你至少熟悉一些机器学习算法**。或者你可能知道如何快速提取它们，例如使用算法描述模板方法。我认为学习机器学习算法的方式和原因的细节是一项独立的任务，而不是学习如何在像 r .这样的机器学习平台上使用这些算法。在学习的决定因素方面，它们经常被合并在书籍和课程中。

您可以在这里了解更多关于如何使用模板方法学习任何机器学习算法的信息:

*   [如何学习机器学习算法](https://machinelearningmastery.com/how-to-learn-a-machine-learning-algorithm/)
*   [5 种在没有数学背景的情况下理解机器学习算法的技巧](https://machinelearningmastery.com/techniques-to-understand-machine-learning-algorithms-without-the-background-in-mathematics/)

## 如何在 R 中学习机器学习

本节展示了一个过程，您可以使用该过程开始在 R 平台上构建机器学习预测模型。

它分为两部分:

1.  将机器学习项目的任务映射到 R 平台上。
2.  使用标准数据集完成预测建模项目。

### 1.将机器任务映射到 R

你需要知道如何在 R 平台上做机器学习的具体任务。一旦你知道如何使用平台完成一个离散的任务，并可靠地得到一个结果，你就可以在一个又一个项目中一次又一次地完成它。

这个过程很简单:

1.  列出预测建模机器学习项目的所有离散任务。
2.  创建可靠地完成任务的方法，您可以将其复制粘贴作为未来项目的起点。
3.  添加和维护秘籍是你对平台的理解和机器学习的提高。

#### 预测建模任务

以下是您可能希望映射到 R 平台并创建配方的预测建模任务的最小列表。这并不完整，但确实涵盖了平台的大致内容:

1.  R 语法概述
2.  准备数据
    1.  正在加载数据
    2.  使用数据
    3.  数据汇总
    4.  数据可视化
    5.  数据清理
    6.  特征选择
    7.  数据转换
3.  评估计法
    1.  重采样方法
    2.  评估指标
    3.  抽查算法
    4.  型号选择
4.  改善结果
    1.  算法调整
    2.  集成方法
5.  呈现结果
    1.  最终确定模型
    2.  做出新的预测

您会注意到第一个任务是 R 语法的概述。作为一名开发人员，在做任何事情之前，您需要了解该语言的基础知识。例如赋值、数据结构、流控制以及创建和调用函数。

#### 独立配方库

我建议创建独立的秘籍。这意味着每个配方都是一个完整的程序，拥有完成任务和产生输出所需的一切。这意味着您可以将其直接复制到未来的预测建模项目中。

你可以把秘籍存储在一个目录或者 GitHub 上。

### 2.小型预测建模项目

用机器学习来解决常见的预测建模任务是不够的。

同样，这也是大多数书籍和课程停止的地方。他们让你把秘籍拼凑成端到端的项目。

你需要把秘籍拼凑成端到端的项目。这将教你如何使用平台实际交付结果。我建议只使用来自 [UCI 机器学习存储库](https://machinelearningmastery.com/practice-machine-learning-with-small-in-memory-datasets-from-the-uci-machine-learning-repository/)的理解良好的小型机器学习数据集。

这些数据集以 CSV 下载的形式免费提供，大部分可以通过加载第三方库直接在 R 中获得。这些数据集非常适合练习，因为:

1.  它们很小，意味着它们适合内存，算法可以在合理的时间内对它们进行建模。
2.  它们表现良好，这意味着您通常不需要做大量的功能工程来获得良好的结果。
3.  有标准，这意味着许多人以前使用过它们，你可以得到好算法的想法来尝试，以及你应该期望的好结果。

我至少推荐三个项目:

1.  **你好世界项目(鸢尾花)**。这是对项目步骤的快速浏览，无需对广泛用作机器学习的 hello world 的数据集进行太多调整或优化(更多关于[鸢尾花数据集](https://archive.ics.uci.edu/ml/datasets/Iris))。
2.  **二进制端到端分类**。完成二分类问题的每一步(例如[皮马印第安人糖尿病数据集](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names) ( [csv 文件](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv)))。
3.  **端到端回归**。通过回归问题(例如[波士顿住房数据集](https://archive.ics.uci.edu/ml/datasets/Housing))完成流程的每一步。

### 添加和维护配方

带 R 的机器学习并不仅仅局限于通过一些小的标准数据集。你需要接受更多不同的挑战。

*   **标准数据集**:您可以在 UCI 机器学习存储库中的其他标准数据集上练习，克服不同问题类型的挑战。
*   **竞赛数据集**:你可以尝试通过一些更具挑战性的数据集进行工作，比如来自过去的 Kaggle 竞赛或来自过去的 KDDCup 挑战的数据集。
*   **自己的项目**:理想情况下，你需要开始通过自己的项目进行工作。

一直以来，你都在寻求帮助，修改你的脚本，学习如何从机器学习中获得更多

重要的是，你要把这些知识放回你的机器学习秘籍目录中。这将让您在新项目中快速利用这些知识，并极大地提高您开发预测模型的技能和速度。

### 您在此过程中的成果

你可以在一个周末内完成这个过程。到那个周末结束时，你会有秘籍和项目模板，你可以用它们开始用 r 中的机器学习来建模你自己的问题。

您将从一个对 R 上的机器学习感兴趣的开发人员转变为一个有资源和能力使用 R 端到端地处理新数据集并开发要呈现和部署的预测模型的开发人员。

具体来说，你会知道:

*   如何实现 r 中一个预测建模问题的子任务
*   如何在 r 中学习新的不同的子任务？
*   如何获得 r 的帮助？
*   如何端到端地处理中小型数据集？
*   如何交付一个可以对新的未知数据进行预测的模型？

从这里，您可以开始深入了解所使用的功能、技术和算法的细节，目的是学习如何更好地使用它们，以便在更短的时间内提供更准确、更可靠的预测模型。

## 摘要

在这篇文章中，你发现了一个循序渐进的过程，你可以用它来学习和开始在 r。

该流程的三个高级步骤是:

1.  将预测建模过程的步骤映射到 R 平台，并提供您可以重用的配方。
2.  通过小型标准机器学习数据集，将秘籍组合成项目。
3.  处理更多不同的数据集，最好是你自己的，并添加到你的秘籍库中。

你也发现了这个过程背后的哲学，以及为什么这个过程对你来说是最好的过程。

## 下一步

你想用 R 开始机器学习吗？

1.  立即下载并安装 R。
2.  使用上面的过程大纲，把自己限制在一个周末，尽可能地走远。
3.  回电。请留言。我很想听听你怎么样了。

你对这个过程有疑问吗？留言评论，我会尽力回答。