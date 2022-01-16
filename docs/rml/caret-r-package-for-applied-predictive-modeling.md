# 用于应用预测建模的 Caret 包

> 原文：<https://machinelearningmastery.com/caret-r-package-for-applied-predictive-modeling/>

最后更新于 2019 年 8 月 22 日

统计计算的 R 平台可能是应用机器学习最流行和最强大的平台。

R 中的脱字号套餐被称为“ *R 的竞争优势*”。它使得在 R 中训练、调整和评估机器学习模型的过程一致、简单甚至有趣。

在这篇文章中，你会发现 R 中的 caret 包，它的关键特性和去哪里了解更多。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

[![Caret package in R](img/3536c9556d60967e606a9a2b78ffeeed.png)](https://machinelearningmastery.com/wp-content/uploads/2014/09/Caret-package-in-R.png)

R 中的 Caret 包

## 什么是 Caret 包

caret 建立在机器学习的一个关键哲学之上，即“没有免费的午餐”定理。该定理指出，在没有预测问题的先验知识的情况下，没有任何一种方法可以说比其他方法更好。

面对这个定理，caret 包对如何进行应用机器学习持固执己见的立场。对于给定的问题，你无法知道哪种算法或哪种算法参数是最优的，只能通过经验实验来知道。这是 caret 包旨在促进的过程。

它通过几个关键方式做到这一点:

*   **流线型模型创建**:提供一致的界面，训练大量 r 中最流行的第三方算法。
*   **评估参数对表现的影响**:它提供了一些工具，可以根据一个客观的衡量标准对算法参数的组合进行网格搜索，以了解给定问题的参数对模型的影响。
*   **选择最佳模型**:它提供了评估和比较给定问题的模型的工具，以使用客观标准定位最合适的模型。
*   **评估模型表现**:它提供了根据给定问题的未知数据来评估模型准确性的工具。

## Caret 特征

caret 包有许多围绕核心理念构建的特性。一些例子包括:

*   **数据拆分**:在训练和测试数据集中拆分数据。
*   **数据预处理**:准备数据进行标准化、规范化等建模。
*   **特征选择**:只选择有效预测所需属性的方法。
*   **特征重要性**:评估数据集中每个属性对预测属性的相关性。
*   **模型调整**:评估计法参数对表现的影响，找到最佳配置
*   **并行处理**:使用并行计算(如工作站上的多个内核)来调整和估计模型表现，以提高表现。
*   **可视化**:通过量身定制的可视化，更好地理解训练数据、模型比较以及参数对模型的影响。

## Caret 从何而来

caret 是由辉瑞公司的马克斯·库恩创建和维护的 R 包。开发始于 2005 年，后来被开源并上传到 CRAN。

caret 实际上是一个首字母缩略词，代表分类和回归训练。

它最初是出于对给定问题运行多种不同算法的需要而开发的。r 包是由第三方创建的，在训练和生成预测时，它们的参数和语法可能会有所不同。caret 包的最初版本旨在统一模型训练和预测。

它后来扩展到进一步标准化相关的常见任务，如参数调整和确定变量重要性。

## 马克斯·库恩访谈

马克斯·库恩接受数据科学采访。洛杉矶在用户大会上。在采访中，Max 谈到了 caret 的发展和他对 r 的使用，他谈到了在给定问题上测试多个模型的重要性和同时使用多个不同包的痛苦，以及创建包的动力。

<iframe loading="lazy" title="Max Kuhn Interviewed by DataScience.LA at useR" width="500" height="281" src="about:blank" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen="" data-rocket-lazyload="fitvidscompatible" data-lazy-src="https://www.youtube.com/embed/YmHyAHkjX_A?feature=oembed"><iframe title="Max Kuhn Interviewed by DataScience.LA at useR" width="500" height="281" src="https://www.youtube.com/embed/YmHyAHkjX_A?feature=oembed" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen=""/></div> <p/> <h2>马克斯·库恩对 caret 的演示</h2> <p>马克斯·库恩演示了 caret，并在本次演示中介绍了 caret 的发展和特点。他再次谈到了“没有免费午餐”定理和测试多个模型的必要性。演示的核心是一些流失数据的模型示例。他涉及到评估模型表现、算法调整等等。</p> <p><span class="1KTublOcYeJaAsW"/></p> <div class="responsive-video"><iframe loading="lazy" title="caret package webinar" width="500" height="375" src="about:blank" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen="" data-rocket-lazyload="fitvidscompatible" data-lazy-src="https://www.youtube.com/embed/7Jbb2ItbTC4?feature=oembed"/><iframe title="caret package webinar" width="500" height="375" src="https://www.youtube.com/embed/7Jbb2ItbTC4?feature=oembed" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen=""/></div> <p/> <h2>Caret 资源</h2> <p>如果您对 caret 包中的更多信息感兴趣，请查看下面的一些链接。</p> <ul> <li><a href="https://topepo.github.io/caret/index.html">Caret 包主页</a></li> <li><a href="https://cran.r-project.org/web/packages/caret/">CRAN 上的 caret 包</a></li> <li><a href="https://cran.r-project.org/web/packages/caret/caret.pdf">脱字号包装手册</a> (PDF，所有功能)</li> <li><a href="https://cran.r-project.org/web/packages/caret/vignettes/caret.pdf">Caret 包简介</a></li> <li><a href="http://www.jstatsoft.org/v28/i05">使用 Caret 包</a>在 R 中构建预测模型(PDF 论文)</li> <li><a href="https://github.com/topepo/caret">GitHub 上的开源项目</a>(源代码)</li> </ul> <!-- Shortcode does not match the conditions --> <p/> </body></html></iframe>