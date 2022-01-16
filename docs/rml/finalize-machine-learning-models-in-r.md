# 在 R 中保存并最终确定您的机器学习模型

> 原文：<https://machinelearningmastery.com/finalize-machine-learning-models-in-r/>

最后更新于 2019 年 8 月 22 日

找到一个准确的机器学习并不是项目的终点。

在这篇文章中，你将发现如何在 R 中最终确定你的机器学习模型，包括:对看不见的数据进行预测，从零开始重建模型，并保存模型以备后用。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

![Finalize Your Machine Learning Model in R](img/fb2f7abfda5e21f87c090240d3f9d7b7.png)

将你的机器学习模型最终确定在 R.
图片由[克里斯蒂安·施奈特尔克](https://www.flickr.com/photos/manoftaste-de/9483602817/)提供，保留部分权利。

## 最终确定您的机器学习模型

一旦你的测试工具上有了一个准确的模型，你就差不多了。但还没有。

要最终确定您的模型，还有许多任务要做。为数据集创建精确模型的整个想法是对看不见的数据进行预测。

您可能会关注三项任务:

1.  根据看不见的数据做出新的预测。
2.  使用所有训练数据创建独立模型。
3.  将模型保存到文件中，以便以后加载和预测新数据。

一旦你确定了你的模型，你就可以使用它了。你可以直接使用 R 模型。您还可以发现学习算法找到的关键内部表示(像线性模型中的系数)，并在另一个平台上的预测算法的新实现中使用它们。

在下一节中，您将了解如何在 r 中完成您的机器学习模型。

## 最终确定预测模型

caret 是一个很好的工具，你可以用它来找到好的甚至最好的机器学习算法和机器学习算法的参数。

但是当你发现了一个足够精确可以使用的模型后，你会怎么做呢？

一旦你在 R 中找到了一个好的模型，你有三个主要的关注点:

1.  使用您调整后的 Caret 模型进行新的预测。
2.  使用整个训练数据集创建独立模型。
3.  将独立模型保存/加载到文件。

本节将向您介绍如何在 r

### 1.对新数据进行预测

您可以使用您使用 caret 使用 *predict.train()* 函数调整的模型进行新的预测。

在下面的配方中，数据集被分为验证数据集和训练数据集。验证数据集也可以是存储在单独文件中并作为数据框加载的新数据集。

使用线性判别分析找到了数据的良好模型。我们可以看到，Caret 提供了对 finalModel 变量中的训练运行的最佳模型的访问。

我们可以使用该模型进行预测，方法是使用自动使用最终模型的训练拟合调用预测。我们必须通过 *newdata* 参数指定要进行预测的数据。

```py
# load libraries
library(caret)
library(mlbench)
# load dataset
data(PimaIndiansDiabetes)
# create 80%/20% for training and validation datasets
set.seed(9)
validation_index <- createDataPartition(PimaIndiansDiabetes$diabetes, p=0.80, list=FALSE)
validation <- PimaIndiansDiabetes[-validation_index,]
training <- PimaIndiansDiabetes[validation_index,]
# train a model and summarize model
set.seed(9)
control <- trainControl(method="cv", number=10)
fit.lda <- train(diabetes~., data=training, method="lda", metric="Accuracy", trControl=control)
print(fit.lda)
print(fit.lda$finalModel)
# estimate skill on validation dataset
set.seed(9)
predictions <- predict(fit.lda, newdata=validation)
confusionMatrix(predictions, validation$diabetes)
```

运行该示例，我们可以看到训练数据集上的估计准确率为 76.91%。在拟合中使用 finalModel，我们可以看到搁置验证数据集的准确率为 77.78%，与我们的估计非常相似。

```py
Resampling results

  Accuracy   Kappa    Accuracy SD  Kappa SD 
  0.7691169  0.45993  0.06210884   0.1537133

...

Confusion Matrix and Statistics

          Reference
Prediction neg pos
       neg  85  19
       pos  15  34

               Accuracy : 0.7778          
                 95% CI : (0.7036, 0.8409)
    No Information Rate : 0.6536          
    P-Value [Acc > NIR] : 0.000586        

                  Kappa : 0.5004          
 Mcnemar's Test P-Value : 0.606905        

            Sensitivity : 0.8500          
            Specificity : 0.6415          
         Pos Pred Value : 0.8173          
         Neg Pred Value : 0.6939          
             Prevalence : 0.6536          
         Detection Rate : 0.5556          
   Detection Prevalence : 0.6797          
      Balanced Accuracy : 0.7458          

       'Positive' Class : neg
```

### 2.创建独立模型

在这个例子中，我们为设置为 2000 的 *mtry* 和 *ntree* 调整了一个随机森林，其中有 3 个不同的值。通过打印拟合和最终模型，我们可以看到 *mtry* 的最准确值是 2。

现在我们知道了一个好的算法(随机森林)和好的配置(mtry=2， *ntree=2000* )我们可以直接使用所有的训练数据创建最终的模型。我们可以在[Caret 模型列表](https://topepo.github.io/caret/modelList.html)中查找 Caret 使用的“ *rf* ”随机森林实现，注意它使用的是 *randomForest* 包以及 *randomForest()* 函数。

该示例直接创建一个新模型，并使用它对新数据进行预测，这种情况下模拟为验证数据集。

```py
# load libraries
library(caret)
library(mlbench)
library(randomForest)
# load dataset
data(Sonar)
set.seed(7)
# create 80%/20% for training and validation datasets
validation_index <- createDataPartition(Sonar$Class, p=0.80, list=FALSE)
validation <- Sonar[-validation_index,]
training <- Sonar[validation_index,]
# train a model and summarize model
set.seed(7)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
fit.rf <- train(Class~., data=training, method="rf", metric="Accuracy", trControl=control, ntree=2000)
print(fit.rf)
print(fit.rf$finalModel)
# create standalone model using all training data
set.seed(7)
finalModel <- randomForest(Class~., training, mtry=2, ntree=2000)
# make a predictions on "new data" using the final model
final_predictions <- predict(finalModel, validation[,1:60])
confusionMatrix(final_predictions, validation$Class)
```

我们可以看到，最优配置的估计准确率为 85.07%。我们可以看到，在所有训练数据集上训练的最终独立模型和对验证数据集的预测的准确率为 82.93%。

```py
Random Forest 

167 samples
 60 predictor
  2 classes: 'M', 'R' 

No pre-processing
Resampling: Cross-Validated (10 fold, repeated 3 times) 
Summary of sample sizes: 151, 150, 150, 150, 151, 150, ... 
Resampling results across tuning parameters:

  mtry  Accuracy   Kappa      Accuracy SD  Kappa SD 
   2    0.8507353  0.6968343  0.07745360   0.1579125
  31    0.8064951  0.6085348  0.09373438   0.1904946
  60    0.7927696  0.5813335  0.08768147   0.1780100

Accuracy was used to select the optimal model using  the largest value.
The final value used for the model was mtry = 2\. 

...

Call:
 randomForest(x = x, y = y, ntree = 2000, mtry = param$mtry) 
               Type of random forest: classification
                     Number of trees: 2000
No. of variables tried at each split: 2

        OOB estimate of  error rate: 14.37%
Confusion matrix:
   M  R class.error
M 83  6  0.06741573
R 18 60  0.23076923

...

Confusion Matrix and Statistics

          Reference
Prediction  M  R
         M 20  5
         R  2 14

               Accuracy : 0.8293          
                 95% CI : (0.6794, 0.9285)
    No Information Rate : 0.5366          
    P-Value [Acc > NIR] : 8.511e-05       

                  Kappa : 0.653           
 Mcnemar's Test P-Value : 0.4497          

            Sensitivity : 0.9091          
            Specificity : 0.7368          
         Pos Pred Value : 0.8000          
         Neg Pred Value : 0.8750          
             Prevalence : 0.5366          
         Detection Rate : 0.4878          
   Detection Prevalence : 0.6098          
      Balanced Accuracy : 0.8230          

       'Positive' Class : M
```

一些更简单的模型，如线性模型，可以输出它们的系数。这很有用，因为通过这些，您可以用您选择的语言实现简单的预测过程，并使用系数来获得相同的准确率。随着表示的复杂性增加，这变得更加困难。

### 3.保存并加载您的模型

您可以将最佳模型保存到文件中，以便以后加载它们并进行预测。

在本例中，我们将声纳数据集分为训练数据集和验证数据集。我们将验证数据集作为新数据来测试最终模型。我们使用训练数据集和我们的最佳参数训练最终模型，然后将其保存到本地工作目录中名为 final_model.rds 的文件中。

模型是序列化的。可以在以后通过调用 readRDS()并将加载的对象(在本例中是随机的林拟合)分配给变量名来加载它。然后，加载的随机森林用于对新数据进行预测，在本例中是验证数据集。

```py
# load libraries
library(caret)
library(mlbench)
library(randomForest)
library(doMC)
registerDoMC(cores=8)
# load dataset
data(Sonar)
set.seed(7)
# create 80%/20% for training and validation datasets
validation_index <- createDataPartition(Sonar$Class, p=0.80, list=FALSE)
validation <- Sonar[-validation_index,]
training <- Sonar[validation_index,]
# create final standalone model using all training data
set.seed(7)
final_model <- randomForest(Class~., training, mtry=2, ntree=2000)
# save the model to disk
saveRDS(final_model, "./final_model.rds")

# later...

# load the model
super_model <- readRDS("./final_model.rds")
print(super_model)
# make a predictions on "new data" using the final model
final_predictions <- predict(super_model, validation[,1:60])
confusionMatrix(final_predictions, validation$Class)
```

我们可以看到验证数据集的准确率为 82.93%。

```py
Confusion Matrix and Statistics

          Reference
Prediction  M  R
         M 20  5
         R  2 14

               Accuracy : 0.8293          
                 95% CI : (0.6794, 0.9285)
    No Information Rate : 0.5366          
    P-Value [Acc > NIR] : 8.511e-05       

                  Kappa : 0.653           
 Mcnemar's Test P-Value : 0.4497          

            Sensitivity : 0.9091          
            Specificity : 0.7368          
         Pos Pred Value : 0.8000          
         Neg Pred Value : 0.8750          
             Prevalence : 0.5366          
         Detection Rate : 0.4878          
   Detection Prevalence : 0.6098          
      Balanced Accuracy : 0.8230          

       'Positive' Class : M
```

## 摘要

在这篇文章中，你发现了使用最终预测模型的三种方法:

1.  如何使用脱字号优化的最佳模型进行预测？
2.  如何使用 Caret 调优过程中找到的参数创建独立模型。
3.  如何保存和稍后加载独立模型并使用它进行预测。

你可以通过这些秘籍来更好地理解它们。您也可以将它们用作模板，并将其复制粘贴到当前或下一个机器学习项目中。

## 下一步

你试过这些秘籍吗？

1.  开始你的互动环境。
2.  键入或复制粘贴上面的秘籍并试用。
3.  使用 R 中的内置帮助来了解有关所用函数的更多信息。

你有问题吗？在评论里问，我会尽力回答。