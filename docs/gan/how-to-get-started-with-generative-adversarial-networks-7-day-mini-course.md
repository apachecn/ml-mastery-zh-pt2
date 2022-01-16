# 如何入门生成对抗网络（7 天小型课程）

> 原文：<https://machinelearningmastery.com/how-to-get-started-with-generative-adversarial-networks-7-day-mini-course/>

最后更新于 2019 年 7 月 12 日

#### 生成对抗网络与 Python 速成班。
在 7 天内将生成对抗网络带到你的项目中。

生成对抗网络，简称 GANs，是一种训练生成模型的深度学习技术。

GANs 的研究和应用只有几年的历史，但所取得的成果却令人瞩目。因为这个领域如此年轻，知道如何开始、关注什么以及如何最好地使用现有技术可能是一项挑战。

在本速成课程中，您将发现如何在七天内开始并自信地使用 Python 开发深度学习生成对抗网络。

**注**:这是一个很大很重要的岗位。你可能想把它做成书签。

我们开始吧。

*   **更新 2019 年 7 月**:更改了 LeakyReLU 和 BatchNorm 层的顺序(谢谢 Chee)。

![How to Get Started With Generative Adversarial Networks (7-Day Mini-Course)](img/9df8871f61f4008c9606af1d46da6767.png)

如何开始使用生成对抗网络(7 天迷你课程)
图片由[马蒂亚斯·里普](https://www.flickr.com/photos/56218409@N03/15238615085/)提供，保留部分权利。

## 这个速成班是给谁的？

在我们开始之前，让我们确保你在正确的地方。

下面的列表提供了一些关于本课程是为谁设计的一般指南。

如果这些点不完全匹配，不要惊慌；你可能只需要在一个或另一个领域进行复习就能跟上。

你需要知道:

*   围绕基本的 Python、NumPy 和 Keras 进行深入学习。

你不需要:

*   数学天才！
*   深度学习专家！
*   一个计算机视觉研究员！

这门速成课程将把你从一个懂一点机器学习的开发人员带到一个能把 GANs 带到你自己的计算机视觉项目中的开发人员。

**注**:本速成课程假设您有一个正在运行的 Python 2 或 3 SciPy 环境，其中至少安装了 NumPy、Pandas、Sklearn 和 Keras 2。如果您需要环境方面的帮助，可以遵循这里的逐步教程:

*   [如何为机器学习和深度学习建立 Python 环境](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

## 速成班概述

这门速成课分为七节课。

您可以每天完成一节课(推荐)或一天内完成所有课程(硬核)。这真的取决于你有多少时间和你的热情程度。

以下是让您开始使用 Python 中的生成对抗网络并提高工作效率的七堂课:

*   **第 01 课**:什么是生成对抗网络？
*   **第 02 课** : GAN 提示、技巧和黑客
*   **第 03 课**:鉴别器和发电机模型
*   **第 04 课** : GAN 损耗函数
*   **第 05 课** : GAN 训练算法
*   **第 06 课**:图像转换的 GANs
*   **第 07 课**:高级 GANs

每节课可能花费你 60 秒到 30 分钟。慢慢来，按照自己的节奏完成课程。提问，甚至在下面的评论中发布结果。

这些课程可能期望你去发现如何做事。我会给你提示，但每节课的部分要点是强迫你学习去哪里寻找关于深度学习和 GANs 的帮助(提示:我在这个博客上有所有的答案；只需使用搜索框)。

在评论中发布您的结果；我会为你加油的！

**坚持住；不要放弃。**

**注**:这只是速成班。关于更多的细节和充实的教程，请参见我的书，题目是“使用 Python 的生成对抗网络”

## 第一课:什么是生成对抗网络？

在本课中，您将发现什么是 GANs 以及基本的模型架构。

生成对抗网络，简称 GANs，是一种使用深度学习方法的生成建模方法，例如[卷积神经网络](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)。

GANs 是一种训练生成模型的聪明方法，它通过两个子模型将问题框架化为有监督的学习问题:生成器模型，我们训练它来生成新的示例，以及鉴别器模型，它试图将示例分类为真实的(来自域)或虚假的(生成的)。

*   **发电机**。用于从问题域生成新的似是而非的示例的模型。
*   **鉴别器**。用于将示例分类为真实(来自领域)或虚假(生成)的模型。

这两个模型在一个零和博弈中一起训练，对抗性的，直到鉴别器模型被愚弄了大约一半的时间，这意味着生成器模型正在生成似是而非的例子。

### 发电机

生成器模型以固定长度的随机向量作为输入，并在域中生成图像。

向量是从高斯分布(称为潜在空间)中随机抽取的，该向量用于为生成过程播种。

经过训练后，生成器模型被保留并用于生成新的样本。

### 歧视者

鉴别器模型以域中的一个例子作为输入(真实的或生成的)，并预测真实的或虚假的二进制类标签(生成的)。

真实的例子来自训练数据集。生成的示例由生成器模型输出。

鉴别器是一个正常的(并且很好理解的)分类模型。

在训练过程之后，鉴别器模型被丢弃，因为我们对生成器感兴趣。

### GAN 培训

生成器和鉴别器这两个模型是一起训练的。

单个训练周期包括首先从问题域中选择一批真实图像。生成一批潜在点并将其馈送到生成器模型以合成一批图像。

然后使用该批真实和生成的图像更新鉴别器，最小化任何二进制分类问题中使用的二进制交叉熵损失。

然后通过鉴别器模型更新生成器。这意味着生成的图像呈现给鉴别器，就好像它们是真实的(不是生成的)，并且误差通过生成器模型传播回来。这具有更新生成器模型以生成更可能欺骗鉴别器的图像的效果。

然后对给定次数的训练迭代重复该过程。

### 你的任务

本课的任务是列出生成对抗网络的三种可能应用。你可能会从最近发表的研究论文中获得灵感。

在下面的评论中发表你的发现。我很想看看你的发现。

在下一课中，您将发现成功培训 GAN 模型的技巧和诀窍。

## 第 02 课:GAN 提示、技巧和技巧

在本课中，您将发现成功训练 GAN 模型所需了解的技巧、诀窍和技巧。

生成对抗网络很难训练。

这是因为该架构同时涉及一个生成器和一个鉴别器模型，它们在零和游戏中竞争。一个模型的改进是以另一个模型的表现下降为代价的。结果是一个非常不稳定的训练过程，经常会导致失败，例如，一个生成器总是生成相同的图像或生成无意义的图像。

因此，有许多试探法或最佳实践(称为“ [GAN 黑客](https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/)”)可以在配置和训练您的 GAN 模型时使用。

在稳定 GAN 模型的设计和训练中，最重要的一步可能是被称为[深度卷积 GAN](https://arxiv.org/abs/1511.06434) 或 DCGAN 的方法。

在实现您的 GAN 模型时，此体系结构涉及到七个要考虑的最佳实践:

1.  使用条纹卷积进行下采样(例如，不要使用[池化层](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/))。
2.  使用交错卷积进行上采样(例如，使用转置卷积层)。
3.  使用 LeakyReLU(例如不要使用[标准 ReLU](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/) )。
4.  使用[批量标准化](https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/)(例如激活后标准化层输出)。
5.  使用高斯权重初始化(例如，平均值为 0.0，标准偏差为 0.02)。
6.  使用[亚当随机梯度下降](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)(例如学习率 0.0002，beta 1 0.5)。
7.  [将图像](https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/)缩放至范围[-1，1](例如，在发生器的输出中使用 tanh)。

这些试探法是从业者在一系列问题上测试和评估数百或数千个配置操作组合来之不易的。

### 你的任务

在本课中，您的任务是列出三个可以在培训中使用的额外的 GAN 技巧或技巧。

在下面的评论中发表你的发现。我很想看看你的发现。

在下一课中，您将发现如何实现简单的鉴别器和生成器模型。

## 第 03 课:鉴别器和生成器模型

在本课中，您将发现如何使用 Keras 深度学习库实现一个简单的鉴别器和生成器模型。

我们将假设我们域中的图像大小和颜色为 28×28 像素，这意味着它们有三个颜色通道。

### 鉴别器模型

鉴别器模型接受大小为 28x28x3 像素的图像，并且必须通过 sigmoid 激活函数将其分类为真实(1)或虚假(0)。

我们的模型有两个卷积层，每个卷积层有 64 个滤波器，并使用相同的填充。每个卷积层将使用 [2×2 步长](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)对输入进行下采样，这是 GANs 的最佳实践，而不是使用池层。

同样遵循最佳实践，卷积层之后是斜率为 0.2 的 LeakyReLU 激活和批处理标准化层。

```py
...
# define the discriminator model
model = Sequential()
# downsample to 14x14
model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=(28,28,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
# downsample to 7x7
model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
# classify
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
```

### 发电机模型

生成器模型以潜在空间中的 100 维点作为输入，并生成 28x28x3。

潜在空间中的点是[高斯随机数](https://machinelearningmastery.com/how-to-generate-random-numbers-in-python/)的向量。这是使用密集层投影到 64 个微小的 7×7 图像的基础上。

然后，使用两个 2×2 步长的转置卷积层对小图像进行两次上采样，然后是 BatchNormalization 和 LeakyReLU 层，这是 GANs 的最佳实践。

通过 tanh 激活功能，输出是像素值在[-1，1]范围内的三通道图像。

```py
...
# define the generator model
model = Sequential()
# foundation for 7x7 image
n_nodes = 64 * 7 * 7
model.add(Dense(n_nodes, input_dim=100))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(Reshape((7, 7, 64)))
# upsample to 14x14
model.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
# upsample to 28x28
model.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
```

### 你的任务

本课的任务是实现这两个鉴别器模型并总结它们的结构。

对于加分，更新模型以支持 64×64 像素的图像。

在下面的评论中发表你的发现。我很想看看你的发现。

在下一课中，您将了解如何配置损耗函数来训练 GAN 模型。

## 第 04 课:GAN 损耗函数

在本课中，您将了解如何配置用于训练 GAN 模型权重的[损失函数](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)。

### 鉴别器损耗

鉴别器模型被优化，以最大化从数据集正确识别真实图像和生成器输出的假图像或合成图像的概率。

这可以实现为二进制分类问题，其中鉴别器输出给定图像的假和真的概率分别在 0 和 1 之间。

然后，该模型可以直接在成批的真实和虚假图像上进行训练，并最小化负对数似然性，最常见的实现方式是二进制交叉熵损失函数。

作为最佳实践，可以使用具有小学习率和保守动量的随机梯度下降的 [Adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) 版本来优化模型。

```py
...
# compile model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
```

### 发电机损耗

发电机不直接更新，该型号没有损失。

相反，鉴频器用于为发生器提供学习或间接损失函数。

这是通过创建一个复合模型来实现的，在该模型中，生成器输出一个图像，该图像直接输入鉴别器进行分类。

然后，可以通过在潜在空间中提供随机点作为输入并向鉴别器指示所生成的图像实际上是真实的来训练合成模型。这具有更新生成器权重的效果，以输出更有可能被鉴别器分类为真实的图像。

重要的是，鉴别器权重在此过程中不会更新，并且标记为不可训练。

复合模型使用与独立鉴别器模型相同的分类交叉熵损失和相同的随机梯度下降亚当版本来执行优化。

```py
# create the composite model for training the generator
generator = ...
discriminator = ...
...
# make weights in the discriminator not trainable
d_model.trainable = False
# connect them
model = Sequential()
# add generator
model.add(generator)
# add the discriminator
model.add(discriminator)
# compile model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
```

### 你的任务

本课中，您的任务是研究和总结可用于训练 GAN 模型的三种额外类型的损失函数。

在下面的评论中发表你的发现。我很想看看你的发现。

在下一课中，您将发现用于更新 GAN 模型权重的训练算法。

## 第 05 课:GAN 训练算法

在本课中，您将发现 GAN 训练算法。

定义 GAN 模型是困难的部分。GAN 训练算法相对简单。

该算法的一个周期包括首先选择一批真实图像，并使用当前的生成器模型生成一批假图像。您可以开发一些小函数来执行这两个操作。

然后，通过调用 *train_on_batch()* Keras 函数，这些真实和虚假图像被用来直接更新鉴别器模型。

接下来，可以生成潜在空间中的点作为复合生成器-鉴别器模型的输入，并且可以提供“真实”(类=1)的标签来更新生成器模型的权重。

然后，训练过程重复数千次。

生成器模型可以定期保存，然后加载以检查生成图像的质量。

下面的例子演示了 GAN 训练算法。

```py
...
# gan training algorithm
discriminator = ...
generator = ...
gan_model = ...
n_batch = 16
latent_dim = 100
for i in range(10000)
	# get randomly selected 'real' samples
	X_real, y_real = select_real_samples(dataset, n_batch)
	# generate 'fake' examples
	X_fake, y_fake = generate_fake_samples(generator, latent_dim, n_batch)
	# create training set for the discriminator
	X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
	# update discriminator model weights
	d_loss = discriminator.train_on_batch(X, y)
	# prepare points in latent space as input for the generator
	X_gan = generate_latent_points(latent_dim, n_batch)
	# create inverted labels for the fake samples
	y_gan = ones((n_batch, 1))
	# update the generator via the discriminator's error
	g_loss = gan_model.train_on_batch(X_gan, y_gan)
```

### 你的任务

在本课中，您的任务是将本课和上一课中的元素联系在一起，并在小型图像数据集(如 [MNIST](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/) 或 [CIFAR-10](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/) 上训练一个 GAN。

在下面的评论中发表你的发现。我很想看看你的发现。

在下一课中，您将发现 GANs 在图像转换中的应用。

## 第六课:图像转换的甘斯

在本课中，您将发现用于图像转换的 GANs。

图像到图像的转换是给定源图像到目标图像的受控转换。一个例子可能是黑白照片到彩色照片的转换。

图像到图像的翻译是一个具有挑战性的问题，对于给定的翻译任务或数据集，通常需要专门的模型和损失函数。

GANs 可以被训练来执行图像到图像的转换，两个例子包括 Pix2Pix 和 CycleGAN。

### Pix2Pix

Pix2Pix GAN 是一种通用的图像到图像转换方法。

该模型是在成对示例的数据集上训练的，其中每对示例都涉及所需翻译前后的图像示例。

Pix2Pix 模型基于条件生成对抗网络，在该网络中，根据给定的输入图像生成目标图像。

鉴别器模型被给予一个输入图像和一个真实的或生成的配对图像，并且必须确定配对图像是真实的还是伪造的。

生成器模型以给定的图像作为输入，并生成图像的翻译版本。生成器模型被训练成既欺骗鉴别器模型，又最小化生成的图像和期望的目标图像之间的损失。

Pix2Pix 中使用了更复杂的深度卷积神经网络模型。具体来说，U-Net 模型用于生成器模型，PatchGAN 用于鉴别器模型。

生成器的损失由正常 GAN 模型的对抗损失和生成的和预期的转换图像之间的 L1 损失组成。

### -你好

Pix2Pix 模型的一个限制是，它需要所需翻译前后成对示例的数据集。

有许多图像到图像的翻译任务，我们可能没有翻译的例子，比如把斑马的照片翻译成马。还有其他不存在这种成对例子的图像转换任务，例如将风景艺术翻译成照片。

CycleGAN 是一种在没有成对例子的情况下自动训练图像到图像转换模型的技术。使用来自源域和目标域的不需要以任何方式相关的图像集合，以无监督的方式训练模型。

CycleGAN 是 GAN 架构的扩展，它涉及两个生成器模型和两个鉴别器模型的同时训练。

一个生成器将来自第一域的图像作为输入并输出第二域的图像，另一个生成器将来自第二域的图像作为输入并生成来自第一域的图像。然后使用鉴别器模型来确定生成的图像有多可信，并相应地更新生成器模型。

CycleGAN 对称为周期一致性的体系结构进行了额外的扩展。这是由第一发生器输出的图像可以用作第二发生器的输入并且第二发生器的输出应该与原始图像匹配的想法。反之亦然:第二个发生器的输出可以作为第一个发生器的输入，结果应该与第二个发生器的输入相匹配。

### 你的任务

在本课中，您的任务是列出五个图像到图像转换的示例，您可能希望使用 GAN 模型进行探索。

在下面的评论中发表你的发现。我很想看看你的发现。

在下一课中，您将发现 GAN 模型的一些最新进展。

## 第 07 课:高级 GANs

在这一课中，你将发现一些更先进的 GAN，正在展示显著的成果。

### 比根

BigGAN 是一种将一套最近的最佳实践整合在一起的方法，用于训练 GANs 并扩大批量和模型参数的数量。

顾名思义，BigGAN 专注于放大 GAN 模型。这包括具有以下特点的 GAN 型号:

*   更多模型参数(例如，更多要素图)。
*   更大的批量(例如数百或数千张图像)。
*   架构变化(例如，自我关注模块)。

由此产生的 BigGAN 生成器模型能够在广泛的图像类别中生成高质量的 256×256 和 512×512 图像。

### 渐进式增长 GAN

渐进式增长 GAN 是 GAN 训练过程的扩展，允许稳定训练发电机模型，可以输出大的高质量图像。

它包括从一个非常小的图像开始，逐步增加层块，增加生成器模型的输出大小和鉴别器模型的输入大小，直到达到所需的图像大小。

渐进式增长 GAN 最令人印象深刻的成就可能是生成了大的 1024×1024 像素的真实感生成人脸。

### stylenan

风格生成对抗网络，简称 StyleGAN，是 GAN 架构的扩展，它对生成器模型提出了很大的改变。

这包括使用映射网络将潜在空间中的点映射到中间潜在空间，使用中间潜在空间来控制生成器模型中每个点的样式，以及引入噪声作为生成器模型中每个点的变化源。

生成的模型不仅能够生成令人印象深刻的照片级高质量人脸照片，还可以通过改变样式向量和噪声来控制生成的图像在不同细节级别的样式。

例如，合成网络中较低分辨率的层块控制高级风格，如姿势和发型，较高分辨率的层块控制配色方案和非常精细的细节，如雀斑和发束的位置。

### 你的任务

在本课中，您的任务是列出 3 个示例，说明如何使用能够生成大型照片真实感图像的模型。

在下面的评论中发表你的发现。我很想看看你的发现。

这是最后一课。

## 末日！
(看你走了多远)

你成功了。干得好！

花一点时间，回头看看你已经走了多远。

你发现了:

*   GANs 是一种深度学习技术，用于训练能够合成高质量图像的生成模型。
*   训练 GANs 本质上是不稳定的，并且容易失败，这可以通过在 GAN 模型的设计、配置和训练中采用最佳实践试探法来克服。
*   GAN 架构中使用的生成器和鉴别器模型可以在 Keras 深度学习库中简单直接地定义。
*   鉴别器模型像任何其他二分类深度学习模型一样被训练。
*   生成器模型通过复合模型架构中的鉴别器模型进行训练。
*   GANs 能够有条件地生成图像，例如使用成对和不成对的例子进行图像到图像的转换。
*   GANs 的进步，如放大模型和逐步增长模型，允许生成更大和更高质量的图像。

下一步，用 python 查看我的《生成对抗网络》一书。

## 摘要

**你觉得迷你课程怎么样？**
你喜欢这个速成班吗？

**你有什么问题吗？有什么症结吗？**
让我知道。请在下面留言。