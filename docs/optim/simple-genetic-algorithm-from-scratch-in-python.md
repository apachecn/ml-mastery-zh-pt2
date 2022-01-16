# Python 中从零开始的简单遗传算法

> 原文：<https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/>

最后更新于 2021 年 10 月 12 日

**遗传算法**是一种随机全局优化算法。

它可能是最受欢迎和最广为人知的生物启发算法之一，与人工神经网络一起。

该算法是一种进化算法，通过具有二元表示的自然选择和基于遗传重组和遗传突变的简单算子，执行受生物进化理论启发的优化过程。

在本教程中，您将发现遗传算法优化算法。

完成本教程后，您将知道:

*   遗传算法是一种受进化启发的随机优化算法。
*   如何在 Python 中从头实现遗传算法？
*   如何将遗传算法应用于连续目标函数？

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![Simple Genetic Algorithm From Scratch in Python](img/46ebae089519dd220fe7482698436211.png)

Python 中从零开始的简单遗传算法
图片由 [Magharebia](https://www.flickr.com/photos/magharebia/5323128313/) 提供，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  遗传算法
2.  从零开始的遗传算法
3.  OneMax 的遗传算法
4.  连续函数优化的遗传算法

## 遗传算法

[遗传算法](https://en.wikipedia.org/wiki/Genetic_algorithm)是一种随机全局搜索优化算法。

它的灵感来自于通过自然选择进化的生物学理论。具体来说，就是将遗传学的理解和理论结合起来的新合成。

> 遗传算法(算法 9.4)从生物进化中借用灵感，在生物进化中，更健康的个体更有可能将他们的基因传递给下一代。

—第 148 页，[优化算法](https://amzn.to/2Traqek)，2019。

该算法使用遗传表示(位串)、适应度(功能评估)、遗传重组(位串交叉)和变异(翻转位)的类似物。

该算法首先创建一个固定大小的随机位串群体。算法的主循环重复固定的迭代次数，或者直到在给定的迭代次数上最佳解没有进一步的改进。

算法的一次迭代就像是进化的一代。

首先，使用目标函数评估位串(候选解)的总体。将每个候选解的目标函数评估作为解的适合度，该适合度可以最小化或最大化。

然后，根据他们的健康状况选择父母。给定的候选解可以被零次或多次用作父解。一种简单有效的选择方法包括从人群中随机抽取 *k* 个候选人，并从具有最佳适应度的组中选择成员。这被称为锦标赛选择，其中 *k* 是一个超参数，并被设置为一个值，如 3。这个简单的方法模拟了一个更昂贵的适合度比例选择方案。

> 在锦标赛选择中，每个父母是群体中随机选择的 k 条染色体中最合适的

—第 151 页，[优化算法](https://amzn.to/2Traqek)，2019。

父母被用作生成下一代候选点的基础，并且人口中的每个职位需要一个父母。

然后父母成对出现，用来生两个孩子。使用交叉算子执行重组。这包括在位串上选择一个随机分割点，然后创建一个子串，其位从第一个父串到分割点，从第二个父串到串的末尾。然后，对于第二个孩子，这个过程是相反的。

比如父母两个:

*   parent1 = 00000
*   parent2 = 11111

可能会导致两个交叉的孩子:

*   child1 = 00011
*   child2 = 11100

这被称为单点交叉，还有许多其他的操作符变体。

对每对父代应用概率交叉，这意味着在某些情况下，父代的副本被当作子代，而不是重组操作符。交叉由设置为较大值(如 80%或 90%)的超参数控制。

> 交叉是遗传算法的显著特点。它包括混合和匹配父母的部分来形成孩子。如何混合和匹配取决于个人的表现。

—第 36 页，[元试探法精要](https://amzn.to/2HxZVn4)，2011。

变异包括翻转已创建的子候选解中的位。通常，突变率设置为 *1/L* ，其中 *L* 是位串的长度。

> 二进制值染色体中的每一位通常都有很小的翻转概率。对于 m 位的染色体，这个突变率通常设置为 1/m，平均每个子染色体产生一个突变。

—第 155 页，[优化算法](https://amzn.to/2Traqek)，2019。

例如，如果一个问题使用 20 位的位串，那么良好的默认突变率将是(1/20) = 0.05 或概率为 5%。

这定义了简单的遗传算法程序。这是一个很大的研究领域，算法有很多扩展。

现在我们已经熟悉了简单的遗传算法过程，让我们看看如何从零开始实现它。

## 从零开始的遗传算法

在这一部分，我们将开发一个遗传算法的实现。

第一步是创建随机位串群体。我们可以使用布尔值*真*和*假*，字符串值“0”和“1”，或者整数值 0 和 1。在这种情况下，我们将使用整数值。

我们可以使用 [randint()函数](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html)在一个范围内生成整数值数组，我们可以将该范围指定为从 0 开始小于 2 的值，例如 0 或 1。我们还将把一个候选解决方案表示为一个列表，而不是一个 NumPy 数组，以保持简单。

随机位串的初始群体可以如下创建，其中“ *n_pop* ”是控制群体大小的超参数，“ *n_bits* ”是定义单个候选解决方案中的位数的超参数:

```py
...
# initial population of random bitstring
pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
```

接下来，我们可以枚举固定次数的算法迭代，在这种情况下，由名为“ *n_iter* ”的超参数控制。

```py
...
# enumerate generations
	for gen in range(n_iter):
		...
```

算法迭代的第一步是评估所有候选解。

我们将使用一个名为 *objective()* 的函数作为通用目标函数，并调用它来获得一个适应度得分，我们将最小化该得分。

```py
...
# evaluate all candidates in the population
scores = [objective(c) for c in pop]
```

然后，我们可以选择将用于创建孩子的父母。

锦标赛选择过程可以实现为一个函数，该函数接受人口并返回一个选定的父代。 *k* 值用默认参数固定为 3，但是如果您愿意，可以尝试不同的值。

```py
# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]
```

然后，我们可以为群体中的每个位置调用这个函数一次，以创建一个父代列表。

```py
...
# select parents
selected = [selection(pop, scores) for _ in range(n_pop)]
```

然后我们可以创造下一代。

这首先需要一个函数来执行交叉。这个函数将采用两个父节点和交叉率。交叉率是一个超参数，它决定是否执行交叉，如果不执行，父代将被复制到下一代。这是一个概率，通常具有接近 1.0 的大值。

下面的*交叉()*函数使用在范围[0，1]内抽取一个随机数来确定是否执行交叉，然后选择一个有效的分割点来执行交叉。

```py
# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]
```

我们还需要一个功能来执行突变。

该过程只是以由“ *r_mut* ”超参数控制的低概率翻转位。

```py
# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]
```

然后，我们可以遍历父代列表，创建一个子代列表作为下一代，根据需要调用交叉和变异函数。

```py
...
# create the next generation
children = list()
for i in range(0, n_pop, 2):
	# get selected parents in pairs
	p1, p2 = selected[i], selected[i+1]
	# crossover and mutation
	for c in crossover(p1, p2, r_cross):
		# mutation
		mutation(c, r_mut)
		# store for next generation
		children.append(c)
```

我们可以将所有这些联系到一个名为 *genetic_algorithm()* 的函数中，该函数采用目标函数的名称和搜索的超参数，并返回搜索过程中找到的最佳解。

```py
# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(pop[0])
	# enumerate generations
	for gen in range(n_iter):
		# evaluate all candidates in the population
		scores = [objective(c) for c in pop]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]
```

现在我们已经开发了遗传算法的实现，让我们来探索如何将其应用于目标函数。

## OneMax 的遗传算法

在这一节中，我们将把遗传算法应用于一个基于二进制字符串的优化问题。

这个问题叫做 OneMax，它根据字符串中 1 的数量来计算二进制字符串。例如，长度为 20 位的位串对于全 1 的串的得分为 20。

假设我们已经实现了最小化目标函数的遗传算法，我们可以给这个评估添加一个负号，使得大的正值变成大的负值。

下面的 *onemax()* 函数实现了这一点，它将整数值的位串作为输入，并返回这些值的负和。

```py
# objective function
def onemax(x):
	return -sum(x)
```

接下来，我们可以配置搜索。

搜索将运行 100 次迭代，我们将在候选解决方案中使用 20 位，这意味着最佳适应度将为-20.0。

种群规模将为 100，我们将使用 90%的交叉率和 5%的变异率。这种配置是经过反复试验后选择的。

```py
...
# define the total iterations
n_iter = 100
# bits
n_bits = 20
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)
```

然后可以调用搜索并报告最佳结果。

```py
...
# perform the genetic algorithm search
best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))
```

将这些联系在一起，下面列出了将遗传算法应用于 OneMax 目标函数的完整示例。

```py
# genetic algorithm search of the one max optimization problem
from numpy.random import randint
from numpy.random import rand

# objective function
def onemax(x):
	return -sum(x)

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(pop[0])
	# enumerate generations
	for gen in range(n_iter):
		# evaluate all candidates in the population
		scores = [objective(c) for c in pop]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]

# define the total iterations
n_iter = 100
# bits
n_bits = 20
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)
# perform the genetic algorithm search
best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))
```

运行该示例将报告一路上找到的最佳结果，然后是搜索结束时的最终最佳解决方案，我们希望这是最佳解决方案。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到搜索在大约八代之后找到了最优解。

```py
>0, new best f([1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]) = -14.000
>0, new best f([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0]) = -15.000
>1, new best f([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1]) = -16.000
>2, new best f([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]) = -17.000
>2, new best f([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) = -19.000
>8, new best f([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) = -20.000
Done!
f([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) = -20.000000
```

## 连续函数优化的遗传算法

优化 OneMax 函数不是很有趣；我们更可能想要优化连续函数。

例如，我们可以定义 x^2 最小化函数，该函数接受输入变量，并且在 f(0，0) = 0.0 时具有最优值。

```py
# objective function
def objective(x):
	return x[0]**2.0 + x[1]**2.0
```

我们可以用遗传算法最小化这个函数。

首先，我们必须定义每个输入变量的界限。

```py
...
# define range for input
bounds = [[-5.0, 5.0], [-5.0, 5.0]]
```

我们将把“ *n_bits* ”超参数作为目标函数的每个输入变量的位数，并将其设置为 16 位。

```py
...
# bits per variable
n_bits = 16
```

这意味着给定两个输入变量，我们的实际位串将有(16 * 2) = 32 位。

我们必须相应地更新我们的突变率。

```py
...
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))
```

接下来，我们需要确保初始群体创建足够大的随机位串。

```py
...
# initial population of random bitstring
pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
```

最后，我们需要在用目标函数评估每个位串之前，将这些位串解码成数字。

我们可以通过首先将每个子串解码成整数，然后将整数缩放到所需的范围来实现这一点。这将给出一个范围内的值向量，然后可以提供给目标函数进行评估。

下面的 *decode()* 函数实现了这一点，将函数的边界、每个变量的位数和一个位串作为输入，并返回一个解码实值列表。

```py
# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
		decoded.append(value)
	return decoded
```

然后，我们可以在算法循环的开始调用它来解码种群，然后评估种群的解码版本。

```py
...
# decode population
decoded = [decode(bounds, n_bits, p) for p in pop]
# evaluate all candidates in the population
scores = [objective(d) for d in decoded]
```

将这些联系在一起，下面列出了用于连续函数优化的遗传算法的完整示例。

```py
# genetic algorithm search for continuous function optimization
from numpy.random import randint
from numpy.random import rand

# objective function
def objective(x):
	return x[0]**2.0 + x[1]**2.0

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
		decoded.append(value)
	return decoded

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
	# enumerate generations
	for gen in range(n_iter):
		# decode population
		decoded = [decode(bounds, n_bits, p) for p in pop]
		# evaluate all candidates in the population
		scores = [objective(d) for d in decoded]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]

# define range for input
bounds = [[-5.0, 5.0], [-5.0, 5.0]]
# define the total iterations
n_iter = 100
# bits per variable
n_bits = 16
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))
# perform the genetic algorithm search
best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
decoded = decode(bounds, n_bits, best)
print('f(%s) = %f' % (decoded, score))
```

运行该示例会报告沿途的最佳解码结果以及运行结束时的最佳解码解决方案。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到算法发现一个非常接近 f(0.0，0.0) = 0.0 的输入。

```py
>0, new best f([-0.785064697265625, -0.807647705078125]) = 1.268621
>0, new best f([0.385894775390625, 0.342864990234375]) = 0.266471
>1, new best f([-0.342559814453125, -0.1068115234375]) = 0.128756
>2, new best f([-0.038909912109375, 0.30242919921875]) = 0.092977
>2, new best f([0.145721435546875, 0.1849365234375]) = 0.055436
>3, new best f([0.14404296875, -0.029754638671875]) = 0.021634
>5, new best f([0.066680908203125, 0.096435546875]) = 0.013746
>5, new best f([-0.036468505859375, -0.10711669921875]) = 0.012804
>6, new best f([-0.038909912109375, -0.099639892578125]) = 0.011442
>7, new best f([-0.033111572265625, 0.09674072265625]) = 0.010455
>7, new best f([-0.036468505859375, 0.05584716796875]) = 0.004449
>10, new best f([0.058746337890625, 0.008087158203125]) = 0.003517
>10, new best f([-0.031585693359375, 0.008087158203125]) = 0.001063
>12, new best f([0.022125244140625, 0.008087158203125]) = 0.000555
>13, new best f([0.022125244140625, 0.00701904296875]) = 0.000539
>13, new best f([-0.013885498046875, 0.008087158203125]) = 0.000258
>16, new best f([-0.011444091796875, 0.00518798828125]) = 0.000158
>17, new best f([-0.0115966796875, 0.00091552734375]) = 0.000135
>17, new best f([-0.004730224609375, 0.00335693359375]) = 0.000034
>20, new best f([-0.004425048828125, 0.00274658203125]) = 0.000027
>21, new best f([-0.002288818359375, 0.00091552734375]) = 0.000006
>22, new best f([-0.001983642578125, 0.00091552734375]) = 0.000005
>22, new best f([-0.001983642578125, 0.0006103515625]) = 0.000004
>24, new best f([-0.001373291015625, 0.001068115234375]) = 0.000003
>25, new best f([-0.001373291015625, 0.00091552734375]) = 0.000003
>26, new best f([-0.001373291015625, 0.0006103515625]) = 0.000002
>27, new best f([-0.001068115234375, 0.0006103515625]) = 0.000002
>29, new best f([-0.000152587890625, 0.00091552734375]) = 0.000001
>33, new best f([-0.0006103515625, 0.0]) = 0.000000
>34, new best f([-0.000152587890625, 0.00030517578125]) = 0.000000
>43, new best f([-0.00030517578125, 0.0]) = 0.000000
>60, new best f([-0.000152587890625, 0.000152587890625]) = 0.000000
>65, new best f([-0.000152587890625, 0.0]) = 0.000000
Done!
f([-0.000152587890625, 0.0]) = 0.000000
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [搜索、优化和机器学习中的遗传算法](https://amzn.to/3jADHgZ)，1989。
*   [遗传算法导论](https://amzn.to/3kK8Osd)，1998。
*   [优化算法](https://amzn.to/2Traqek)，2019。
*   [元试探法精要](https://amzn.to/2HxZVn4)，2011。
*   [计算智能:导论](https://amzn.to/2HzjbjV)，2007。

### 应用程序接口

*   num py . random . ranint API。

### 文章

*   [遗传算法，维基百科](https://en.wikipedia.org/wiki/Genetic_algorithm)。
*   [遗传算法，Scholarpedia](http://www.scholarpedia.org/article/Genetic_algorithms) 。

## 摘要

在本教程中，您发现了遗传算法优化。

具体来说，您了解到:

*   遗传算法是一种受进化启发的随机优化算法。
*   如何在 Python 中从头实现遗传算法？
*   如何将遗传算法应用于连续目标函数？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。