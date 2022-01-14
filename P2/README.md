# 实验报告

## 1. 数据预处理

(a) 为什么需要验证集？

(b) 应该如何选取验证集，验证集和测试集应该有什么联系?

(c) 为什么要在切分数据后再划分训练和验证集？直接在原始的时间序列中，划分几条作为验证集会出现什么问题？

(d) 数据参与训练的时候打乱顺序(shuffle)输入模型，会有什么好处，什么坏处？

## 2. 模型建立

(a) 输入层和输出层应该是什么层?为什么?

(b) 隐藏层应该有多少层?过多或过少分别会带来什么问题?

(c) 除了全连接网络(MLP)，还可以使用什么网络模块来解决这个问题？

(d)【代码题】请用至少一种其它网络模块来搭建网络，并和 MLP 网络对比优缺点。


## 3. 损失函数

(a) mean absolute error(MAE)和 mean squared error(MSE)做损失函数有区别吗？请简单分析一下。

(b)【代码题】请分别用这两种损失函数进行训练，对比模型的指标表现，并分析原因。

(c) 如果让你改进损失函数，你会如何改进?(指出当前损失函数存在的问题， 并提出一个解决方案。无需具体证明)

## 4. 优化方法

(a) 你选择了什么优化方法(例如 SGD，Adam 等)？为什么？

(b) 在 SGD 上增加动量会改变模型性能吗？为什么? 

## 5. 训练

(a) 在上述所有尝试过的模型中，请分别汇报训练集和验证集的损失

(b) 在上述所有尝试过的模型中，请分别汇报训练集和验证集的 sMAPE 



## 6. 测试

(a) 在上述的尝试中，请汇报测试集上的最优损失，请标明模型，损失函数和优化方法

(b) 在上述的尝试中，请汇报测试集上的最优 sMAPE，请标明模型，损失函数和优化方法 

## 7. 不同训练长度的测试结果对比

(a)【代码题】请至少尝试三种输入长度，对网络进行训练和测试。

(b) 你认为选择输入长度的依据应该是什么？最优的是哪一个输入长度？

(c) 输入长度的选择可以自动化吗？请给出理由。

(d) 总结汇报最优的训练/验证/测试集损失和 sMAPE，请标明输入长度，模型，损失函数和优化方法