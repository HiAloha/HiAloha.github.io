---
title: self-attention、transformer解读
date: 2021-11-21 10:11:47
categories: NLP & ML & DL
tags: [NLP]
mathjax: true
---



## 背景

之前用在NLP上，现在又用到语音、CV上，推荐场景也有用到。
面试考了几次，我最近甚至仍有答错之处，那行，我彻底花点时间来整理一下，尽管网上有N多博客，N多解读，但看了好多，仍然没有彻底攻下，所以我还是以输出倒闭输入来完成对知识的吸收和理解。

*（我会慢慢一轮一轮地完善本文的）*



## 模型详解

### 相关资源推荐

transformer是在论文*《Attention is all you need》*里面把self-attention发扬光大，说实话，该文其实对新手非常不友好，强烈推荐以李宏毅为首的资源：

- [台大李宏毅自注意力机制和Transformer详解](https://www.bilibili.com/video/BV1v3411r78R?p=1)   [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
  ps: *李宏毅的课程ppt可以在其[主页](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html)下载*——[self-attention课件](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/self_v7.pdf)，[transformer课件](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/seq2seq_v9.pdf)
- [李沐Transformer论文逐段精读](https://www.bilibili.com/video/BV1pu411o7BE?spm_id_from=333.999.0.0)

- [某知乎用户](https://www.zhihu.com/column/c_1314881818630352896) 的解答还不错，给我补充了不少东西

一开始不可能深入代码细节，我当初看论文的时候也看得云里雾里，看了N多博客也似懂非懂，所以还是从看最好的资源入手。
注意： 我几乎完全参考了李宏毅的视频，所以非重点内容已经把标题置为斜体了。从0开始看应该问题不大。



### self-attention 

#### input、output

如何把把一个词汇表示成一个向量呢，最简单做法是**One-hot Encoding**，这种做法没有包含语义信息，比如，cat分别和dog、apple做点积，结果都是0，但从语义上来说，cat和dog应该是更接近的，另外一种做法是用[**Word Embedding**](https://www.youtube.com/watch?v=X7PH3NuYW0Q)，先挖个坑。

可以作为input的东西很多，如语言，语音(10ms用某种方法采样为一个向量)，分子

输入和输出的长度:
输入为N，输出为N，输出如果是一个数字，那就是回归问题，如果是一个label，那就是分类问题。如**POS tagging问题(词性标注)**。
输入为N，输出为1，如根据一段话，判断是谁说的；或者输入一个分子，判断是否有毒性。
输入为N，输出不确定，由机器自己决定，如翻译任务，**seq2seq任务**。



输入输出等长，又叫Sequence Labeling问题。要注意，输入一排向量，长度是不确定的。
*fully connected这里没搞懂，参数共享不，再挖个坑*



#### 结构

输入是向量，输出也是向量（大小和输入一致）。 然后后面你可以丢到一个FC里面，得到标签，解决分类问题。



![](https://user-images.githubusercontent.com/35519242/143065257-e174970e-88f0-4c91-9ffb-5f2419ef28c6.png)

**self-attention + FC可以叠加多个**，经过第一个FC的输出，可以作为下一个self-attention的输入。

![](https://user-images.githubusercontent.com/35519242/143065279-245d861b-bac2-4365-9653-44f185451a4f.png)



**计算权重**

经过self-attention，从输入到输出，你的每一个输出都是考虑了每一个输入的。所以，对于a1，

![](https://user-images.githubusercontent.com/35519242/143065299-3257d596-5b7d-4af2-873d-2a751abb49ff.png)

用矩阵来解决，最终只需要学习的参数是Wq Wk Wv

![](https://user-images.githubusercontent.com/35519242/143265338-288dff4a-fb9e-4f52-a219-9120ea6c466b.png)

**Multi-head Self-attention**

多头操作，q 出来以后乘以两个矩阵，得到qi1 qi2，k 和 v 是一样的，然后分别单独求出输出，bi1 bi2，拼起来再经过一个矩阵 W0 得到最终输出。多头的数量，也是需要调的参数。

![image](https://user-images.githubusercontent.com/35519242/143266491-ad88961b-ddf1-4635-8785-55199f0e565f.png)

**position**

提前加进去即可。位置向量可以手工，可以学习得到，并不一定要采用transformer那篇文章的方法，关于这个问题有很多研究的，见下图2。

![](https://user-images.githubusercontent.com/35519242/143268082-cca79ac9-f27f-4ef3-bfa0-c9eef6e3efcf.png)

![](https://user-images.githubusercontent.com/35519242/143268544-362e2917-f072-4bc9-9a1b-601137bc462f.png)

#### *其他应用*

- self-attention 用在语音和cv上现在也很常见。

- RNN v.s. self-attention：单向的RNN，你每一个输出只考虑了前面的输入，但是self-attention考虑了整个输入。如果是双向，尽管不存在这个问题，也有别的问题，比如——你输入序列的最后一个embedding，如果要考虑第一个embedding，那第一个embedding得逐个传下去不消失，耗时长。self-attention，直接操作就行了。RNN也无法平行处理所有的数据，运算效率比较低。更多比较可以参考[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) (一般来说我都没看，先留个记录，以备不时之需)

- self-attention for Graph 应用也出现了

- 图神经网络参考：[Graph Neural Network (1/2)](https://www.youtube.com/watch?app=desktop&v=eybCCtNKwzA&feature=youtu.be)、[Graph Neural Network (2/2)](https://www.youtube.com/watch?app=desktop&v=M9ht8vsVEw8&feature=youtu.be)
- [Long Range Arena: A Benchmark for Efficient Transformers](https://arxiv.org/abs/2011.04006)
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)

### Transformer

#### 神图

神图先放，慢慢解释

![](https://user-images.githubusercontent.com/35519242/143689653-f796349a-acd4-4bf6-8d54-64640ecdc55b.png)

#### **Seq2seq 的广泛应用**

transformer本质上是一个Seq2seq的模型，常见任务比如机器翻译、语音识别、语音合成、聊天机器人等等都是Seq2seq任务，输出的长度由模型来决定，提前未知。很多别的任务都可以用Seq2seq来解决，**但是，或许针对特定任务设定特定模型会取得比Seq2seq更好的结果。**

- **QA**：

  > [The Natural Language Decathlon: Multitask Learning as Question Answering](https://arxiv.org/abs/1806.08730)
  > [LAMOL: LAnguage MOdeling for Lifelong Language Learning](https://arxiv.org/abs/1909.03329)

- **Syntactic Parsing** *（参考文章：**Grammar as a Foreign Language**）*![](https://user-images.githubusercontent.com/35519242/143675381-910204a7-6a35-43a9-ad71-8afb26b5cd2c.png)

- **Multi-label Classification**

  注意和 Multi-class Classification 区别，前者的意思是某个输入可以属于多个类别，而且类别数未知。

  ![](https://user-images.githubusercontent.com/35519242/143675607-8c2e0dd8-8de8-459e-ad4a-1d03cf35397b.png)

- **Object Detection**

  ![](https://user-images.githubusercontent.com/35519242/143675668-39a17c43-44df-4a30-8374-a0c47b48aa3e.png)

#### Seq2seq 的网络结构

起源于 [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) 一文，是为了解决机器翻译问题。

![](https://user-images.githubusercontent.com/35519242/143675718-ff185279-5a01-427d-b105-59b04732be89.png)

#### **Encoder**

输入一排向量，输出另外一排向量。

 ![](https://user-images.githubusercontent.com/35519242/143675806-1136d6a4-397c-422d-a294-ee7c94fe7588.png)

​	Encoder 可以用 RNN 或者 CNN，也可以用 self-attention 来实现。

​	self-attention 具体操作的时候，输入经过多个 block 后才得到输出（经过第一个 Block 后得到的输出作为下一个 Block 的输入，最后一个 Block 的输出作为最终输出），每个 block 是由多个 lay 来组成，下图中每个 Block 的操作具体如图右所示。

​	![](https://user-images.githubusercontent.com/35519242/143675889-e99a5317-d277-4abc-b93a-166240c34f12.png)

具体的操作如下，图非常清晰了，残差和 norm 的做法也指出了：

![](https://user-images.githubusercontent.com/35519242/143676429-1b4e8684-e679-44ae-a932-c62c32697740.png)





[Layer Normalization](https://www.bilibili.com/video/BV1v3411r78R?p=3&spm_id_from=pageDriver) 不用考虑 batch 的资讯， 输入一个向量，经过 Norm，输出另一个向量，操作的时候计算输入向量的均值和方差。
***关于 lay normalization 和 batch normalization 的详细区别，再挖个坑，简单来说是 Batch Normolization 对一批数据的同一维度做归一化，Layer Normolization 是对同一样本的不同维度进行归一化，和batch无关。***

关于 Layer Norm 的操作放在哪个位置更好，以及思考 normalization 的方式，以下文章有探讨。

![](https://user-images.githubusercontent.com/35519242/143689648-be9ed296-1b35-4019-a25f-133a88d51d11.png)

#### Decoder

注：斜体标题不是重点。

##### autoregressive（自回归）
上一时刻的输出会当作下一时刻的输入。

![](https://user-images.githubusercontent.com/35519242/143690076-30227e04-16fb-4c69-874c-f7551320c984.png)

把中间遮住，除了一个 Masked 的操作，decoder 其实和 encoder 是一样的。

![](https://user-images.githubusercontent.com/35519242/143690220-d00078e1-91cb-4a74-a80a-581be519c10e.png)

##### Masked Self-attention 

Masked 的操作很简单，attention 的时候只考虑左边的东西，不考虑右边的东西。就是你当前求 a1 的输出 b1 时，只能看到 a1，求 b2，只能看到 a1、a2。原因其实很简单，你输出的时候，是一个一个出来的，你输出b2的时候，你是没有a3、a4......的。

![](https://user-images.githubusercontent.com/35519242/143690269-31ff93aa-523a-4c76-9a12-985954ae6901.png)

如何停下来？最后一个输出是 end，出现 end 就停止

![](https://user-images.githubusercontent.com/35519242/143690538-13b253aa-98b1-4267-962e-e7bbbd661cdd.png)

##### **Encoder 和 Decoder 的交互**

上文 decoder 被遮住的部分，其实是一个 **cross attention**，有两个输入来自 encoder，有一个输入来自 decoder，具体实现如下图

![](https://user-images.githubusercontent.com/35519242/143691063-21a33705-1fbf-4adf-8c74-f67604a369f0.png)

接下来的第二个输出的操作是一样的，如下图所示

![](https://user-images.githubusercontent.com/35519242/143691257-890668ff-2645-45bf-9197-32a0eae63190.png)

##### *Non-autoregressive (**NAT**)*

两种方法来决定输出：1.把encoder的输出吃进一个classical求得一个长度，由这个长度来决定decoder的输出的长度。2.假设按经验最长就是300了，全部输进去，输出的时候，找到end，舍弃掉end右端的。

NAT的好处：1.并行，不用像AT那样一个一个地输出（需要做多次decoder操作），因此速度比AT快。2.能更好地控制输出的长度。
NAT的缺陷：表现不如AT，原因是 Multi-modality，[**NAT参考视频**](https://www.youtube.com/watch?app=desktop&v=jvyKmU4OM3c&feature=youtu.be)。

![](https://user-images.githubusercontent.com/35519242/143690588-6bff23ba-f3e8-4699-8cc2-4f0ad3906f11.png)

##### *cross attention的历史及其他操作*

关于 cross attention，这个之前就有了，模型表现，吐出每一个字母时，关注范围从坐上到右下，很符合直觉。

​	![](https://user-images.githubusercontent.com/35519242/143691448-726c68f0-1b6a-4576-aeef-dc8563459582.png)

cross attention，transformer里面，encoder和decoder都有多层，但是每层decoder都是使用encoder最后一层的输出的。但其实可以考虑用encoder非最后一层的输出来作为输入的。

![](https://user-images.githubusercontent.com/35519242/143691587-9d94ef3f-ed1c-4fd9-913e-9ceb5f841f8a.png)

#### 训练

机被表示成一个one-hot的向量，只有机对应的那个维度为1，其余为0，这是正确答案。decoder的输出是一个distribution，是一个概率的分布，希望这个分布和one-hot越接近越好，所以计算cross entropy（交叉熵），希望这个值越小越好，该问题和分类很像。最后end也要计算，把5下图所示的5个交叉熵加起来，目标是最小化交叉熵和。

![](https://user-images.githubusercontent.com/35519242/143691835-33d8f346-9ff3-4ea3-900b-cc4b3ff3db88.png)

![](https://user-images.githubusercontent.com/35519242/143691902-dcfb1131-c079-469f-ad76-c0d8aec23289.png)

这里面还有一个 Teacher Forcing 的说法，如上图所示。

##### *tips*

- **Copy Mechanism**

  - Chat-bot

    比如在对话机器人中，有些奇怪的名词不需要生成，可以直接使用拷贝。

    ![Alt](https://user-images.githubusercontent.com/35519242/143693485-f82b8ccf-a987-4a91-9c18-1553676cd0a8.png)

  - summarization

    通常需要百万篇文章才够！所以之前做的病例生成项目，是无法使用端到端的模型来解决的。

    参考：

    [文章：Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)

    [视频：Pointer Network](https://www.youtube.com/watch?app=desktop&v=VdOyqNQ9aww&feature=youtu.be)

    [文章：Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393)

    ![Alt](https://user-images.githubusercontent.com/35519242/143695554-ec6510c8-88ae-4a7b-9749-62f5c1b809c4.png)

- guided attention

  Monotonic attention、Location-aware attention

- Beam Search

  简单说就是不必找所有路径。参考 [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)

- opimizing evaluation metrics

  训练的时候是 cross attention，所以是每个单独算loss，但是BLUE score是整体的，如何在训练的时候也用BLUE score来评估呢？***当你不知道怎么样optmize的时候，可以考虑reinforcement learning（RL, 强化学习）**，参考 [Sequence Level Training with Recurrent Neural Networks](https://arxiv.org/abs/1511.06732)

  ![Alt](https://user-images.githubusercontent.com/35519242/143727318-43464c9a-113b-4149-9f76-5556ee208ab0.png)

- exposure bias

  解决办法是给decoder的输入加一些错误的东西

​		![Alt](https://user-images.githubusercontent.com/35519242/143727343-dc5549ba-adb1-463c-87ac-8f06666f9c60.png)

​			scheduled sampling，参考方法如下

​		![Alt](https://user-images.githubusercontent.com/35519242/143727358-be2d13fb-c7cc-4407-a2b8-0ee871834d6e.png)

## *挖坑*

检索文章“挖个坑”，如下：

- fully connected 这里没搞懂，参数共享不。逐位的FNN？

  每一层经过attention之后，还会有一个FFN，这个FFN的作用就是**空间变换**。FFN包含了2层linear transformation层，中间的激活函数是ReLu。**引入了非线性(ReLu激活函数)，变换了attention output的空间, 从而增加了模型的表现能力**。把FFN去掉模型也是可以用的，但是效果差了很多。

- [**Word Embedding**](https://www.youtube.com/watch?v=X7PH3NuYW0Q)

- lay normalization 和 batch normalization 的详细区别

- 维度问题，*Wq*,*Wk*,*Wv* 三个参数是共享的吗？确认下

  $W_{i}^{Q} \in \mathbb{R}^{d_{\text {model }} \times d_{k}}, W_{i}^{K} \in \mathbb{R}^{d_{\text {model }} \times d_{k}}, W_{i}^{V} \in \mathbb{R}^{d_{\text {model }} \times d_{v}}$

  但是大小还没解决。

  或许可以参考 [链接1:Q，K，V进行了很详细的讲解](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImQ0ZTA2Y2ViMjJiMDFiZTU2YzIxM2M5ODU0MGFiNTYzYmZmNWE1OGMiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2MzgwOTEzMzYsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjExNjM2ODg4ODU5ODA0NTkzMDMxMyIsImVtYWlsIjoidHpsbHh6QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJuYW1lIjoiU3R1cGlkIEh1bWFuIiwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdoLW5EeUVWSl9jdWxXRzVjaXFrUmxTWXBPNmJwa1BYWFJoNkx0cj1zOTYtYyIsImdpdmVuX25hbWUiOiJTdHVwaWQiLCJmYW1pbHlfbmFtZSI6Ikh1bWFuIiwiaWF0IjoxNjM4MDkxNjM2LCJleHAiOjE2MzgwOTUyMzYsImp0aSI6IjhmMTBhNDYxMjQzOTk5MGIyNmQ5YjI2NGQ0ZmU0M2M2Y2Y1ZGE1MDcifQ.IXG9hvowUXF3v61G-dKpQTwpEFQCbaGU4pbeMJA8xRBMzsrqJjW9dAlyWONLmV2ku8Kizm21knX4EtL7-KOq1mVy8D47taP0Cfx3sR9nWD2yUfk3KotijIiHRigb2-M5idmo1ZGhpD7nLavDn_1oSmxoRGAvDJUERjNUKMujCtsk_uNio1Qz6q9nUYJdMaQg9B0jPza7g3enipe1ekUQx5W7kn9m-g_xmY1zl3gJk8K1RK9WBZd1KzlR1Qn4vcASGT1PPO7f3AuODVw6SYNK9ZWlQxAf36_iq6uy4OPkFNdSCxaNDCZzbGgkntD0cS3JvJVpZHA6EK2mOE5wPyAEbg)  [参考2](transformer中为什么使用不同的K 和 Q， 为什么不能使用同一个值？ - TniL的回答 - 知乎 https://www.zhihu.com/question/319339652/answer/1617078433)

- Positional Encoding
- embedding 层的参数需要在训练中学习吗
- 里面的矩阵操作，具体顺序，维度变化。

- 线性代数挖个坑吧，以后补一下，有很好的课程的。
- 残差 —— https://zhuanlan.zhihu.com/p/80226180?utm_source=wechat_session

## *补充Tips*

- 矩阵的点乘

  矩阵第m行与第n列交叉位置的那个值，等于第一个矩阵第m行与第二个矩阵第n列，对应位置的每个值的乘积之和。**矩阵的本质就是线性方程式，两者是一一对应关系**。

- 关于超参数

  ![Alt](https://user-images.githubusercontent.com/35519242/143730798-83389c0c-5705-4355-ab54-965468950ed2.png)

- **mask**

  Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。

  **padding mask**：每个批次输入序列长度是不一样的也就是说，我们要对输入序列进行对齐。具体来说，就是给在较短的序列后面填充 0。但是如果输入的序列太长，则是截取左边的内容，把多余的直接舍弃。因为这些填充的位置，其实是没什么意义的，所以我们的attention机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样的话，经过 softmax，这些位置的概率就会接近0。

  **Sequence mask**：time_step 为 t 的时刻，我们的解码输出应该只能依赖于 t 时刻之前的输出，而不能依赖 t 之后的输出。**产生一个上三角矩阵，上三角的值全为0。把这个矩阵作用在每一个序列上，就可以达到我们的目的**。

  ![](https://pic1.zhimg.com/80/v2-4d82d55512d23ca63a42ac5b85b8c494_720w.jpg)

  对于 decoder 的 self-attention，里面使用到的 scaled dot-product attention，同时需要padding mask 和 sequence mask 作为 attn_mask，具体实现就是两个mask相加作为attn_mask。其他情况，attn_mask 一律等于 padding mask。[**参考**](https://zhuanlan.zhihu.com/p/311156298)

- **Linear & Softmax**

  Decoder最后是一个线性变换和softmax层。解码组件最后会输出一个实数向量。我们如何把浮点数变成一个单词？这便是线性变换层要做的工作，它之后就是Softmax层。
  线性变换层是一个简单的全连接神经网络，它可以**把解码组件产生的向量投射到一个比它大得多的、被称作对数几率（logits）的向量里**。不妨假设我们的模型从训练集中学习一万个不同的英语单词（我们模型的“输出词表”）。因此对数几率向量为一万个单元格长度的向量——每个单元格对应某一个单词的分数（**相当于做vocaburary_size大小的分类**）。接下来的Softmax 层便会把那些分数变成概率（都为正数、上限1.0）。**概率最高的单元格被选中，并且它对应的单词被作为这个时间步的输出。**  [**参考**](https://zhuanlan.zhihu.com/p/311156298)

  ![](https://pic2.zhimg.com/80/v2-5ed7b1947af14225715fd25751133855_720w.jpg)

## 常见面试问答

- 为什么Transformer中K 、Q不能使用同一个值  [**参考**](https://www.cnblogs.com/jins-note/p/14508523.html)

- 为什么不直接使用$X$而要对其进行线性变换（通过*Wq*,*Wk*,*Wv* 得到q k v)

  为了提升模型的拟合能力，矩阵*Wq*,*Wk*,*Wv* 都是可以训练的，起到一个缓冲的效果。

- $\sqrt{d_{k}}$ 的意义

  ![](https://user-images.githubusercontent.com/35519242/143764642-ca5af1d2-3454-4ea1-92aa-ccd8fd90898a.png)

- **Multi-Head Attention** 怎么操作的（腾讯ieg游戏广告部门问到，答错了，一面挂）

  通过权重矩阵 ![[公式]](https://www.zhihu.com/equation?tex=W_i%5EQ%2CW_i%5Ek%2CW_i%5EV) 将![[公式]](https://www.zhihu.com/equation?tex=Q%2CK%2CV) 分割，每个头分别计算 single self-attention，因为权重矩阵 ![[公式]](https://www.zhihu.com/equation?tex=W_i%5EQ%2CW_i%5Ek%2CW_i%5EV) 各不相同， ![[公式]](https://www.zhihu.com/equation?tex=QW_i%5EQ%2CKW_i%5Ek%2CVW_i%5EV)的结果各不相同，因此我们说每个头的关注点各有侧重。最后，将每个头计算出的 single self-attention进行**concat**，通过总的权重矩阵$W^{O}$决定对每个头的关注程度，从而能够做到在不同语境下对相同句子进行不同理解。[**参考**](https://zhuanlan.zhihu.com/p/311156298)

- softmax的维度



- **Layer Normalization**作用，以及为什么不用 Batch Normalization 

  $L N\left(x_{i}\right)=\alpha \frac{x_{i}-\mu_{i}}{\sqrt{\sigma^{2}+\xi}}+\beta$，作用是**规范优化空间，加速收敛**。我们使用梯度下降算法做优化时，我们可能会对输入数据进行归一化，但是经过网络层作用后，我们的数据已经不是归一化的了。随着网络层数的增加，数据分布不断发生变化，偏差越来越大，导致我们不得不使用更小的学习率来稳定梯度。Layer Normalization 的作用就是保证数据特征分布的稳定性，将数据标准化到ReLU激活函数的作用区域，可以使得激活函数更好的发挥作用。[**参考**](https://zhuanlan.zhihu.com/p/311156298)

- **Residual Network 残差网络** 

  在神经网络可以收敛的前提下，随着网络深度的增加，网络表现先是逐渐增加至饱和，然后迅速下降，这就是我们经常讨论的**网络退化**问题，为了使当模型中的层数较深时仍然能得到较好的训练效果，模型中引入了残差网络。**暂时回答解决梯度消失问题**。

- **Transformer相比于RNN/LSTM，有什么优势**

  RNN系列的模型，并行计算能力很差。RNN并行计算的问题就出在这里，因为 T 时刻的计算依赖 T-1 时刻的隐层计算结果，而 T-1 时刻的计算依赖 T-2 时刻的隐层计算结果，如此下去就形成了所谓的序列依赖关系。

  Transformer的特征抽取能力比RNN系列的模型要好。
  具体实验对比可以参考：[放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)

- scaled

  $\sqrt{d}$ 代表 $k^{i}$ 的维度，除以$\sqrt{d}$ 的原因是：进行点乘后的数值很大，导致通过softmax后梯度变的很小，所以通过除以$\sqrt{d}$ 来进行缩放。

## *代码细节*

*待续......*
看源码就知道很多细节了，一定要看，欠账必还。

看下 https://zhuanlan.zhihu.com/p/411311520 



## 参考

- ......
