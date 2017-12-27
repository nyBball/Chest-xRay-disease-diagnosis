# Chest-xRay-disease-diagnosis
## 项目需求：
输入一张胸腔x光片，输出该x光片有病的概率，并且对于有病的图片标出病灶位置。
## 项目指标：
在二分类fn=1%的情况下，期望得到更高的没病图片查全率。由于在实际情况下，没病图片和有病图片的数量比例约为1:5甚至差异更大，也就是说有病图片的数量远少于没病图片数量，如果单追求整个数据集上的准确率的话，意义不大。
## 环境配置：
编程语言：python2.7

库：keras2.0.2, tensorflow-GPU, SimpleITK, numpy, matplotlib, scipy

## 项目难点：
- 高质量胸腔X光片数据集匮乏。个人认为学术上x光片的工作远少于CT的主要原因在于缺少高质量公开的数据集，针对近期公开的ChestXray14X光胸片数据集，一些医学专家认为目前的ChestXray14并不适合训练医疗AI系统进行诊断工作，详情参阅[https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/](https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/)

- 疾病特征不明显，如下图所示，标注区域为肺部感染病症，肉眼很难发现特征![](https://i.imgur.com/0PAGMMf.png)

- 疾病尺度过多，单就肺部感染一种病来看，本团队自建的312张有病图片病灶区域面积、宽长比统计直方图如下图所示![](https://i.imgur.com/TrynJLU.png)

## 项目解决方案：
只针对肺部感染一种病来做，选用如下表所示的14个尺度![](https://i.imgur.com/SixhCwI.png)

- 训练过程

	一次读入一张abnormal图片和一张normal图片，先进行z-score标准化，对于abnormal图片，根据病灶的面积和宽长比从预设的14个尺度中选择最相似的一个尺度，用该尺度在病灶的附近随机裁剪n数量的与病灶矩形框的Iou>0.7的abnormal图像块，同时用该尺度在abnormal图片normal区域随机裁剪n数量的normal图像块，同时也用该尺度在normal图片上随机裁剪2n数量的normal图像块（保证正负样本比例约为1:3）。裁剪完毕后，将这4n数量的图像块送入inception v3（去掉全连接层并加入空间金字塔池化层）进行fine-tune，训练4个epoch“热身”后，再采用online hard example mining策略训练。![](https://i.imgur.com/0VU4ink.png)

- 测试过程

	对于每张图片，用预设的14个尺度分别进行16x16裁剪，将裁剪得到的图像块输入网络测试得到每个图像块有病的概率，取所有图像块有病的最大概率值作为该张图片的最终有病概率值。


## 项目构成：
- config.py 

	环境变量配置文件。包括训练集路径、验证集路径、x光图片标注文件路径、预训练模型权重路径等。

- utils.py 
	
	常用函数定义文件。其中包括读取图片、计算Iou、裁剪图片等。

- generator.py

	训练、测试生成器函数文件。主要包含两部分：train生成器和test生成器。对于train生成器，TrainBatchGenerator/ValBatchGenerator、all_resize_generator、onlineLocalGenerator、onlineMultiScaleSPPGenerator分别意指10x10裁剪训练、resize成512x512训练、在线单一尺度切割后resize训练、在线多尺度切割后SPP池化训练。对于test生成器，test_local_generator、test_global_generator、test_multiScale_SPP_generator分别意指10x10裁剪测试、resize成512x512测试、在线多尺度切割后SPP池化测试。

- myinceptionv3.py

	googleNet inception_v3网络构建文件。与keras内置网络构建文件相比，主要区别是在卷积层加入正则项，以避免训练过程中出现的过拟合。

- preProcessData.py

	对数据预处理文件。其中countDises函数用来读取图片病变属性信息，并保存到txt文件；moveDisesRatio函数用来按病变面积与总面积的比是否超过0.1区分成global、local两类疾病；plotHist函数用来画正交比、面积统计直方图，以决定多尺度训练时选择哪些尺度；countCropANratio函数用来统计10x10裁剪后，abnormal patch和normal patch数量比例，以选择对abnormal patch增广的倍数，以平衡正负样本比例约为1:3；saveLocalPatch函数用来离线保存10x10裁剪训练集patch，以用于10x10裁剪训练； saveGlobalPatch用来离线保存resize 512x512 patch，以用于直接resize成512x512图片训练。

- proProcessResult.py

	对测试数据后处理文件。其中countMaxPro函数用来统计单一尺度每张图片最大有病概率；countFNnormalNum函数用来统计单一尺度FN取固定值时，normal图片判错个数；countMulScalePro函数用来统计多尺度方法时每张图片每个尺度最大有病概率；countMultiScaleFNnormalNum函数用来统计多尺度FN取固定值时，normal图片判错个数。

- SpatialPyramidPooling.py

	空间金字塔池化层定义文件。

- train_inceptionV3.py

	训练网络模型文件。其中包括SGD学习率衰减速率设置、模型构建及参数设置以及第一批数据all resize into 512x512训练、第一批数据10x10crop训练、第二批数据集10x10crop训练、肺部感染多尺度训练、第二批数据集多尺度训练等操作。

- test.py

	载入预训练好的模型在测试集上进行测试文件。其中包括第一批数据集all resize into 512x512测试、第一批数据集/第二批数据集10x10crop测试、肺部感染多尺度测试、第二批数据集多尺度测试等操作。

- testSingleImg.py

	载入预训练好的模型，用10x10裁剪的方法测试单张图片有病概率的文件。


## 实验效果：
- 数据集：312张abnormal图像和1500张normal图像。
- 准确率：5-fold交叉验证，在fn=1%的情况下，没病图片查全率可以达到30%左右。整体数据集，按训练过程切块后判别准确率可以达到88.23%
![](https://i.imgur.com/9SmG0mM.png)



