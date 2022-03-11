# A03智能冰箱食材精准识别

## 比赛说明：

- 背景：“冰箱食材精准识别”赛题是冰箱场景食材检测问题，即给定若干冰箱场景下的图片，食材检测算法需要在图像中定位到人手中的食材并识别出类别。
- 作品要求：参赛团队应聚焦智能冰箱场景食材定位识别，要求选手在给定的训练数据集上开发出高效可靠的计算机视觉算法，实现食材的定位识别，要求模型尽可能快而准的给出食材的`位置`和`类别`。【目标检测问题】

- 评价指标：mAP（mean Average Precision）
- 初赛数据集：
	- 训练集：60类，3w张1280x720像素
		- 每个类文件夹下，图片+其对应txt包含类别id和bbox
	- 测试集：6k张1280x720像素
- 决赛数据集：
	- 训练集：150类，10w张1280x720像素
	- 测试集：2w张

## 数据处理

比赛给的数据格式是yolo格式，而mmdet支持的是coco格式，因此需要先进行数据转换

### step1：将数据转换成标准yolo格式

代码：code/data_preprocess.ipynb

- 标准yolo：classes.txt  images  labels三个文件/文件夹
	- classes.txt中放入每个类的名称，一行一类
	- images和labels中文件名一一对应，只有后缀不一样



### step2：使用工具将yolo转换成coco

代码：/mmdetection/mmdetection/data/yolo2coco.py

执行

```bash
python yolo2coco.py --root_dir /mmdetection/mmdetection/data/coco/ --random_split
```

分割后：

![image-20210829155452369](https://yumytest.oss-cn-chengdu.aliyuncs.com/img/image-20210829155452369.png)

## 环境搭建

这里直接使用mmdet的docker，然后在安装一些常用工具比如ssh、jupyter

cocoAPI安装

```bash
python setup.py build_ext install
```



## 修改demo模型

/mmdetection/mmdetection/mmdet/datasets/coco.py下CLASSES修改成自己的类

![image-20210825162405426](https://yumytest.oss-cn-chengdu.aliyuncs.com/img/image-20210825162405426.png)

/mmdetection/mmdetection/mmdet/core/evaluation/class_names.py下修改coco_classes的return

![image-20210825162439623](https://yumytest.oss-cn-chengdu.aliyuncs.com/img/image-20210825162439623.png)

/mmdetection/mmdetection/configs/_base_/models/cascade_rcnn_r50_fpn.py下所有的num_classes修改

![image-20210825162603926](https://yumytest.oss-cn-chengdu.aliyuncs.com/img/image-20210825162603926.png)

因为是demo文件，所以data格式必须这样命名：

![image-20210825162736455](https://yumytest.oss-cn-chengdu.aliyuncs.com/img/image-20210825162736455.png)

## 训练和测试

https://www.jianshu.com/p/42891e5a0422

https://blog.csdn.net/weixin_41010198/article/details/106258366

### train

```bash
# --work-dir为保存的权重路径，默认为work_dirs
CUDA_VISIBLE_DEVICES=5 nohup python tools/train.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py --work-dir work_dir &
```

多GPU

```bash
CUDA_VISIBLE_DEVICES=0,5 nohup bash ./tools/dist_train.sh configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py 2 --work-dir work_dir/cascade_rcnn_pretrain_train_log &> train_pretrain_log &
```



### test

```bash
CUDA_VISIBLE_DEVICES=6 nohup python tools/test.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py work_dir/cascade_rcnn_base_train_log/latest.pth --out work_dir/result.pkl --eval bbox &> testlog1 &
```



```bash
# 可视化GUI下
CUDA_VISIBLE_DEVICES=6 nohup python tools/test.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py work_dir/latest.pth --show
```



多GPU

```bash
CUDA_VISIBLE_DEVICES=0,5 nohup bash tools/dist_test.sh configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py     work_dir/latest.pth 2 --out work_dir/base_model_result.pkl --eval mAP  &> test_testlog &
```



### test!

```bash
CUDA_VISIBLE_DEVICES=4 nohup python tools/test.py work_dir/cascade_rcnn_r50_1x_dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_voc.py work_dir/cascade_rcnn_r50_1x_dcn/latest.pth --format-only --eval-options "jsonfile_prefix=./cascade_rcnn_test_results"  &> work_dir/cascade_rcnn_r50_1x_dcn/out.log  &


```



## 可视化

### 训练过程可视化

```bash
tensorboard --logdir=work --host=127.0.0.1 --port=8097
```

### 画loss图

```bash
python tools/analysis_tools/analyze_logs.py plot_curve work_dir/cascade_rcnn_base_train_log/20210830_133546.log.json  --keys loss --out losses.pdf

python tools/analysis_tools/analyze_logs.py plot_curve work_dir/cascade_rcnn_base_train_log/20210830_133546.log.json  --keys bbox_mAP  --out bbox_mAP.pdf

# 画两个图
python tools/analysis_tools/analyze_logs.py plot_curvework_dir/cascade_rcnn_base_train_voc_log/20210901_075427.log.json work_dir/cascade_rcnn_base_r101_voc_train_log/20210902_015637.log.json work_dir/cascade_rcnn_base_train_e20_voc_log/20210902_145448.log.json work_dir/cascade_rcnn_base_r101_e20_voc_train_log/20210904_030906.log.json  work_dir/cascade_rcnn_pretrain_train_e20_voc_log/20210905_070828.log.json work_dir/cascade_rcnn_r50_1x_dcn/20210907_084612.log.json --keys mAP --legend r50 r101 r50+e20 r101+e20 e20+pre new --out mAP.pdf
```

### 查看mAP

```bash
CUDA_VISIBLE_DEVICES=6 nohup python my_voc_eval.py work_dir/cascade_rcnn_base_train_voc_log/result.pkl work_dir/cascade_rcnn_base_train_voc_log/cascade_rcnn_r50_fpn_1x_voc.py &
```

### Flops

```bash
CUDA_VISIBLE_DEVICES=1 python tools/analysis_tools/get_flops.py work_dir/cascade_rcnn_base_r101_e20_voc_train_log/cascade_rcnn_r101_fpn_20e_voc.py --shape 1280 720
```



## 具体代码

### 消融实验

#### r50

- [x] r50==训练==

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup bash ./tools/dist_train.sh configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_voc.py 4 --work-dir work_dir/cascade_rcnn_base_train_voc_log &> train_base_voc_log &
```

- [x] r50==测试==（mAP_thr要在evaluation中修改成数组[0.3, 0.5, 0.7]）

```bash
CUDA_VISIBLE_DEVICES=5 nohup python tools/test.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_voc.py work_dir/cascade_rcnn_base_train_voc_log/latest.pth --out work_dir/cascade_rcnn_base_train_voc_log/result_357.pkl --eval mAP &> test_base_voc_357_log &
```

#### r101

- [x] r101==训练==

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup bash ./tools/dist_train.sh configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_voc.py 4 --work-dir work_dir/cascade_rcnn_base_r101_voc_train_log &> train_r101_voc_log &
```

- [x] r101==测试==（mAP_thr要在evaluation中修改成数组[0.3, 0.5, 0.7]）

```bash
CUDA_VISIBLE_DEVICES=5 nohup python tools/test.py configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_voc.py work_dir/cascade_rcnn_base_r101_voc_train_log/latest.pth --out work_dir/cascade_rcnn_base_r101_voc_train_log/result_357.pkl --eval mAP &> test_base_r101_voc_357_log &
```

#### r50+e20

- [x] r50+e20==训练==

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup bash ./tools/dist_train.sh configs/cascade_rcnn/cascade_rcnn_r50_fpn_20e_voc.py 4 --work-dir work_dir/cascade_rcnn_base_train_e20_voc_log &> train_base_e20_voc_log &
```

- [x] r50+e20==测试==

```bash
CUDA_VISIBLE_DEVICES=5 nohup python tools/test.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_20e_voc.py work_dir/cascade_rcnn_base_train_e20_voc_log/latest.pth --out work_dir/cascade_rcnn_base_train_e20_voc_log/result_357.pkl --eval mAP &> test_base_e20_voc_357_log &
```

#### r101+e20

- [x] r101+e20==训练==

```bash
CUDA_VISIBLE_DEVICES=1,5,3,4 nohup bash ./tools/dist_train.sh configs/cascade_rcnn/cascade_rcnn_r101_fpn_20e_voc.py 4 --work-dir work_dir/cascade_rcnn_base_r101_e20_voc_train_log &> train_base_r101_e20_voc_log &
```

- [x] r101+e20==测试==

```bash
CUDA_VISIBLE_DEVICES=1 nohup python tools/test.py configs/cascade_rcnn/cascade_rcnn_r101_fpn_20e_voc.py work_dir/cascade_rcnn_base_r101_e20_voc_train_log/latest.pth --out work_dir/cascade_rcnn_base_r101_e20_voc_train_log/result_357.pkl --eval mAP &> test_base_r101_e20_voc_357_log &
```

#### x101+32x4d+1x

- [x] x101 32x4d+1x==训练==

```bash
CUDA_VISIBLE_DEVICES=1,5,3,4 nohup bash ./tools/dist_train.sh configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_voc.py 4 --work-dir work_dir/cascade_rcnn_x101_32x4d_fpn_1x_voc_train_log &> train_cascade_rcnn_x101_32x4d_fpn_1x_voc_log &
```

- [x] x101 32x4d+1x==测试==

```bash
CUDA_VISIBLE_DEVICES=1 nohup python tools/test.py configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_voc.py work_dir/cascade_rcnn_x101_32x4d_fpn_1x_voc_train_log/latest.pth --out work_dir/cascade_rcnn_x101_32x4d_fpn_1x_voc_train_log/result_357.pkl --eval mAP &> test_cascade_rcnn_x101_32x4d_fpn_1x_voc_357_log &
```

#### x101+64x4d+1x【终止】


- [ ] x101 64x4d+1x==训练==

```bash
CUDA_VISIBLE_DEVICES=0,6,7 nohup bash ./tools/dist_train.sh configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_voc.py 3 --work-dir work_dir/cascade_rcnn_x101_64x4d_fpn_1x_voc_train_log &> train_cascade_rcnn_x101_64x4d_fpn_1x_voc_log &
```

- [ ] x101 64x4d+1x==测试==

```bash
CUDA_VISIBLE_DEVICES=1 nohup python tools/test.py configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_voc.py work_dir/cascade_rcnn_x101_64x4d_fpn_1x_voc_train_log/latest.pth --out work_dir/cascade_rcnn_x101_64x4d_fpn_1x_voc_train_log/result_357.pkl --eval mAP &> test_cascade_rcnn_x101_64x4d_fpn_1x_voc_357_log &
```





### 预训练r50_20e

- [ ] 预训练训练（标记预训练模型load_from）

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup bash ./tools/dist_train.sh configs/cascade_rcnn/cascade_rcnn_r50_fpn_20e_voc.py 4 --work-dir work_dir/cascade_rcnn_pretrain_train_e20_voc_log &> train_pretrain_e20_voc_log &
```

- [ ] 预训练测试

```bash
CUDA_VISIBLE_DEVICES=5 nohup python tools/test.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_20e_voc.py work_dir/cascade_rcnn_pretrain_train_e20_voc_log/latest.pth --out work_dir/cascade_rcnn_pretrain_train_e20_voc_log/result_357.pkl --eval mAP &> test_pretrain_e20_voc_357_log &
```

## 测试代码



```bash
# 测试coco数据集是否输出结果
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_test.sh abandon_work_dir/coco_work_dir/cascade_rcnn_base_train_log/cascade_rcnn_r50_fpn_1x_coco.py abandon_work_dir/coco_work_dir/cascade_rcnn_base_train_log/latest.pth 8 --format-only --eval-options "jsonfile_prefix=./out"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_test.sh /mmdetection/mmdetection/work_dir/cascade_rcnn_r50_1x_dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_voc.py     /mmdetection/mmdetection/work_dir/cascade_rcnn_r50_1x_dcn/latest.pth 8 --format-only --eval-options "jsonfile_prefix=./out"
```



CUDA_VISIBLE_DEVICES=4,5 nohup bash ./tools/dist_train.sh configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py 2 --work-dir zzz/coco_cas &> zzz/coco_cas/train_log &



# 跑代码检查

1. 数据集预处理+加载路径
2. 模型策略一般不改
3. 运行时是否有预加载模型load_from

单GPU

```bash
CUDA_VISIBLE_DEVICES=5 nohup python tools/train.py \
configs/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py \
--work-dir work/dcn_data_aug \
&> work/dcn_data_aug/train.log &
```

多GPU

```bash
#train
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 nohup bash ./tools/dist_train.sh \
configs/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py \
6 \
--work-dir work/test1 \
&> work/test1/train.log &



# test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup bash tools/dist_test.sh \
/mmdetection/mmdetection/work/dcn_r50_1x_base/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py \
/mmdetection/mmdetection/work/dcn_r50_1x_base/latest.pth \
6 \
--format-only \
--eval-options "jsonfile_prefix=./dcn_r50_1x_base" \
&> work/dcn_r50_1x_base/test.log &


# visualize the results, save images to the directory results/
python tools/analysis_tools/analyze_results.py \
       configs/xxxxxxx.py \
       result.pkl \
       results \
       --show




```

# coco

基本：

```bash
CUDA_VISIBLE_DEVICES=2,3 nohup bash ./tools/dist_train.sh \
configs/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py \
2 \
--work-dir work/dcn_r50_1x_base \
&> work/dcn_r50_1x_base/train.log &
```

```bash
# test下生成json结果
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash tools/dist_test.sh /mmdetection/mmdetection/work/dcn_r50_1x_base/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py /mmdetection/mmdetection/work/dcn_r50_1x_base/latest.pth 6 --out work/dcn_r50_1x_base/val_out/result.pkl   --format-only--options "jsonfile_prefix=./dcn_r50_1x_base"

# 改test为val，生成val的json和pkl
# CUDA_VISIBLE_DEVICES=1,3 bash tools/dist_test.sh /mmdetection/mmdetection/work/dcn_r50_1x_base/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py /mmdetection/mmdetection/work/dcn_r50_1x_base/latest.pth 2 --out work/dcn_r50_1x_base/val_out/result.pkl   --format-only  --options "jsonfile_prefix=work/dcn_r50_1x_base/val_out/val_out"

# 错误分析
# python tools/analysis_tools/coco_error_analysis.py \
#        work/dcn_r50_1x_base/val_out/val_out.bbox.json \
#        zzz/error \
#        --ann /mmdetection/mmdetection/data/food_60/annotations/val.json \

nohup python tools/analysis_tools/analyze_results.py work/dcn_r50_1x_base/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py work/dcn_r50_1x_base/val_out/result.pkl work/dcn_r50_1x_base/analyze_results/ --show --topk 50 &> nohup.out &
# 不太能用good全是置信度很低的。。。

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup bash tools/dist_test.sh \
/mmdetection/mmdetection/work/dcn_r50_1x_base/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py \
/mmdetection/mmdetection/work/dcn_r50_1x_base/latest.pth \
6 \
--out work/dcn_r50_1x_base/result.pkl  --eval bbox \
&> work/dcn_r50_1x_base/result.log &



```



基本+data_aug：修改数据增强代码

```bash
CUDA_VISIBLE_DEVICES=6,7 nohup bash ./tools/dist_train.sh \
configs/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py \
2 \
--work-dir work/dcn_r50_1x_dataAug \
&> work/dcn_r50_1x_dataAug/train.log &
```

基本+data_aug+pretrain：修改数据增强代码+修改load_from dcnr50

```bash
CUDA_VISIBLE_DEVICES=4,5 nohup bash ./tools/dist_train.sh \
configs/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py \
2 \
--work-dir work/dcn_r50_1x_dataAug+pretrain \
&> work/dcn_r50_1x_dataAug+pretrain/train.log &
```

基本+data_aug+pretrain+r101

```bash
CUDA_VISIBLE_DEVICES=6,7 nohup bash ./tools/dist_train.sh \
configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py \
2 \
--work-dir work/dcn_r101_1x_dataAug+pretrain \
&> work/dcn_r101_1x_dataAug+pretrain/train.log &
```

基本+data_aug+pretrain+r101+20e

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash ./tools/dist_train.sh \
configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_20e_coco.py \
4 \
--work-dir work/dcn_r101_20e_dataAug+pretrain \
&> work/dcn_r101_20e_dataAug+pretrain/train.log &
```

基本+data_aug2：修改数据增强代码

```bash
CUDA_VISIBLE_DEVICES=5,6,7 nohup bash ./tools/dist_train.sh \
configs/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py \
3 \
--work-dir work/dcn_r50_1x_dataAug2 \
&> work/dcn_r50_1x_dataAug2/train.log &
```

基本+data_aug3：修改数据增强代码

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash ./tools/dist_train.sh \
configs/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py \
4 \
--work-dir work/dcn_r50_1x_dataAug3 \
&> work/dcn_r50_1x_dataAug3/train.log &

```

## 消融实验

### 去掉DCN：（fpn+soft-nms）

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash ./tools/dist_train.sh \
configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py \
4 \
--work-dir work/cascade_r50_fpn_1x \
&> work/cascade_r50_fpn_1x/train.log &
```

### fpn+DCN

修改soft nms配置为nms：

```bash
CUDA_VISIBLE_DEVICES=6,7 nohup bash ./tools/dist_train.sh \
configs/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py \
2 \
--work-dir work/fpn+dcn \
&> work/fpn+dcn/train.log &
```



### 去掉soft-nms：（fpn）

修改soft nms配置为nms：

```bash
CUDA_VISIBLE_DEVICES=4,5 nohup bash ./tools/dist_train.sh \
configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py \
2 \
--work-dir work/fpn \
&> work/fpn/train.log &
```



### 去掉fpn(二阶段不能去掉，试试faster？)





## 对比实验

faster

```bash
CUDA_VISIBLE_DEVICES=6,7 nohup bash ./tools/dist_train.sh \
configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
2 \
--work-dir work/faster \
&> work/faster/train.log &
```

retinanet

```bash
CUDA_VISIBLE_DEVICES=4,7 nohup bash ./tools/dist_train.sh \
configs/retinanet/retinanet_r50_fpn_1x_coco.py \
2 \
--work-dir work/retinanet \
&> work/retinanet/train.log &
```

retinanet+dataAug【2021年10月25日】

```bash
CUDA_VISIBLE_DEVICES=4,7 nohup bash ./tools/dist_train.sh \
configs/retinanet/retinanet_r50_fpn_1x_coco.py \
2 \
--work-dir work/retinanet \
&> work/retinanet/train.log &
```

faster+dataAug

```bash
```

cascade+dataAug



# 常用工具

## test生成标注信息

p.s.测试只能单GPU才能运行，不知道为啥。。

```bash
CUDA_VISIBLE_DEVICES=3 python tools/test.py work/dcn_r50_1x_base/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py work/dcn_r50_1x_base/latest.pth --show-dir work/dcn_r50_1x_base/show_test/ --show-score-thr 0.6
```

生成一部分的话，直接中途强制中断就行

## 生成json结果文件

```bash
CUDA_VISIBLE_DEVICES=3,4,5 bash tools/dist_test.sh \
work/dcn_r50_1x_dataAug+pretrain/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py \
work/dcn_r50_1x_dataAug+pretrain/latest.pth \
3 \
--format-only \
--eval-options "jsonfile_prefix=work/dcn_r50_1x_dataAug+pretrain/dcn_r50_1x_dataAug+pretrain" \
```

这个文件即使设置了show-score-thr里很多低分的也筛选进来了，需要在输出csv中设置阈值

## 生成excel结果文件

需要先生成

```bash
python tools/post_process/json2submit.py --test_json work/dcn_r50_1x_dataAug+pretrain/dcn_r50_1x_dataAug+pretrain.bbox.json --submit_file dcn_r50_1x_dataAug+pretrain.csv --score 0.6
```



## 生成val的pkl结果

先讲配置文件的test改成val（==两处==）

```bash
```

## 绘制多个模型训练的mAP趋势

```bash
# 综合分析
python tools/analysis_tools/analyze_logs.py plot_curve work/dcn_r50_1x_base/20210921_004654.log.json work/dcn_r50_1x_dataAug/20210921_004758.log.json work/dcn_r50_1x_dataAug+pretrain/20210921_011221.log.json --keys bbox_mAP --legend base aug aug+pre --out aa.pdf
```

