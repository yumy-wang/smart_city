from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result_pyplot

root_dir = '/mmdetection/mmdetection/'

# 模型配置文件
config_file = root_dir + 'work_dir/cascade_rcnn_base_train_log/cascade_rcnn_r50_fpn_1x_coco.py'

# 预训练模型文件
checkpoint_file = root_dir + 'work_dir/cascade_rcnn_base_train_log/latest.pth'

# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并进行展示

imgtest = root_dir + 'data/food_60/train/270001401.jpg'
print(imgtest)
result = inference_detector(model, imgtest)
# model.show_result(img, result, model.CLASSES)
print(result)
show_result_pyplot(model, imgtest, result)