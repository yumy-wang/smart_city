import json
import os
import argparse

# underwater_classes = ['1苹果', '2香蕉', '3火龙果', '4雪花梨', '5圣女果', '6猕猴桃', '7青芒', '8葡萄', '9玉米', '10绿甘蓝', '11紫甘蓝', '12鲜切紫甘蓝', '13花菜', '14西蓝花', '15西红柿', '16贝贝南瓜', '17金南瓜', '18青辣椒', '19绿圆椒', '20红圆椒', '21黄圆椒', '22茄子', '23西葫芦', '24秋葵', '25胡萝卜', '26鹌鹑蛋', '27木瓜', '28鲜切木瓜', '29菠菜', '30生菜', '31油菜', '32哈密瓜', '33鲜切哈密瓜', '34平菇', '35青萝卜', '36娃娃菜', '37鸡蛋', '38黄瓜', '39黄芒', '40青提', '41蓝莓', '42草莓', '43桂圆', '44山楂', '45红樱桃', '46水蜜桃', '47油桃', '48百香果', '49李子', '50牛油果', '51山竹', '52橙子', '53黄皮橘子', '54柠檬', '55柚子', '56啤梨', '57香梨', '58豆芽', '59油麦菜', '60芹菜']
underwater_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant', '', 
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', '',
               'backpack', 'umbrella', '', '', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', '', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# SCORE = 0.6
def parse_args():
    parser = argparse.ArgumentParser(description='json2submit_nms')
    parser.add_argument('--test_json', help='test result json', type=str)
    parser.add_argument('--submit_file', help='submit_file_name', type=str)
    parser.add_argument('--score', help='threahold', type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    test_json_raw = json.load(open("data/coco/annotations/test.json", "r"))
    test_json = json.load(open("./" + args.test_json, "r"))
    submit_file_name = args.submit_file
    submit_path = 'submit/'
    os.makedirs(submit_path, exist_ok=True)
    img = test_json_raw['images']
    images = []
    csv_file = open(submit_path + submit_file_name, 'w', encoding='utf-8-sig')
    csv_file.write("name,image_id,confidence,xmin,ymin,xmax,ymax\n")
    imgid2anno = {}
    imgid2name = {}
    for imageinfo in test_json_raw['images']:
        imgid = imageinfo['id']
        imgid2name[imgid] = imageinfo['file_name']
    for anno in test_json:
        img_id = anno['image_id']
        if img_id not in imgid2anno:
            imgid2anno[img_id] = []
        imgid2anno[img_id].append(anno)
    for imgid, annos in imgid2anno.items():
        for anno in annos:
            xmin, ymin, w, h = anno['bbox']
            xmax = xmin + w
            ymax = ymin + h
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            confidence = anno['score']
            if confidence < args.score:
                continue
            class_id = int(anno['category_id'])
            # print(class_id)
            class_name = underwater_classes[class_id-1]
            image_name = imgid2name[imgid]
            image_id = (image_name.split('.')[0] + '.jpg')
            csv_file.write(class_name + ',' + image_id + ',' + str(confidence) + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + '\n')
    csv_file.close()