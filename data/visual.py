import json
import os, cv2

# train_json = r'C:\Users\mage\Desktop\detectron2-maskrcnn\datasets\coco\annotations\instances_test2014.json'
# train_path = r'C:\Users\mage\Desktop\detectron2-maskrcnn\datasets\coco\test2014'
# visual_output = r'C:\Users\mage\Desktop\SolarCOCO\visual\test'





#  可视化coco格式json标注中的box到图片上
import json
import shutil
import cv2

def select(json_path, outpath, image_path):
    json_file = open(json_path)
    infos = json.load(json_file)
    images = infos["images"]
    annos = infos["annotations"]
    assert len(images) == len(images)
    for i in range(len(images)):
        im_id = images[i]["id"]
        im_path = image_path + "/" + images[i]["file_name"]
        img = cv2.imread(im_path)
        for j in range(len(annos)):
            if annos[j]["image_id"] == im_id:
                x, y, w, h = annos[j]["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)
                x2, y2 = x + w, y + h
                # object_name = annos[j][""]
                img = cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), thickness=2)
                img_name = outpath + "/" + images[i]["file_name"]
                cv2.imwrite(img_name, img)
                print("保存成功")
                # continue
        print(i)

if __name__ == "__main__":

    # train_json = r'D:\PY_SCI\Drawing board\data\COCO\annotations\train.json'
    # train_path = r'D:\PY_SCI\Drawing board\data\COCO\train'
    # visual_output = r'D:\PY_SCI\Drawing board\data\COCO\visual\train'

    # train_json = r'D:\PY_SCI\Drawing board\data\COCO\annotations\test.json'
    # train_path = r'D:\PY_SCI\Drawing board\data\COCO\test'
    # visual_output = r'D:\PY_SCI\Drawing board\data\COCO\visual\test'
    #
    train_json = "/home/hzy/NAT/detection/data/aluminum/annotations/instances_train2017.json"
    train_path = "/home/hzy/NAT/detection/data/aluminum/train2017"
    # visual_output = "/home/hzy/NAT/detection/data/NEU-DET_visual"
    visual_output = "/home/hzy/NAT/detection/data/al_visual"
    # json_path = "/Users/wanghao/Desktop/Tianchi_bottle/train1/fix_anno.json"
    # out_path = "/Users/wanghao/Desktop/Tianchi_bottle/train1/vis"
    # image_path = "/Users/wanghao/Desktop/Tianchi_bottle/train1/images"


    select(train_json, visual_output, train_path)

