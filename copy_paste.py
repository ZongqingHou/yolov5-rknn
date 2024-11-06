import glob
import os.path

import albumentations as A
import cv2
import numpy as np
import random
import copy
import tqdm


def crop_image(image, x, y, width, height):
    cropped_image = image[y:y + height, x:x + width]
    return cropped_image


def convert_to_absolute(label, image_width, image_height):
    class_id, relative_x_center, relative_y_center, relative_width, relative_height = label

    # 计算边界框的绝对坐标
    absolute_x_center = relative_x_center * image_width
    absolute_y_center = relative_y_center * image_height
    absolute_width = relative_width * image_width
    absolute_height = relative_height * image_height

    # 计算边界框的左上角和右下角坐标
    left = absolute_x_center - absolute_width / 2
    top = absolute_y_center - absolute_height / 2
    right = absolute_x_center + absolute_width / 2
    bottom = absolute_y_center + absolute_height / 2

    # 返回绝对坐标形式的边界框
    return [class_id, left, top, right, bottom]


def convert_to_yolo_format(class_id, left, top, right, bottom, image_width, image_height):
    # 计算目标框的中心点坐标和宽高
    x = (left + right) / 2
    y = (top + bottom) / 2
    width = right - left
    height = bottom - top

    # 将坐标和尺寸归一化到[0, 1]之间
    x /= image_width
    y /= image_height
    width /= image_width
    height /= image_height

    # 返回Yolo格式的标注
    return f"{class_id} {round(x,3)} {round(y,3)} {round(width,3)} {round(height,3)}\n"


# set sources file path
def get_src():
    img_list = glob.glob(r"/Users/nnddpp/datasets/tools/colorring_clean/images/*.jpg")
    random.shuffle(img_list)
    img_path = img_list[0]
    txt_path = img_list[0].replace("images", "labels").replace(".jpg", ".txt")
    return img_path, txt_path


# target file folder path
img_list = glob.glob(r"/Users/nnddpp/datasets/tools/tool_clean_rename_with_colorring/images/*.jpg")
for img_b_path in tqdm.tqdm(img_list):
    img_a_path, img_a_txt = get_src()
    image_a = cv2.imread(img_a_path)
    image_height, image_width, _ = image_a.shape
    img_b_txt = img_b_path.replace(".jpg", ".txt")
    img_b_path_new = img_b_path

    image_b = cv2.imread(img_b_path)
    res_list = []

    src_location_map = []
    bbox = []
    bbox_classes = []
    if os.path.exists(img_a_txt):
        pass
    else:
        continue

    with open(img_a_txt) as f:
        for line_str in f:
            line_info = line_str.strip().split(" ")
            bbox_classes.append(line_info[0])
            bbox.append([float(line_info[1]), float(line_info[2]), float(line_info[3]), float(line_info[4])])

    for _ in range(10):
        trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.OneOf([
            #     A.GaussNoise(),  # 将高斯噪声应用于输入图像。
            # ], p=0.2),  # 应用选定变换的概率
            A.OneOf([
                A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
            ], p=0.2),
            # A.ShiftScaleRotate(shift_limit=0.01, scale_limit=(-0.25, 0.1), rotate_limit=15, p=1),
            A.RandomBrightnessContrast(p=0.1),  # 随机明亮对比度
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["bbox_classes"]))

        try:
            trans_result = trans(image=image_a, bboxes=bbox, bbox_classes=bbox_classes)
            image_a = trans_result["image"]
            image_height, image_width, _ = image_a.shape

            for index, tmp_transformed_bbox in enumerate(trans_result["bboxes"]):
                label = [bbox_classes[0],
                         tmp_transformed_bbox[0],
                         tmp_transformed_bbox[1],
                         tmp_transformed_bbox[2],
                         tmp_transformed_bbox[3]]
                class_id, left, top, right, bottom = convert_to_absolute(label, image_width, image_height)
                if left or top or right or bottom:
                    x = int(left)  # 指定区域的起始横坐标
                    y = int(top)  # 指定区域的起始纵坐标
                    width = int(right - left)  # 指定区域的宽度
                    height = int(bottom - top)  # 指定区域的高度
                    cropped_image_a = crop_image(image_a, int(x), int(y), int(width), int(height))
                    # cv2.imshow("tt", cropped_image_a)
                    # cv2.waitKey(0)
                    image_b_height, image_b_width, _ = image_b.shape

                    with open(img_b_txt, "a") as f:
                        b_x = random.randint(0, int(image_b_width - width - 5))
                        b_y = random.randint(0, int(image_b_height - height - 5))
                        # tmp_value_index = np.where(cropped_image_a != 0)
                        # tmp_targer_value_index = list(copy.copy(tmp_value_index))
                        # image_b[tmp_targer_value_index[0] + b_y, tmp_value_index[1] + b_x, tmp_value_index[2]] = 0
                        image_b[b_y:b_y + height, b_x:b_x + width] = cropped_image_a

                        res = convert_to_yolo_format(class_id, b_x, b_y, b_x + width, b_y + height, image_b_width,
                                                         image_b_height)
                        # print("--==", img_b_txt)
                        f.write(res)
        except:
            print("error")

    cv2.imwrite(img_b_path_new, image_b)
