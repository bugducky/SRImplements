import cv2
import glob
import os

# 分割后的数据目录
patch_hr = "datasets/bsds200/hr"
patch_lr = "datasets/bsds200/lr"
os.makedirs(patch_hr, exist_ok=True)
os.makedirs(patch_lr, exist_ok=True)
# 原始数据根目录
hr_img_paths = glob.glob("D:\\code\\\dataset\\\set5\\SR_training_datasets\\BSDS200\\*")
f = open("datasets/bsds200/meta_info.txt", "w")
# 缩放因子
scaling_factor = 4
lr_size = 48
gt_size = lr_size * scaling_factor

for idx, path in enumerate(hr_img_paths):
    print(idx, path)
    img = cv2.imread(path)
    w, h, c = img.shape
    counter = 0
    row_num = h // gt_size
    col_num = w // gt_size

    for i in range(col_num):
        for j in range(row_num):
            start_x, start_y = i * gt_size, j * gt_size
            t_hr_img = img[start_x:start_x + gt_size, start_y:start_y + gt_size, :]
            t_lr_img = cv2.resize(t_hr_img, (lr_size, lr_size), cv2.INTER_CUBIC)
            t_hr_fn = f"{patch_hr}/{idx}_{counter}.png"
            t_lr_fn = f"{patch_lr}/{idx}_{counter}.png"
            cv2.imwrite(t_hr_fn, t_hr_img)
            cv2.imwrite(t_lr_fn, t_lr_img)
            f.write(f"hr/{idx}_{counter}.png,lr/{idx}_{counter}.png\n")
            counter += 1

f.close()
