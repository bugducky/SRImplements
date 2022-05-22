import cv2
import glob
import os

# 分割后的数据目录
patch_hr = "D:\\code\\deeplearning\\dataset\\set14_val\\hr"
patch_lr = "D:\\code\\deeplearning\\dataset\\set14_val/lr"
os.makedirs(patch_hr, exist_ok=True)
os.makedirs(patch_lr, exist_ok=True)
# 原始数据根目录
hr_img_paths = glob.glob("D:\\code\\deeplearning\\dataset\\set5\\SR_testing_datasets\\Set14/*")
f = open("D:\\code\\deeplearning\\dataset\\set14_val/meta_info.txt", "w")
# 缩放因子
scaling_factor = 4
lr_size = 48
gt_size = lr_size * scaling_factor

for idx, path in enumerate(hr_img_paths):
    filename = path.split("\\")[-1]

    print(idx, path)
    img = cv2.imread(path)
    w, h, c = img.shape
    img = img[0:w - (w % scaling_factor), 0:h - (h % scaling_factor), :]
    w, h, c = img.shape
    t_hr_fn = f"{patch_hr}/{filename}"
    t_lr_fn = f"{patch_lr}/{filename}"
    img_lr = cv2.resize(img, (h // scaling_factor, w // scaling_factor))
    cv2.imwrite(t_hr_fn, img)
    cv2.imwrite(t_lr_fn, img_lr)

    f.write(f"hr/{filename},lr/{filename}\n")
