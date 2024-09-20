from romatch import tiny_roma_v1_outdoor
import torch
import cv2
from PIL import Image
import numpy as np

imA_path = "assets/00837.jpg"
imB_path = "assets/Out_00837_resized.jpg"
imgA = cv2.imread(imA_path)
imgB = cv2.imread(imB_path)
W_A, H_A = Image.open(imA_path).size
W_B, H_B = Image.open(imB_path).size
device = "cuda" if torch.cuda.is_available() else "cpu"


def draw_matches(img1, keypoints1, img2, keypoints2):
    # 将两张图像水平拼接在一起
    img1_height, img1_width = img1.shape[:2]
    img2_height, img2_width = img2.shape[:2]

    # 创建一个空的拼接图像，宽度为两张图像宽度之和，高度为两张图像的最大高度
    combined_image = np.zeros(
        (max(img1_height, img2_height), img1_width + img2_width, 3), dtype=np.uint8
    )

    # 将图像1和图像2放到拼接图上
    combined_image[:img1_height, :img1_width] = img1
    combined_image[:img2_height, img1_width : img1_width + img2_width] = img2

    # 将 keypoints2 的坐标移位，使得它们相对于拼接图像的位置正确
    shifted_keypoints2 = [(x + img1_width, y) for (x, y) in keypoints2]

    # 在图像上绘制匹配的关键点
    for (x1, y1), (x2, y2) in zip(keypoints1, shifted_keypoints2):
        cv2.circle(
            combined_image, (int(x1), int(y1)), 2, (0, 255, 0), -1
        )  # 图像1中的点
        cv2.circle(
            combined_image, (int(x2), int(y2)), 2, (0, 0, 255), -1
        )  # 图像2中的点
        cv2.line(combined_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    return combined_image


roma_model = tiny_roma_v1_outdoor(device=device)
# Match
warp, certainty = roma_model.match(imA_path, imB_path)
# Sample matches for estimation
matches, certainty = roma_model.sample(warp, certainty)
# Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
# Find a fundamental matrix (or anything else of interest)
F, mask = cv2.findFundamentalMat(
    kptsA.cpu().numpy(),
    kptsB.cpu().numpy(),
    ransacReprojThreshold=0.2,
    method=cv2.USAC_MAGSAC,
    confidence=0.999999,
    maxIters=10000,
)
mask = mask.ravel().astype(bool)
mask_indices = np.where(mask)[0]
good_matches = [matches[i] for i in mask_indices]
import random

sample_idx = np.random.choice(kptsA.cpu().numpy()[mask].shape[0], 100)
match_img = draw_matches(
    imgA,
    kptsA.cpu().numpy()[mask][sample_idx],
    imgB,
    kptsB.cpu().numpy()[mask][sample_idx],
)
cv2.imwrite("assets/matches.jpg", match_img)
print("success")
