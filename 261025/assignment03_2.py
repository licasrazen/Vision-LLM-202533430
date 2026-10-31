import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.data
from random import randint, gauss

# IoU 구현

# TODO: 1) BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의
# TODO: 2) IoU 계산 함수 정의

def convert_bbox_coord(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def calculate_iou(bbox_a, bbox_b):
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b

    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    w = max(0, intersection_x2 - intersection_x1)
    h = max(0, intersection_y2 - intersection_y1)
    intersection_area = w * h

    area_a = max(0, xa_2 - xa_1) * max(0, ya_2 - ya_1)
    area_b = max(0, xb_2 - xb_1) * max(0, yb_2 - yb_1)
    union_area = area_a + area_b - intersection_area

    # TODO: ZeroDivisionError 주의
    if union_area == 0:
        return 0.0
    return intersection_area / union_area


apples1 = cv2.imread("src/images/two_apples1.png")
apples2 = cv2.imread("src/images/two_apples2.png")
apples3 = cv2.imread("src/images/two_apples3.png")

if any(img is None for img in (apples1, apples2, apples3)):
    raise FileNotFoundError("src/images/two_apples1.png~two_apples3.png 파일을 확인하세요.")

apples = [apples1, apples2, apples3]
apple_rgbs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in apples]
ious = []

for img, img_rgb in zip(apples, apple_rgbs):
    cntr_red, cntr_grn = get_contour(img)
    if cntr_red is None or cntr_grn is None:
        ious.append(0.0)
        continue

    bbox1 = cv2.boundingRect(cntr_red)
    bbox2 = cv2.boundingRect(cntr_grn)

    x_r, y_r, w_r, h_r = bbox1
    x_g, y_g, w_g, h_g = bbox2

    draw_bbox(img_rgb, (x_r, y_r), (x_r + w_r, y_r + h_r),
              (255, 0, 0), txt="Red Apple")
    draw_bbox(img_rgb, (x_g, y_g), (x_g + w_g, y_g + h_g),
              (0, 255, 0), txt="Green Apple")

    ious.append(calculate_iou(
        convert_bbox_coord(bbox1), convert_bbox_coord(bbox2)
    ))

plt.figure(figsize=(16, 6))
for i, (img_rgb, iou) in enumerate(zip(apple_rgbs, ious), start=1):
    plt.subplot(1, 3, i)
    plt.imshow(img_rgb)
    plt.title(f"IoU: {iou:.4f}")
    plt.axis("off")
plt.show()


def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]
    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))
    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)
    return x1, y1, x1 + w, y1 + h


def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))


hubble = skimage.data.hubble_deep_field()
N = 20
boxes = []
scores = []

for i in range(N):
    boxes.append(random_bbox(hubble))
    scores.append(random_conf())

for i, box in enumerate(boxes):
    draw_bbox(
        hubble, (box[0], box[1]), (box[2], box[3]),
        (randint(0, 255), randint(0, 255), randint(0, 255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10, 8))
plt.imshow(hubble)
plt.axis("off")
plt.show()


# TODO: Confidence Thresholding 함수 정의
# TODO: Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거

def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """
    Confidence Thresholding 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    Returns
    -------
    pruned_boxes : list
        Confidence Score가 임계값 이상인 Bounding Box 목록

    pruned_scores : list
        Confidence Score가 임계값 이상인 Bounding Box의 Confidence Score 목록
    """
    pruned_boxes = []
    pruned_scores = []

    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)

    return pruned_boxes, pruned_scores


# TODO: NMS 함수 정의
# TODO: np.sort() 함수를 사용하면 1차원 배열을 정렬
# TODO: np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환

# NMS 구현
def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score 목록
    """
    nms_boxes = []
    nms_scores = []

    pruned_boxes, pruned_scores = prune_low_conf(boxes, scores)

    if len(pruned_boxes) == 0:
        return nms_boxes, nms_scores

    order = np.argsort(pruned_scores)[::-1]

    while len(order) > 0:
        best_index = order[0]
        best_box = pruned_boxes[best_index]
        best_score = pruned_scores[best_index]

        nms_boxes.append(best_box)
        nms_scores.append(best_score)

        remaining_indices = order[1:]
        kept_indices = []

        for index in remaining_indices:
            iou = calculate_iou(best_box, pruned_boxes[index])
            if iou <= iou_threshold:
                kept_indices.append(index)

        order = np.array(kept_indices, dtype=np.int64)

    return nms_boxes, nms_scores


new_boxes, new_scores = prune_low_conf(boxes, scores)
new_boxes, new_scores = apply_nms(boxes, scores, iou_threshold=0.2)

print(len(boxes), len(scores))
print(len(new_boxes), len(new_scores))

hubble_result = skimage.data.hubble_deep_field()
for i, box in enumerate(new_boxes):
    draw_bbox(
        hubble_result, (box[0], box[1]), (box[2], box[3]),
        (randint(0, 255), randint(0, 255), randint(0, 255)),
        txt=f"{new_scores[i]:.2f}"
    )

plt.figure(figsize=(10, 8))
plt.imshow(hubble_result)
plt.axis("off")
plt.show()
