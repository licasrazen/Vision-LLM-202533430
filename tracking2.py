import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import skimage.data

green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720,framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, ball = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # LAB 색공간으로 변환
    ball_lab = cv2.cvtColor(ball, cv2.COLOR_BGR2LAB)
    
    # 각 채널 분리
    l, a, b = cv2.split(ball_lab)
    
    # CLAHE 생성 (clipLimit은 3.0, tileGridSize는 8x8로 설정)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    # L 채널에 CLAHE 적용 후 강화된 이미지 생성
    l_clahe_enhanced = clahe.apply(l)
    ball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))
    
    # 강화된 이미지로 binary mask 생성
    mask_enhanced = cv2.inRange(ball_enhanced_lab, green_lower, green_upper)
    
    ball_enhanced_rgb = cv2.cvtColor(ball_enhanced_lab, cv2.COLOR_LAB2RGB)
    
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    ball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)
    
    # Contour를 그릴 원본 이미지 copy
    img_final = ball.copy()
    
    # 컨투어 추출 (두 번째 반환값인 hierarchy는 바깥쪽 테두리만 검출하는 현재 방식상 무의미하기에 저장하지 않음)
    contours, _ = cv2.findContours(
        ball_mask_opened,        # binary mask
        cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
        cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
    )
    
    if len(contours) > 0:
        # 가장 큰 contour
        largest_contour = max(contours, key=cv2.contourArea)
    
        # 가장 큰 contour만 채운 binary mask 생성
        final_mask = np.zeros_like(ball_mask_opened)
        cv2.drawContours(
            final_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )
    
        # 가장 큰 contour만 채운 binary mask를 반투명 초록색으로 overlay
        mask_color = np.zeros_like(img_final)
        mask_color[:, :, 1] = final_mask
        alpha = 0.35
        img_final = cv2.addWeighted(img_final, 1.0, mask_color, alpha, 0)
    
        # Contour 그리기 (초록색 외곽선)
        cv2.drawContours(img_final, [largest_contour], -1, (0,255,0), 2)
    
    
    
    final_ball_mask = final_mask
    
    # Contour 및 무게중심 기준 원을 그릴 원본 이미지 copy
    final_img_enhanced = ball_enhanced_rgb.copy()
    
    if len(contours) > 0:
        # 가장 큰 contour
        largest_contour = max(contours, key=cv2.contourArea)
    
        # 가장 큰 contour만 채운 binary mask 생성
        largest_mask = np.zeros_like(final_ball_mask)
        cv2.drawContours(
            largest_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )
    
        # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
        mask_color = np.zeros_like(final_img_enhanced)
        mask_color[:, :, 0] = largest_mask
        alpha = 0.15
        final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)
    
        # Contour 그리기 (빨간색 외곽선)
        cv2.drawContours(final_img_enhanced, [largest_contour], -1, (255,0,0), 2)
    
        # 최소 외접원 중심 및 반지름
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center_circle = (int(x), int(y))
        radius = int(radius)
    
        # Moments를 통해 무게중심 (Centroid) 산출
        M = cv2.moments(largest_contour)
    
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = center_circle
    
        centroid = (cX, cY)
    
        # 노란색 무게중심 기반 원
        cv2.circle(final_img_enhanced, centroid, radius, (255,255,0), 3)
        cv2.circle(final_img_enhanced, centroid, 5, (255,255,0), -1)
    
    
    
    final_img_enhanced = cv2.cvtColor(final_img_enhanced, cv2.COLOR_RGB2BGR)
    cv2.imshow("Live Camera Video", final_img_enhanced)
    
    
cap.release()
cv2.destroyAllWindows()