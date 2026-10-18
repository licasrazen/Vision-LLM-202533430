import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import skimage.data



def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma


# Kalman filter 변수 정의
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])
mu_t = np.array([[0, 0], [0, 0]])
sigma_t = np.array([[0, 0], [0, 0]])

# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출

# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

# TODO: 각 프레임 이미지를 전처리

# TODO: Color Segmentation으로 Binary Mask 생성

# TODO: Morphological Operation을 적용하여 객체 영역 추출

# TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay


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
    ret, frame = cap.read()
    print(ret)

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # LAB 색공간으로 변환
    ball_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
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


    contour_lst, _ = cv2.findContours(ball_mask_opened,        # binary mask
            cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
            cv2.CHAIN_APPROX_SIMPLE)

    if len(contour_lst) > 0:
            # 가장 큰 contour 선택
            contour = max(contour_lst, key=cv2.contourArea)
            # 최소 외접원 반지름
            radius = cv2.minEnclosingCircle(contour)
            # 무게중심
            center = cv2.moments(contour)
    
            # 검출된 객체에 파란 원 overlay
            cv2.circle(frame, center, int(radius), (0,0,255), 3)
            cv2.circle(frame, center, 5, (0,0,255), -1)
    
    
            # 객체 최초 검출
            if not found:
                found = True
    
        # 최초 검출 이후 Kalman Filter 적용
        # 측정값 사용 (visible) -> Prediction & Update
    if found and (len(contour_lst) > 0):
            mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([list(center)]))
            x_bel, y_bel = mu_t[0][0], mu_t[0][1]
    
        # 측정값 미사용 (occluded) -> Prediction
    elif found and (len(contour_lst) <= 0):
            mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
            x_bel, y_bel = mu_t[0][0], mu_t[0][1]
    
        # 예측한 객체 위치에 노란 원 overlay
    cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (255,255,0), 3)
    cv2.circle(frame, (int(x_bel), int(y_bel)), 5, (255,255,0), -1)

    
cap.release()
cv2.destroyAllWindows()
    