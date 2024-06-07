import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from picamera2 import Picamera2, Preview

# GPIO 핀 설정
LEFT_MOTOR_PIN1 = 19
LEFT_MOTOR_PIN2 = 13
RIGHT_MOTOR_PIN1 = 6
RIGHT_MOTOR_PIN2 = 5

GPIO.setmode(GPIO.BCM)
GPIO.setup(LEFT_MOTOR_PIN1, GPIO.OUT)
GPIO.setup(LEFT_MOTOR_PIN2, GPIO.OUT)
GPIO.setup(RIGHT_MOTOR_PIN1, GPIO.OUT)
GPIO.setup(RIGHT_MOTOR_PIN2, GPIO.OUT)

# PWM 설정
left_motor_pwm = GPIO.PWM(LEFT_MOTOR_PIN1, 10)
right_motor_pwm = GPIO.PWM(RIGHT_MOTOR_PIN1, 10)
left_motor_pwm.start(0)
right_motor_pwm.start(0)

# 모델 로드
model = load_model('traffic_light_model.h5')

# 클래스 레이블 설정
class_labels = ['red', 'yellow', 'green']

def set_motor_speed(left_speed, right_speed):
    # 모터 속도 설정
    left_motor_pwm.ChangeDutyCycle(left_speed)
    right_motor_pwm.ChangeDutyCycle(right_speed)
    
    # 모터 방향 설정
    GPIO.output(LEFT_MOTOR_PIN2, GPIO.LOW if left_speed > 0 else GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_PIN2, GPIO.LOW if right_speed > 0 else GPIO.HIGH)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    if lines is None:
        return img
    img = np.copy(img)
    line_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 0), 10)  # 검정색으로 라인 그리기
    
    img = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    return img

def process_traffic_light(image):
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    reshaped_image = np.reshape(normalized_image, (1, 224, 224, 3))
    
    prediction = model.predict(reshaped_image)
    class_index = np.argmax(prediction)
    return class_labels[class_index]

def process_image(image):
    height, width = image.shape[:2]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 180,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    image_with_lines = draw_lines(image, lines)
    
    if lines is not None:
        left_lane, right_lane = [], []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if slope < 0:
                    left_lane.append(line)
                else:
                    right_lane.append(line)

        if left_lane and right_lane:
            left_line = np.mean(left_lane, axis=0).astype(int)
            right_line = np.mean(right_lane, axis=0).astype(int)
            
            left_x1, left_y1, left_x2, left_y2 = left_line[0]
            right_x1, right_y1, right_x2, right_y2 = right_line[0]

            mid_x = (left_x2 + right_x2) // 2

            # 중앙에서 벗어난 정도에 따라 모터 속도 조절
            if mid_x < width // 2 - 50:
                # 왼쪽으로 치우쳤을 때 오른쪽으로 회전
                set_motor_speed(70, 100)  # 왼쪽 모터는 70% 출력, 오른쪽 모터는 100% 출력
            elif mid_x > width // 2 + 50:
                # 오른쪽으로 치우쳤을 때 왼쪽으로 회전
                set_motor_speed(100, 70)  # 왼쪽 모터는 100% 출력, 오른쪽 모터는 70% 출력
            else:
                # 중앙에 있을 때 직진
                set_motor_speed(100, 100)  # 양쪽 모터 모두 100% 출력
        else:
            # 차선을 인식하지 못했을 때 멈춤
            set_motor_speed(0, 0)  # 양쪽 모터 모두 정지
    else:
        # 차선을 인식하지 못했을 때 멈춤
        set_motor_speed(0, 0)  # 양쪽 모터 모두 정지

    return image_with_lines

def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
    picam2.start()

    while True:
        image = picam2.capture_array()
        image = cv2.flip(image, -1)  # 카메라 화면을 180도 회전시킵니다.
        # 신호등 인식
        traffic_light_color = process_traffic_light(image)
        print(f'Traffic Light: {traffic_light_color}')
        
        # 신호에 따라 모터 속도 조절
        if traffic_light_color == 'red':
            # 빨간 신호: 모터 정지
            set_motor_speed(0, 0)
        elif traffic_light_color == 'yellow':
            # 노란 신호: 모터 속도 10%
            set_motor_speed(10, 10)
        elif traffic_light_color == 'green':
            # 초록 신호: 모터 속도 50%
            set_motor_speed(50, 50)

        # 차선 인식
        processed_frame = process_image(image)

        # 텍스트 정보 표시
        cv2.putText(processed_frame, f'Traffic Light: {traffic_light_color}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(processed_frame, 'Lane Detection', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 화면에 표시
        cv2.imshow('Camera Feed', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    GPIO.cleanup()

if __name__ == '__main__':
    main()
