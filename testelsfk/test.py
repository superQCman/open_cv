import cv2
import mediapipe as mp
import time

# 定义并引用mediapipe中的hands模块
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 初始化摄像头
cap = cv2.VideoCapture(3)

# 帧率时间计算
pTime = 0
cTime = 0

# 初始化数据结构


while True:
    success, img = cap.read()
    if not success:
        print("Fail to grab data from camera!")
        break
    
    # 倒置图像
    img = cv2.flip(img, 1)
    
    # 将图像从BGR颜色空间转换为RGB颜色空间
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 处理图像，检测手部
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # 记录特定关键点的坐标
                if id == 8:
                    print(cx)
                    print(cy)
                if id == 12:
                    print(cx)
                    print(cy)
                if id == 16:
                    print(cx)
                    print(cy)
                if id == 20:
                    print(cx)
                    print(cy)
                if id == 3:
                    print(cx)
                    print(cy)
                if id == 2:
                    print(cx)
                    print(cy)
                if id == 17:
                    print(cx)
                    print(cy)
                
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            
            # 绘制手部特征点
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # FPS显示
    
    cv2.imshow("HandsImage", img)  # 显示图像
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
