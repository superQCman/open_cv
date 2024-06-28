import cv2
import mediapipe as mp
import cv2 as cv
from pynput.mouse import Controller, Button
import pyautogui
import math

figure_xyz = [[[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]]

cooldown_frames = 20  # 冷却时间，单位为帧数
last_click_frame = -cooldown_frames  # 上一次检测到点击的帧索引

def update_figure_xyz(new_figure):
    if new_figure:
        figure_xyz.insert(0, new_figure)
        if len(figure_xyz) > 90:
            figure_xyz.pop()
    return

def give_back_figurexyz(cap, orbbec_cap):
    is_figure = True
    #读取普通数据
    success, img = cap.read()
    if not success:
        print("Fail to grab data from camera!")
    # 倒置图像
    img = cv2.flip(img, 1)
    figurexy = give_back_figurexy(img)

    #读取深度数据
    if orbbec_cap.grab():
        ret_depth, depth_map = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_DEPTH_MAP)
    else:
        print("Fail to grab data from camera!")

    # 识别成功判定
    if figurexy and ret_depth:
        for i in range(0, 6):
            if figurexy[0][0] != -1:
                cv2.circle(img, (figurexy[i][0], figurexy[i][1]), 5, (0, 0, 255), -1)
        # 翻转深度图像
        depth_map = cv2.flip(depth_map, 1)
        # 将深度数据转换为伪彩色图像
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_JET)

        # 获取掌根深度信息并在图像上绘制颜色
        for i in range(0, 6):
            if figurexy[i][0] < 640 and figurexy[i][1] < 480 and 0 < figurexy[i][1] and 0 < figurexy[i][0]:
                depth_num = depth_map[figurexy[i][1], figurexy[i][0]]
                # 插入深度数据
                figurexy[i].append(depth_num)
                #临时插值
                if depth_num < 100 or depth_num > 1000:
                    print(i, depth_num)
                    continue
                #画图
                color_value = depth_colormap[figurexy[i][1], figurexy[i][0]]
                cv2.circle(img, (figurexy[i][0], figurexy[i][1]), 10, color_value.tolist(), cv2.FILLED)
            else:
                # 取上一帧 优化处
                figurexy[i].append(figure_xyz[0][i][2])
    else:
        is_figure = False

    # 不管成不成功判定都画个图和等待
    cv2.imshow("HandsImage", img)  # 显示图像
    cv2.waitKey(1)
    return figurexy

def main():
    current_frame = 0
    mouse = Controller()
    screen_width, screen_height = pyautogui.size()

    # 打开RGB相机
    cap = cv2.VideoCapture(3)
    # 打开深度相机
    orbbec_cap = cv2.VideoCapture(0, cv2.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print("Fail to open obsensor capture!")
        exit(0)

    while True:
        new_figure = give_back_figurexyz(cap, orbbec_cap)
        update_figure_xyz(new_figure)

        move_mouse()
        is_click(current_frame)
        if_shoushi1()
        if_shoushi2()
        #detect_click_gesture()
        current_frame += 1  # 更新当前帧索引
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_depth_map(orbbec_cap):
    """
    从深度相机读取RGB和深度图数据，并返回深度图信息。
    """
    if orbbec_cap.grab():
        ret_depth, depth_map = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_DEPTH_MAP)
        return ret_depth, depth_map
    else:
        print("Fail to grab data from camera!")
        return False, None

def give_back_figurexy(img):
    """
    从图像中检测手部关键点，并返回特定关键点的坐标列表。
    """
    figurexy = []
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                p = [-1, -1]
                if id == 8:
                    p[0] = cx
                    p[1] = cy
                    figurexy.append(p)
                if id == 0:
                    p[0] = cx
                    p[1] = cy
                    figurexy.append(p)
                if id == 4:
                    p[0] = cx
                    p[1] = cy
                    figurexy.append(p)
                if id == 12:
                    p[0] = cx
                    p[1] = cy
                    figurexy.append(p)
                if id == 16:
                    p[0] = cx
                    p[1] = cy
                    figurexy.append(p)
                if id == 20:
                    p[0] = cx
                    p[1] = cy
                    figurexy.append(p)
    return figurexy

def is_click(current_frame):
    """
    使用食指尖的深度数据变化来检测点击手势。
    检测食指在连续多帧中的平均深度变化，以判定是否为点击操作。
    """
    global last_click_frame
    num_frames_to_consider = 10  # 考虑最近10帧的数据
    threshold = 50  # 深度变化的阈值

    if current_frame - last_click_frame < cooldown_frames:
        return False  # 在冷却时间内不进行检测

    if len(figure_xyz) < num_frames_to_consider:
        return False  # 如果没有足够的历史帧来比较，则返回 False

    # 计算最近几帧中食指深度变化的平均值
    total_depth_change = 0
    for i in range(1, num_frames_to_consider):
        current_depth = figure_xyz[i - 1][1][2]  # 当前帧食指的深度
        previous_depth = figure_xyz[i][1][2]  # 上一帧食指的深度
        # 处理深度数据溢出问题
        depth_change = previous_depth - current_depth
        if abs(depth_change) > 32767:  # 假设深度数据为16位无符号整数
            depth_change = -depth_change
        total_depth_change += depth_change

    average_depth_change = total_depth_change / (num_frames_to_consider - 1)

    # 检测平均深度变化是否超过阈值
    if average_depth_change > threshold:
        print("---------------------------------------------------------------------------------------------Click detected based on average depth change over multiple frames.")
        mouse = Controller()
        mouse.click(Button.left, 1)
        last_click_frame = current_frame  # 更新最后一次检测到点击的帧索引
        return True

    return False


def move_mouse():
    return

def if_shoushi1():
    return

def if_shoushi2():
    return


if __name__ == '__main__':
    main()
