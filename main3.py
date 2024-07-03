import time
import cv2
import mediapipe as mp
import cv2 as cv
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key
import pyautogui
import numpy as np
from collections import deque

#DEBUG模式开启所有函数进行输出，关闭所有函数进行操作    
DEBUG = False

NUM_SAMPLES = 10

#点击用参数
cooldown_frames = 10  # 冷却时间，单位为帧数
# initial_frames = 20  # 初始缓冲期帧数
last_click_frame = -cooldown_frames  # 上一次检测到点击的帧索引
last_click_frame_shoushi_3 = -cooldown_frames  # 上一次检测到点击的帧索引
last_click_frame_shoushi_2 = -cooldown_frames  # 上一次检测到点击的帧索引

# 移动用参数
debounce_threshold = 3  # 抖动消除阈值
history_length = 5  # 均值滤波历史帧数
direction_consistency_threshold = 3  # 方向一致性阈值
# 历史坐标队列
x_history = deque(maxlen=history_length)
y_history = deque(maxlen=history_length)
previous_direction = None
consistent_direction_count = 0

figure_xyz = [[[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]]

def update_figure_xyz(new_figure):
    if new_figure:
        figure_xyz.insert(0,new_figure)
        if len(figure_xyz) > 90:
            figure_xyz.pop()
    return
    
def give_back_figurexyz(cap,orbbec_cap,hands):
    is_figure = True
    #读取普通数据
    success, img = cap.read()
    if not success:
        print("Fail to grab data from camera!")
    # 倒置图像
    img = cv2.flip(img, 1)
    figurexy, middle_finger_length = give_back_figurexy(img,hands)
    
    #读取深度数据
    if orbbec_cap.grab():
        ret_depth, depth_map = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_DEPTH_MAP)
    else:
        print("Fail to grab data from camera!")
    
    # 识别成功判定
    if figurexy and ret_depth and middle_finger_length is not None:
        for i in range(0,6):
            if figurexy[0][0] != -1:
                cv2.circle(img,(figurexy[i][0],figurexy[i][1]),5,(0,0,255),-1)
        # 翻转深度图像
        depth_map = cv2.flip(depth_map, 1)
        # 将深度数据转换为伪彩色图像
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_JET)
        
        # 获取掌根深度信息并在图像上绘制颜色
        for i in range(6):
            x, y = figurexy[i][0], figurexy[i][1]
            if 0 < x < 640 and 0 < y < 480:
                depths = []
                
                for _ in range(NUM_SAMPLES):
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(0, middle_finger_length)
                    dx = int(radius * np.cos(angle))
                    dy = int(radius * np.sin(angle))
                    if 0 <= x + dx < 640 and 0 <= y + dy < 480:
                        depth = depth_map[y + dy, x + dx]
                        if 100 <= depth <= 1000:
                            depths.append(depth)
                
                if depths:
                    avg_depth = sum(depths) / len(depths)
                else:
                    avg_depth = figure_xyz[0][i][2]
                figurexy[i].append(int(avg_depth))
                
                color_value = depth_colormap[y, x]
                cv2.circle(img, (x, y), 10, color_value.tolist(), cv2.FILLED)
            else:
                figurexy[i].append(figure_xyz[0][i][2])
        '''for i in range(6):
            x, y = figurexy[i][0], figurexy[i][1]
            if 0 < x < 640 and 0 < y < 480:
                depths = []
                for dx in range(-int(middle_finger_length), int(middle_finger_length) + 1):
                    for dy in range(-int(middle_finger_length), int(middle_finger_length) + 1):
                        if 0 <= x + dx < 640 and 0 <= y + dy < 480 and dx * dx + dy * dy <= middle_finger_length * middle_finger_length:
                            depth = depth_map[y + dy, x + dx]
                            if 100 <= depth <= 1000:
                                depths.append(depth)
                if depths:
                    avg_depth = sum(depths) / len(depths)
                else:
                    avg_depth = figure_xyz[0][i][2]
                figurexy[i].append(int(avg_depth))
                
                # 画图
                color_value = depth_colormap[y, x]
                cv2.circle(img, (x, y), 10, color_value.tolist(), cv2.FILLED)
            else:
                # 取上一帧 优化处
                figurexy[i].append(figure_xyz[0][i][2])'''
       
    else:
        is_figure = False
            
        '''if ret_depth:
            # 显示深度图
            if ret_depth:
                
               # 显示深度伪彩色图像
                cv.imshow("Depth Image", depth_colormap)'''
    #不管成不成功判定都画个图和等待
    
    cv2.imshow("HandsImage", img)  # 显示图像
    cv2.waitKey(1)
    return figurexy

def main():
    #当前第几帧
    current_frame = 0
    #初始化鼠标对象与屏幕宽高
    mouse = MouseController()
    keyboard =  KeyboardController()
    screen_width, screen_height = pyautogui.size()
    #打开RGB相机
    cap = cv2.VideoCapture(3)
    # 打开深度相机
    orbbec_cap = cv2.VideoCapture(0, cv2.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print("Fail to open obsensor capture!")
        exit(0)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    
    while 1 :
        new_figure = give_back_figurexyz(cap, orbbec_cap, hands)
        update_figure_xyz(new_figure)
        
        move_mouse(mouse,screen_width, screen_height)
        
        if is_click(current_frame):
            if DEBUG :
                print("click")
            else :
                mouse.click(Button.left, 1)

        if if_shoushi2(current_frame):
            if DEBUG :
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!!!!!!检测到手势2！########################!!!!!!!!!!!!!!!!!!')
            else :
                press_esc(keyboard)
        if if_shoushi3(current_frame):
            if DEBUG :
                print('--------------------------------------------检测到手势3！-----------------------------------------------')
            else :
                press_spade(keyboard)
        current_frame += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    


def get_depth_map(orbbec_cap):
    """
    从深度相机读取RGB和深度图数据,并返回深度图信息。

    参数:
        orbbec_cap: cv.VideoCapture 对象，表示打开的深度相机。

    返回:
        tuple: (ret_bgr, bgr_image, ret_depth, depth_map)
            - ret_bgr: 布尔值，表示是否成功获取RGB图像。
            - bgr_image: RGB图像。
            - ret_depth: 布尔值，表示是否成功获取深度图像。
            - depth_map: 深度图像。
    """
    if orbbec_cap.grab():
        # 解码grab()获取的帧数据
        # 深度数据
        ret_depth, depth_map = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_DEPTH_MAP)
        
        return ret_depth, depth_map
    else:
        print("Fail to grab data from camera!")
        return False, None

def give_back_figurexy(img,hands):
    """
    从图像中检测手部关键点，并返回特定关键点的坐标列表。
    
    Args:
        img (np.ndarray): 输入的BGR图像。
    
    Returns:
        list: 包含多个二元组坐标列表的列表，每个二元组代表一个手部关键点的(x, y)坐标。
              列表中只包含特定关键点（id为8, 0, 4, 12, 16, 20）的坐标
              分别为掌根，拇指，食指,中指，无名指，小拇指
    
    """
    # 定义并引用mediapipe中的hands模块
    
    figurexy = []
    middle_finger_length = None
    
    # 将图像从BGR颜色空间转换为RGB颜色空间
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 处理图像，检测手部
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            landmarks = {}
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks[id] = (cx, cy)
                p = [-1, -1]
                if id in [0, 4, 8, 12, 16, 20]:
                    p[0] = cx
                    p[1] = cy
                    figurexy.append(p)
            if 11 in landmarks and 12 in landmarks:
                middle_finger_length = ((landmarks[9][0] - landmarks[12][0]) ** 2 + (landmarks[9][1] - landmarks[12][1]) ** 2) ** 0.5 / 2 * 1.5
    
    return figurexy, middle_finger_length 

def distance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def euclidean_distance_3d(point1, point2):
    """计算两点之间的三维欧式距离"""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5

def distance(x1,x2,y1,y2):
    return ((x1-x2)**2+(y1-y2)**2)**(1/2)

def is_allowed_click_gesture():
    """判断当前手势是否允许点击
    Args:
        figure_xyz: 手部关键点的坐标列表，格式为[[x, y, z], [x, y, z], ...]

    Returns:
        bool: 如果当前手势是允许点击的手势，返回True
    """
    wrist = figure_xyz[0][0]  # 假设掌根是列表的第一个元素
    index_finger_tip = figure_xyz[0][2]  # 食指尖
    middle_finger_tip = figure_xyz[0][3]  # 中指尖
    ring_finger_tip = figure_xyz[0][4]  # 无名指尖
    pinky_tip = figure_xyz[0][5]  # 小指尖

    index_distance = euclidean_distance_3d(index_finger_tip, wrist)
    other_fingers_distance = (euclidean_distance_3d(middle_finger_tip, wrist) +
                              euclidean_distance_3d(ring_finger_tip, wrist) +
                              euclidean_distance_3d(pinky_tip, wrist)) / 2  # 计算其余三指到掌根的平均距离

    return index_distance > other_fingers_distance  # 判断食指距离是否大于其他指的平均距离

def move_mouse(mouse, screen_width, screen_height):
    global figure_xyz
    global previous_direction, consistent_direction_count
    
    if is_allowed_click_gesture():
        index_finger_tip = figure_xyz[0][2]
        
        frame_width = 640  # 假设视频帧宽度为640
        frame_height = 480  # 假设视频帧高度为480
        
        scale_x = 1.5  # 水平方向缩放比例
        scale_y_top = 1.25  # 上半部分垂直方向缩放比例
        scale_y_bottom = 2.0  # 下半部分垂直方向缩放比例
        
        # 先平移坐标，使得坐标中心对称
        centered_x = index_finger_tip[0] - (frame_width / 2)
        centered_y = index_finger_tip[1] - (frame_height / 2)
        
        # 根据手指在视频帧中的位置动态调整垂直方向的缩放比例
        if centered_y >= 0:
            scaled_y = centered_y * scale_y_bottom
        else:
            scaled_y = centered_y * scale_y_top
        
        # 将平移后的坐标进行缩放
        scaled_x = centered_x * scale_x
        
        # 将缩放后的坐标映射到屏幕坐标，并平移回原始中心
        screen_x = int((scaled_x + (frame_width / 2)) / frame_width * screen_width)
        screen_y = int((scaled_y + (frame_height / 2)) / frame_height * screen_height)
        
        # 处理边界问题
        screen_x = max(0, min(screen_x, screen_width))
        screen_y = max(0, min(screen_y, screen_height))

        # 计算当前帧与上一帧的位移
        if x_history and y_history:
            dx = screen_x - x_history[-1]
            dy = screen_y - y_history[-1]
        else:
            dx = dy = 0

        # 更新历史坐标队列
        x_history.append(screen_x)
        y_history.append(screen_y)

        # 均值滤波处理
        avg_x = int(np.mean(x_history))
        avg_y = int(np.mean(y_history))

        # 方向判断和抖动消除
        direction = (np.sign(dx), np.sign(dy))
        if direction == previous_direction:
            consistent_direction_count += 1
        else:
            consistent_direction_count = 0

        previous_direction = direction

        if consistent_direction_count < direction_consistency_threshold and (abs(dx) < debounce_threshold and abs(dy) < debounce_threshold):
            screen_x, screen_y = avg_x, avg_y
        else:
            screen_x, screen_y = avg_x, avg_y

        mouse.position = (screen_x, screen_y)
        
        if DEBUG:
            print(f"Moving mouse to: {screen_x}, {screen_y}")


def if_shoushi2(current_frame):
    # 增加了触发CD 
    global last_click_frame_shoushi_2
    if current_frame - last_click_frame_shoushi_2 < cooldown_frames:
        return False
    """
    手势检测2
    """
    figure_correct=[[320,326],[253,228],[258,274],[282,274],[304,274],[326,266]]
    check=0
    dist_test_sum=1
    dist_train_sum=1
    for i in range(6):
        for j in range(i,6):
            dist_test_sum+=distance(figure_xyz[0][i][0],figure_xyz[0][j][0],figure_xyz[0][i][1],figure_xyz[0][j][1])
            dist_train_sum+=distance(figure_correct[i][0],figure_correct[j][0],figure_correct[i][1],figure_correct[j][1])
    for i in range(6):
        for j in range(i,6):
            dist_test=distance(figure_xyz[0][i][0],figure_xyz[0][j][0],figure_xyz[0][i][1],figure_xyz[0][j][1])
            dist_train=distance(figure_correct[i][0],figure_correct[j][0],figure_correct[i][1],figure_correct[j][1])
            if (dist_test/dist_test_sum-dist_train/dist_train_sum)**2*1e5 >= 50:
                check=1
    if check==0:
        last_click_frame_shoushi_2 = current_frame
        return True
    return False

def if_shoushi3(current_frame):
    global last_click_frame_shoushi_3
    if current_frame - last_click_frame_shoushi_3 < cooldown_frames:
        return False
    """
    手势检测3,比4的手势
    """
    figure_correct=[[277,359],[309,240],[194,59],[248,59],[318,77],[369,134]]
    check=0
    dist_test_sum=1
    dist_train_sum=1
    for i in range(6):
        for j in range(i,6):
            dist_test_sum+=distance(figure_xyz[0][i][0],figure_xyz[0][j][0],figure_xyz[0][i][1],figure_xyz[0][j][1])
            dist_train_sum+=distance(figure_correct[i][0],figure_correct[j][0],figure_correct[i][1],figure_correct[j][1])
    for i in range(6):
        for j in range(i,6):
            dist_test=distance(figure_xyz[0][i][0],figure_xyz[0][j][0],figure_xyz[0][i][1],figure_xyz[0][j][1])
            dist_train=distance(figure_correct[i][0],figure_correct[j][0],figure_correct[i][1],figure_correct[j][1])
            if (dist_test/dist_test_sum-dist_train/dist_train_sum)**2*1e5 >= 40:
                check=1
    if check==0:
        last_click_frame_shoushi_3 = current_frame
        return True
    return False

def is_click(current_frame):
    global last_click_frame, figure_xyz
    num_frames_to_consider = 6 #考虑的帧数,尽可能是偶数
    depth_threshold = 30  # 深度阈值
    move_threshold = 30  # 食指移动阈值

    if current_frame - last_click_frame < cooldown_frames:
        return False

    if len(figure_xyz) < num_frames_to_consider:
        return False


    if not is_allowed_click_gesture():
        return False  # 如果任何一帧的手势不符合条件，则不允许点击

    total_depth_change_front = 0
    total_depth_change_back = 0
    consistent_negative_change = True
    x_start, y_start = figure_xyz[num_frames_to_consider-1][1][0], figure_xyz[num_frames_to_consider-1][1][1]
    x_end, y_end = figure_xyz[0][1][0], figure_xyz[0][1][1]

    #临时检查深度变化用变量
    depth_list = []

    for i in range(1, int(num_frames_to_consider / 2)):
        current_depth = figure_xyz[i - 1][1][2]
        previous_depth = figure_xyz[i][1][2]
        depth_change = previous_depth - current_depth
        if abs(depth_change) > 32767:  # 处理可能的整数溢出
            depth_change = -depth_change

        total_depth_change_back += depth_change
        depth_list.append(depth_change)
        # if depth_change >= 0:  # 确保所有的深度变化是向前的
            # consistent_negative_change = False

    for i in range(int(num_frames_to_consider / 2), int(num_frames_to_consider)):
        current_depth = figure_xyz[i - 1][1][2]
        previous_depth = figure_xyz[i][1][2]
        depth_change = current_depth - previous_depth
        if abs(depth_change) > 32767:  # 处理可能的整数溢出
            depth_change = -depth_change

        total_depth_change_front += depth_change
        depth_list.append(depth_change)
        # if depth_change >= 0:  # 确保所有的深度变化是向前的
            # consistent_negative_change = False

    # 计算初始和结束帧之间的XY轴距离
    move_distance = ((x_end - x_start)**2 + (y_end - y_start)**2)**0.5
    #print({average_depth_change}, {move_distance})
    #print(f"Frame: {current_frame}, Previous depth: {previous_depth}, Current depth: {current_depth}, X movement: {x_end - x_start}, Y movement: {y_end - y_start}")

    #print(consistent_negative_change," ",average_depth_change," ",move_distance)
    if consistent_negative_change and total_depth_change_front < -depth_threshold and total_depth_change_back < -depth_threshold and move_distance < move_threshold:
        if DEBUG :
            print("Click detected based on average depth change over multiple frames and limited XY movement." ,{move_distance})
            print(depth_list)
            # mouse = Controller()
            # mouse.click(Button.left, 1)
        last_click_frame = current_frame

        return True

    return False


def press_esc(keyboard):
    keyboard.press(Key.esc)
    keyboard.release(Key.esc)

def press_spade(keyboard):
    keyboard.press('1')
    keyboard.release('1')

if __name__ == '__main__':
    main()
    
    