import cv2
import mediapipe as mp
import cv2 as cv


last_figurexy = 200

def main():
    
    #一些用到的参数定义
    ret_depth = True
    depth_map = None
    #打开RGB相机
    cap = cv2.VideoCapture(3)
    
    # 打开深度相机
    orbbec_cap = cv2.VideoCapture(0, cv2.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print("Fail to open obsensor capture!")
        exit(0)
    
    while 1 :
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
            for i in range(0,6):
                if figurexy[0][0] != -1:
                    cv2.circle(img,(figurexy[i][0],figurexy[i][1]),5,(0,0,255),-1)

            # 翻转深度图像
            depth_map = cv2.flip(depth_map, 1)
            # 将深度数据转换为伪彩色图像
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_JET)
            
            # 获取掌根深度信息并在图像上绘制颜色
            for i in range(0,6):
                if figurexy[i][0] < 640 and figurexy[i][1] < 480 and 0 < figurexy[i][1] and 0 < figurexy[i][0]:
                    depth_num = depth_map[figurexy[i][1], figurexy[i][0]]
                    if depth_num < 100 or depth_num > 1000:
                        print(i,depth_num)
                        continue
                    color_value = depth_colormap[figurexy[i][1], figurexy[i][0]]
                    cv2.circle(img, (figurexy[i][0], figurexy[i][1]), 10, color_value.tolist(), cv2.FILLED)

        else:
            is_figure = False
            
            
        '''if ret_depth:
            # 显示深度图
            if ret_depth:
                
               # 显示深度伪彩色图像
                cv.imshow("Depth Image", depth_colormap)'''
                    
        #不管成不成功判定都画个图和等待
        cv2.imshow("HandsImage", img)  # 显示图像
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()



def get_depth_map(orbbec_cap):
    """
    从深度相机读取RGB和深度图数据，并返回深度图信息。

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



def give_back_figurexy(img):
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
    
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    
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
                p = [-1,-1]
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

if __name__ == '__main__':
    print(1)
    main()