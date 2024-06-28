import cv2
import numpy as np
from wechatqrcode import WeChatQRCode

def main():
    # 模型文件路径
    detect_prototxt_path = "detect.prototxt"
    detect_caffemodel_path ="detect.caffemodel"
    sr_prototxt_path = "sr.prototxt"
    sr_caffemodel_path = "sr.caffemodel"

    # 加载WeChatQRCode模型
    model = WeChatQRCode(detect_prototxt_path, 
                         detect_caffemodel_path, 
                         sr_prototxt_path, 
                         sr_caffemodel_path)

    # 打开摄像头视频流
    cap = cv2.VideoCapture(0)  # 如果有多个摄像头，可以更改索引

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测二维码
        texts, points = model.infer(frame)

        # 如果检测到二维码
        if points is not None and len(points) > 0:
            for point in points:
                # 将检测到的二维码点连接成框
                point = np.int32(point).reshape((-1, 1, 2))
                cv2.polylines(frame, [point], isClosed=True, color=(0, 255, 0), thickness=2)

        # 显示结果
        cv2.imshow('Detected QR Codes', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
