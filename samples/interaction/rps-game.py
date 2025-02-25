#!/usr/bin/env python3
# encoding:utf-8

import sys
import argparse

import numpy as np
import cv2 as cv

# palm and hand pose detection
from mp_handpose import MPHandPose
from mp_palmdet import MPPalmDet

# robot hand
from hiwonderhand import HiwonderHand


parser = argparse.ArgumentParser(description='Hand Pose Estimation from MediaPipe')
parser.add_argument('--input', '-i', type=int, default='0', help='The index of a camera.')
parser.add_argument('--conf_threshold', type=float, default=0.9, help='Filter out hands of confidence < conf_threshold.')
args = parser.parse_args()

def getFingerBending(fourFingerJoints):
    # return a value between 0 and 1 for finger bending
    # 0: completely bending
    # 1: a straight finger
    dist1 = np.sqrt( np.sum( np.square( fourFingerJoints[0,:]- fourFingerJoints[1,:] )))
    dist2 = np.sqrt( np.sum( np.square( fourFingerJoints[1,:]- fourFingerJoints[2,:] )))
    dist3 = np.sqrt( np.sum( np.square( fourFingerJoints[2,:]- fourFingerJoints[3,:] )))
    dist4 = np.sqrt( np.sum( np.square( fourFingerJoints[0,:]- fourFingerJoints[3,:] )))
    bending = dist4 / (dist1+dist2+dist3)
    bending = (bending - 0.4) / 0.6
    if bending > 1:
        bending = 1
    if bending < 0:
        bending = 0
    return bending

def getFingerBendings(handpose):

    landmarks_world = handpose[67:130].reshape(21, 3)

    bending1 = getFingerBending(landmarks_world[1:5,:])
    bending2 = getFingerBending(landmarks_world[5:9,:])
    bending3 = getFingerBending(landmarks_world[9:13,:])
    bending4 = getFingerBending(landmarks_world[13:17,:])
    bending5 = getFingerBending(landmarks_world[17:21,:])

    bending1 = (bending1 - 0.5) / 0.5 #大拇指特殊处理一下
    if bending1 > 1:
        bending1 = 1
    if bending1 < 0:
        bending1 = 0
    return bending1, bending2, bending3, bending4, bending5

def recognizeHandPose(bending1, bending2, bending3, bending4, bending5):
    rps = 'None'
    if (bending2 > 0.8 and bending3 > 0.8 and bending4 > 0.8 and bending5 > 0.8):
        rps = 'Paper'
    elif (bending2 < 0.5 and bending3 < 0.4 and bending4 < 0.4 and bending5 < 0.4):
        rps = 'Rock'
    elif (bending2 > 0.8 and bending3 > 0.8 and bending4 < 0.55 and bending5 < 0.55):
        rps = 'Scissors'

    return rps

def visualize(image, handpose, print_result=False):
    display_screen = image.copy()
    display_3d = np.zeros((400, 400, 3), np.uint8)
    cv.line(display_3d, (200, 0), (200, 400), (255, 255, 255), 2)
    cv.line(display_3d, (0, 200), (400, 200), (255, 255, 255), 2)
    cv.putText(display_3d, 'Main View', (0, 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv.putText(display_3d, 'Top View', (200, 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv.putText(display_3d, 'Left View', (0, 212), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv.putText(display_3d, 'Right View', (200, 212), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    is_draw = False  # ensure only one hand is drawn

    def draw_lines(image, landmarks, is_draw_point=True, thickness=2):
        cv.line(image, landmarks[0], landmarks[1], (255, 255, 255), thickness)
        cv.line(image, landmarks[1], landmarks[2], (255, 255, 255), thickness)
        cv.line(image, landmarks[2], landmarks[3], (255, 255, 255), thickness)
        cv.line(image, landmarks[3], landmarks[4], (255, 255, 255), thickness)

        cv.line(image, landmarks[0], landmarks[5], (255, 255, 255), thickness)
        cv.line(image, landmarks[5], landmarks[6], (255, 255, 255), thickness)
        cv.line(image, landmarks[6], landmarks[7], (255, 255, 255), thickness)
        cv.line(image, landmarks[7], landmarks[8], (255, 255, 255), thickness)

        cv.line(image, landmarks[0], landmarks[9], (255, 255, 255), thickness)
        cv.line(image, landmarks[9], landmarks[10], (255, 255, 255), thickness)
        cv.line(image, landmarks[10], landmarks[11], (255, 255, 255), thickness)
        cv.line(image, landmarks[11], landmarks[12], (255, 255, 255), thickness)

        cv.line(image, landmarks[0], landmarks[13], (255, 255, 255), thickness)
        cv.line(image, landmarks[13], landmarks[14], (255, 255, 255), thickness)
        cv.line(image, landmarks[14], landmarks[15], (255, 255, 255), thickness)
        cv.line(image, landmarks[15], landmarks[16], (255, 255, 255), thickness)

        cv.line(image, landmarks[0], landmarks[17], (255, 255, 255), thickness)
        cv.line(image, landmarks[17], landmarks[18], (255, 255, 255), thickness)
        cv.line(image, landmarks[18], landmarks[19], (255, 255, 255), thickness)
        cv.line(image, landmarks[19], landmarks[20], (255, 255, 255), thickness)

        if is_draw_point:
            for p in landmarks:
                cv.circle(image, p, thickness, (0, 0, 255), -1)

    conf = handpose[-1]
    bbox = handpose[0:4].astype(np.int32)
    handedness = handpose[-2]
    if handedness <= 0.5:
        handedness_text = 'Left'
    else:
        handedness_text = 'Right'
    landmarks_screen = handpose[4:67].reshape(21, 3).astype(np.int32)
    landmarks_world = handpose[67:130].reshape(21, 3)

    # draw box
    cv.rectangle(display_screen, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    # draw handedness
    cv.putText(display_screen, '{}'.format(handedness_text), (bbox[0], bbox[1] + 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    # Draw line between each key points
    landmarks_xy = landmarks_screen[:, 0:2]
    draw_lines(display_screen, landmarks_xy, is_draw_point=False)

    # z value is relative to WRIST
    for p in landmarks_screen:
        r = max(5 - p[2] // 5, 0)
        r = min(r, 14)
        cv.circle(display_screen, np.array([p[0], p[1]]), r, (0, 0, 255), -1)

    if is_draw is False:
        is_draw = True
        # Main view
        landmarks_xy = landmarks_world[:, [0, 1]]
        landmarks_xy = (landmarks_xy * 1000 + 100).astype(np.int32)
        draw_lines(display_3d, landmarks_xy, thickness=5)
        cv.circle(display_3d, (100,100), 5, (255, 0, 0), -1)

        # Top view
        landmarks_xz = landmarks_world[:, [0, 2]]
        landmarks_xz[:, 1] = -landmarks_xz[:, 1]
        landmarks_xz = (landmarks_xz * 1000 + np.array([300, 100])).astype(np.int32)
        draw_lines(display_3d, landmarks_xz, thickness=5)

        # Left view
        landmarks_yz = landmarks_world[:, [2, 1]]
        landmarks_yz[:, 0] = -landmarks_yz[:, 0]
        landmarks_yz = (landmarks_yz * 1000 + np.array([100, 300])).astype(np.int32)
        draw_lines(display_3d, landmarks_yz, thickness=5)

        # Right view
        landmarks_zy = landmarks_world[:, [2, 1]]
        landmarks_zy = (landmarks_zy * 1000 + np.array([300, 300])).astype(np.int32)
        draw_lines(display_3d, landmarks_zy, thickness=5)

    return display_screen, display_3d


if __name__ == '__main__':
    # robot hand
    # Linux系统应该是 /dev/ttyUSB0 或类似
    # macOS系统应该是 /dev/cu.usbserial-1110 或类似
    # Windows系统应该是 COM3 或类似
    robotHand = HiwonderHand('/dev/cu.usbserial-1220', 9600)
    robotHand.setMotionTime(50)
    # palm detector
    palm_detector = MPPalmDet(modelPath='../models/palm_detection_mediapipe_2023feb.onnx',
                              nmsThreshold=0.3,
                              scoreThreshold=0.6,
                              # backendId=cv.dnn.DNN_BACKEND_TIMVX,
                              # targetId=cv.dnn.DNN_TARGET_NPU
                              )
    # handpose detector
    handpose_detector = MPHandPose(modelPath='../models/handpose_estimation_mediapipe_2023feb.onnx',
                                   confThreshold=args.conf_threshold,
                                   # backendId=cv.dnn.DNN_BACKEND_TIMVX,
                                   # targetId=cv.dnn.DNN_TARGET_NPU
                                   )

    cap = cv.VideoCapture(args.input)

    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        tm.start()
        # Palm detector inference
        palms = palm_detector.infer(frame)
        hands = np.empty(shape=(0, 132))

        # Estimate the pose of each hand
        if len (palms) == 1:
            # for palm in palms:
            palm = palms[0]
            # Handpose detector inference
            handpose = handpose_detector.infer(frame, palm)
            tm.stop()

            if handpose is not None:
                # control robot hand/fingers
                bending1, bending2, bending3, bending4, bending5 = getFingerBendings(handpose)
                # 剪子包袱锤对战
                rps = recognizeHandPose(bending1, bending2, bending3, bending4, bending5)
                if rps =='Scissors':
                    robotHand.setRock()
                elif rps =='Rock':
                    robotHand.setPaper()
                elif rps =='Paper':
                    robotHand.setScissors()
                else:
                    robotHand.setNoAction()

                cv.putText(frame, 'Pose: '+rps, (100, 115), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Draw results on the input image
                frame, view_3d = visualize(frame, handpose)
                print('finger bending: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(bending1, bending2, bending3, bending4, bending5))
                cv.imshow('3D Pose', view_3d)

        elif len (palms) > 1:
            tm.stop()
            cv.putText(frame, 'Too many hands', (100, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        else:
            tm.stop()
            cv.putText(frame, 'No hand', (100, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0))
        tm.reset()
        cv.imshow('Hand', frame)
