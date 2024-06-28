import cv2 as cv

# 1. Load input and deep learning model
frame = cv.imread('lena.jpg')
net = cv.dnn.readNet('../models/la_muse.t7')

# 2. Pre-processing
inWidth = frame.shape[1]
inHeight = frame.shape[0]
inp = cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                          (103.939, 116.779, 123.68), swapRB=False, crop=False)

# 3. Set model input
net.setInput(inp)

# 4. Infer
out = net.forward()

# 5. Post-processing
out = out.reshape(3, out.shape[2], out.shape[3])
out[0] += 103.939
out[1] += 116.779
out[2] += 123.68
out /= 255
out = out.transpose(1, 2, 0)

cv.imshow('source', frame)
cv.imshow('transfer', out)
cv.waitKey()
cv.destroyAllWindows()

