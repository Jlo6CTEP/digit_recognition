import cv2

from neural_network.digit_guess import i_io

nines = i_io.images


result = []
for x in range(0, len(i_io.images) - 19, 20):
    result.append(cv2.hconcat([x.reshape((28, 28)) for x in nines[x:x+20]]))

cv2.imshow('nines', cv2.vconcat(result))
cv2.waitKey()
