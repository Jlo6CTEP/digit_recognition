import cv2

from neural_network.digit_guess import i_io

nines = i_io.images


result = []

result = cv2.hconcat([x.reshape((28, 28)) for x in nines])

cv2.imshow('nines', result)
cv2.waitKey()
