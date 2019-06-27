import cv2

from neural_network.io_images import ImageIO

test_images = ImageIO(0)
nines = [x[0] for x in test_images.train_set if x[1] == 9]


result = []

for x in range(0, len(nines) - 55, 50):
    f = nines[x:x + 20]
    result.append(cv2.hconcat([l.reshape((28, 28)) for l in f]))

result = cv2.vconcat(result)

cv2.imshow('nines', result)
cv2.waitKey()