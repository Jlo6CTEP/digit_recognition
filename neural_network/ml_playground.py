from io_images import ImageIO
from neural_network import Network

nn = Network(64, 64, 3, 32, 100, True)
test_images = ImageIO(0)

print(nn.evaluate_img(test_images.train_set[0][0]))
print(test_images.train_set[0][1])
