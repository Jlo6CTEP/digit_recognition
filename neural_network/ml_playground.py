from neural_network.io_images import ImageIO
from neural_network.network import Network

nn = Network(64, 64, 3, 50, 100, True)
test_images = ImageIO(0)

success_count = 0
for x in test_images.train_set:
    if nn.evaluate_img(x[0]) == x[1]:
        success_count += 1
print("Success ratio: {}%".format(success_count / len(test_images.train_set) * 100.0))
