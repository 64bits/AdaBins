from infer import InferenceHelper
from PIL import Image
import matplotlib.pyplot as plt

infer_helper = InferenceHelper(dataset='nyu', device='cpu')

# predict depth of a batched rgb tensor
# example_rgb_batch = ...
# bin_centers, predicted_depth = infer_helper.predict(example_rgb_batch)

# predict depth of a single pillow image
img = Image.open("test_imgs/house.jpg")  # any rgb pillow image
bin_centers, predicted_depth = infer_helper.predict_pil(img)

plt.imshow(predicted_depth[0][0], cmap='Greys')
plt.show()

# predict depths of images stored in a directory and store the predictions in 16-bit format in a given separate dir
#infer_helper.predict_dir("./test_imgs", "./output")