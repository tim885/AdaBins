import numpy as np
from infer import InferenceHelper
from PIL import Image

infer_helper = InferenceHelper(dataset='nyu')

# predict depth of a single pillow image
img = Image.open("test_imgs/classroom__rgb_00283.jpg")  # any rgb pillow image
out_center_path = "test_imgs/classroom__rgb_00283_center.npy"
out_depth_path = "test_imgs/classroom__rgb_00283_depth.npy"
out_viz_path = "test_imgs/classroom__rgb_00283_depth_viz.png"

bin_centers, predicted_depth, viz = infer_helper.predict_pil(img, visualized=True)

np.save(out_center_path, bin_centers)
np.save(out_depth_path, predicted_depth)
viz.save(out_viz_path)
