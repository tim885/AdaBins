from infer import InferenceHelper
from PIL import Image

infer_helper = InferenceHelper(dataset='nyu')

# predict depths of images stored in a directory and store the predictions in 16-bit format in a given separate dir
infer_helper.predict_dir("/path/to/input/dir/containing_only_images/", "path/to/output/dir/")
