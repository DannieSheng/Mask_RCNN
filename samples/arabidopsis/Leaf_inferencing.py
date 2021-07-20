"""
Mask R-CNN
Trained on

------------------------------------------------
Usage
"""

# Set matplotlib backend 
# This has to be done before other import that might set it
# But only if we're running in script mode. 

# if __name__ == '__main__':
#     import matplotlib
#     # Set 'Agg' as backend which cant display
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os 
import sys
import datetime 
import numpy as np
import skimage.io
import pickle as pkl
import json
from plantcv import plantcv as pcv 

# Root directory of the project. You can modify according to your path 
# ROOT_DIR = '/mnt/efs/data/Mask_RCNN'
ROOT_DIR = os.path.abspath("../../mrcnn")
# Import Mask RCNN 
sys.path.append(ROOT_DIR)
from mrcnn.config import Config 
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import xarray as xr

def _get_ax(rows=1, cols=1, size=16):  # ???
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    fig.tight_layout()
    return ax

PROJ_DIR = os.path.abspath("")
DATA_DIR = os.path.join(PROJ_DIR, 'test_imgs')
# Path to trained weights file. Put the pre-trained weights file under PROJ_DIR
LEAF_WEIGHTS_PATH = os.path.join(PROJ_DIR, 'mask_rcnn_leaves_0060.h5')


# # Generate the list of image_id for train, validation and test dataset
# # The generate_ID function assume the image and mask are stored like
# # synthetic_arabidopsis (image file and mask file are under dataset_dir.
# # There is no sub folder here)
# # If you have different structure of dataset, please modify the generate_IDs
# # function in the gen_IDs.py
# num_images, Image_IDs_train, Image_IDs_val, Image_IDs_test = gen_IDs.generate_IDs(dataset_dir)

# Results directory
RESULTS_DIR = os.path.join(PROJ_DIR, 'results')

# Direcotry to save logs and model checkpoints, if not provided 
# through the command line argument --logs 
DEFAULT_LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

# Set Hyperparameter for Training

class LeavesConfig(Config):

    """Configuration for training on the Synthetic Arabidopsis dataset.
    Derives from the base Config class and overrides values specific to
    the leave dataset """

    # Give the configuration a recognizable name  
    NAME = 'leaves'
    
    # Number of classes(including background)
    NUM_CLASSES = 1 + 1 # background + leaves 
    
    # Train on 1 GPU AND 5 images per GPU. We can put multiples images on each 
    # GPU because the images are samll. Batch size is 5 (GPU * images/GPU)
    GPU_COUNT = 1 
    IMAGES_PER_GPU = 4  # Modify according to your GPU memory. We trained this on AWS P2
    Batch_size = GPU_COUNT * IMAGES_PER_GPU
    
    # # Number of training and validation steps per epoch
    # STEPS_PER_EPOCH = (num_images * 0.6)// Batch_size  # define dataset_IDS
    # VALIDATION_STEPS = max(0, (num_images * 0.2) // Batch_size)
    
    # Don't exclude based on confidence. ##??
    DETECTION_MIN_CONFIDENCE = 0.8
    DETECTION_NMS_THRESHOLD = 0.48
    
    # Backbone network architecture 
    # Supported values are: resnet50, resnet101
    BACKBONE = 'resnet50'
    
    # Input image resizing 
    # Random crops of size 512x512 
    IMAGE_RESIZE_MODE = 'crop'
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0 
    
    # Length of square anchor side in pixels.
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)  
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9    

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256  

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])  

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True  
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 150

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 50   ## ?? you can adjust this to smaller number

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 100  # ?? this number can be much less


# Set Hyperparameter for Testing
    
class LeavesInferenceConfig(LeavesConfig):
    # Set batch size to 1 to run and inference one image at a time 
    GPU_COUNT = 1 
    IMAGES_PER_GPU = 1
    # Don't resize image for inferencing 
    IMAGE_RESIZE_MODE = 'pad64'
    # Non-max suppression threhold to filter RPN proposals 
    # You can increase this during training to generate more proposals
    RPN_NMS_THRESHOLD = 0.9

    IMAGE_RESIZE_MODE = 'square'
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 0


class LeavesDataset(utils.Dataset):
    def __init__(self):
        super().__init__()
        self.image_ds = None
        # def load_leaves(self, dataset_dir, img_names):
    #     """Load a subset of the leaf dataset.
    #
    #     dataset_dir: Root direcotry of the dataset
    #     image_ids: A list that includes all the images to load
    #     """
    #     # Add classes. We only have one class to add---leaves
    #     # Naming the dataset 'leaves'
    #     self.add_class('leaves', 1, 'leaves')
    #
    #     # Add images
    #     for (image_id, img_name) in enumerate(img_names):
    #         self.add_image(
    #             'leaves',
    #             image_id=image_id,
    #             path=os.path.join(dataset_dir, img_name))  ##

    ### Test a new definition of "LeavesDataset" that takes the x-array inputs
    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_leaves(self, dataset_dir, img_names):
        """Load a subset of the leaf dataset.

        dataset_dir: Root direcotry of the dataset
        image_ids: A list that includes all the images to load
        """
        # Add classes. We only have one class to add---leaves
        # Naming the dataset 'leaves'
        self.add_class('leaves', 1, 'leaves')

        # Add images
        for (image_id, img_name) in enumerate(img_names):
            self.add_image(
                'leaves',
                image_id=image_id,
                path=os.path.join(dataset_dir, img_name))  ##

    def load_mask(self, mask_dir, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # # Get mask directory from image path
        # mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .pkl image
        # mask = []
        # for f in next(os.walk(mask_dir))[2]:
        #     if f.endswith(".png"):
        #         m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
        #         mask.append(m)
        # mask = np.stack(mask, axis=-1)
        mask = pkl.load(open(os.path.split(info['path'])[1].replace(".png", ".pkl"), "rb"))["masks"]
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def load_image_ds(self, dataset_dir, plant_idx = 1):
        self.image_ds = xr.open_dataset(dataset_dir)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            import pdb
            pdb.set_trace()
        #     image = skimage.color.gray2rgb(image)
        # # If has an alpha channel, remove it for consistency
        # if image.shape[-1] == 4:
        #     image = image[..., :3]
        return image


def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    """Encodes instance masks to submission format."""
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


# Inferencing
def detect(mrcnn_model, dataset_dir, img_names, save_dir):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory to store detecting result. 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(save_dir, submit_dir)
    os.makedirs(submit_dir)

    obj_config = dict((name, getattr(mrcnn_model.config,name)) for name in
                      ["DETECTION_MIN_CONFIDENCE", "DETECTION_NMS_THRESHOLD", "IMAGE_MAX_DIM","IMAGE_MIN_DIM",
                       "IMAGE_MIN_SCALE", "IMAGE_RESIZE_MODE","IMAGE_SHAPE"])
    # obj_config = dict((name, getattr(mrcnn_model.config,name)) for name in dir(mrcnn_model.config) if (not name.startswith('__')))
    for key in obj_config:
        if type(obj_config[key]) is np.ndarray:
            obj_config[key] = obj_config[key].tolist()

    # save config to json file
    with open(os.path.join(submit_dir,'config.json'), 'w') as outfile:
        json.dump(obj_config, outfile)

    # Load test dataset
    dataset_test = LeavesDataset()
    dataset_test.load_leaves(dataset_dir, img_names)
    dataset_test.prepare()

    # Load over images
    submission = []

    for image_id in dataset_test.image_ids:
        # Load image and run detection
        image = dataset_test.load_image(image_id)

        # crop (optional)
        # image = image[140:140+128,140:140+128,:]

        # Detect objects
        result = mrcnn_model.detect([image], verbose=1)

        r = result[0]

        # Encode image to RLE. Returns a string of multiple lines
        source_id  = dataset_test.image_info[image_id]["id"]
        image_name = dataset_test.image_info[image_id]['path']
        # rle = mask_to_rle(source_id, r["masks"], r["scores"])
        rle = mask_to_rle(image_name, r["masks"], r["scores"])

        submission.append(rle)
        # Save image with masks
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset_test.class_names, r['scores'],
                                    ax=_get_ax(rows=1, cols=1, size=16), show_bbox=True, show_mask=True, title="Predictions")
        vis_name = dataset_test.image_info[image_id]['path'].replace(dataset_dir,submit_dir)
        plt.savefig(vis_name)
        plt.close("all")
    # Save to csv file
    submission = "Image, EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


##%% Running on spyder directory instead command line
#
## Training 
### Configurations
#config = LeavesConfig()
##config.display()
### Create model 
#print('in mask RCNN +++++++++++++++++++++++++++++++++++++++++++++++')
#model = modellib.MaskRCNN(mode='training', config=config, model_dir= DEFAULT_LOGS_DIR)
### Select weights file to load
#weights_path = LEAF_WEIGHTS_PATH
#model.load_weights(weights_path, by_name=True)
#train(model, dataset_dir)

## Running using Command Line parsing

if __name__ == '__main__':
    import argparse
    import tensorflow as tf
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for leaf counting and segmentation')
    parser.add_argument('--command', required=False, default='detect', metavar="<command>", help="'train' or 'detect'")
    parser.add_argument('--img_names', required=False, metavar="/list/of/images/", help='list of image names')
    parser.add_argument('--dataset_dir', required=False, metavar="/path/to/dataset_dir/",
                        help='Root directory of the dataset_dir')
    parser.add_argument('--weights', required=False, metavar="/path/to/weights.h5", help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--save_dir', required=False, metavar="/path/to/save/" )

    args = parser.parse_args()
    if not args.dataset_dir:
        args.dataset_dir = DATA_DIR
    if not args.weights:
        args.weights = LEAF_WEIGHTS_PATH
    if not args.save_dir:
        args.save_dir = RESULTS_DIR

    # Validate arguments
    if args.command == "train":
        assert args.dataset_dir, "Argument --dataset_dir is required for training"
    elif args.command == "detect":
        assert args.dataset_dir, "Provide --dataset_dir to run prediction on"
    else:
        print(f"'{args.command}' is not recognized. Use 'train' or 'detect'")

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset_dir)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = LeavesConfig()
    elif  args.command == "detect":
        config = LeavesInferenceConfig()
    config.display()

    # Create model
    with tf.device("/cpu:0"):
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    # Select weights file to load
    weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    if not args.img_names:
        args.img_names = os.listdir(args.dataset_dir)
    # else:
    #     args.img_names = args.img_names.split(" ")

    if args.command == "train":
        pass
    elif args.command == "detect":
        detect(model, args.dataset_dir, args.img_names, args.save_dir)



# ####### Temp #############
# import tensorflow as tf
# dataset_dir = DATA_DIR
# weights_path = LEAF_WEIGHTS_PATH
#
# assert dataset_dir, "Provide --dataset_dir to run prediction on"
#
# print("Weights: ", weights_path)
# print("Dataset: ", dataset_dir)
# print("Logs: ", DEFAULT_LOGS_DIR)
#
# config = LeavesInferenceConfig()
# config.display()
#
# with tf.device("/cpu:0"):
#     model = modellib.MaskRCNN(mode="inference", config=config,
#                               model_dir=DEFAULT_LOGS_DIR)
#
# # Load weights
# print("Loading weights ", weights_path)
# model.load_weights(weights_path, by_name=True)
#
# img_names = os.listdir(dataset_dir)
#
# # Train or evaluate
# if args.command == "train":
#     train(model, args.dataset_dir)
# elif args.command == "detect":
#     detect(model, dataset_dir, img_names)
# else:
#     print("'{}' is not recognized. "
#           "Use 'train' or 'detect'".format(args.command))

