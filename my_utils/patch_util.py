
try:
    import torch
except ModuleNotFoundError:
    pass
try:
    import tensorflow as tf
except ModuleNotFoundError:
    pass
import numpy as np
from my_utils import tf_util as tu

def initialize_patch(img_size=416, patch_frac=0.015):
    # Initialize adversarial patch w/ random values
    img_size = img_size * img_size
    patch_size = img_size * patch_frac
    patch_dim = int(patch_size**(0.5)) # sqrt
    patch = None
    try:
        patch = torch.rand(1, 3, patch_dim, patch_dim)
    except:
        patch = np.random.rand(1, 3, patch_dim, patch_dim)
        patch = tf.Variable(patch, dtype=tf.float32, trainable=True)
    
    # if patch == None:
        # raise Exception("initialize_patch(...) did not initialize patch")
    return patch

def get_patch_dummy(patch, data_shape, patch_x, patch_y):
    """
    :param patch: Attack patch to be superimposed on original image
    :type patch: np.array of dim (1, 3, patch_dim, patch_dim)
    :param data_shape: Shape of the original image
    :type data_shape: np.array of rank 4 (batch, channel, dim, dim)
                        or (batch, dim, dim, channel)
    :param patch_x: Patch x location (top, left location)
    :type patch_x: int
    :param patch_y: Patch y location (top, left location)
    :type patch_y: int

    :return dummy: Empty image with attached patch
    :type dummy: tensor of shape (batch, channel, dim, dim)
    """
    # Get dummy image which we will place attack patch on.
    dummy = np.zeros(data_shape, dtype=np.float32)
    
    # Get width or height dimension of patch
    patch_size = patch.shape[-1] # patch.shape (1, 3, patch_size, patch_size)
       
    # Apply patch to dummy image - dummy is channel FIRST
    dummy[0][0][patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] = patch[0][0]
    dummy[0][1][patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] = patch[0][1]
    dummy[0][2][patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] = patch[0][2]
    
    return dummy

def get_img_and_patch_masks(patch_dummy):
    """
    :param patch_dummy: Empty tensor with patch attached
    :type patch_dummy: tensor of shape (batch, channel, img dim, img dim)

    :return img_mask: Masks the image
    :type img_mask: tensor of shape (batch, channel, image width, image height)
    :return patch_mask: Masks the patch
    :type patch_mask: tensor of shape (batch, channel, image width, image height)
    """
    img_mask = None
    try:
        img_mask = patch_dummy.clone()
    except:
        img_mask = np.copy(patch_dummy)
    img_mask[img_mask != 0] = 1.0 # Turn patch values into 1's
    patch_mask = (1 - img_mask)
    return img_mask, patch_mask

def patch_on_img(patch_dummy, img, patch_mask):
    # Make sure img and patch_mask are same shape
    if patch_mask.shape != img.shape:
        # Means that img.shape is channel LAST
        patch_mask_copy = tu.channel_to_last(patch_mask)
        patch_dummy_copy = tu.channel_to_last(patch_dummy)

    # img with patch area masked out
    try:
        cutout_img = torch.mul(patch_mask, img)
    except NameError:
        cutout_img = tf.multiply(patch_mask_copy, img)

    adv_img = cutout_img + patch_dummy_copy # img with patch
    return adv_img

def get_patch_loc(attack_bbox, patch_side):
    """ Return center location of attacking bbox to attach patch to
    """
    new_h =  int(((attack_bbox[1] + attack_bbox[3]) / 2) - patch_side/2) # y == h
    new_w = int(((attack_bbox[0] + attack_bbox[2]) / 2) - patch_side/2) # x == w
    return new_h, new_w