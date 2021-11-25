import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.cm as cm
import random
import glob
from skimage.segmentation import chan_vese
from shutil import copy
from skimage.morphology import white_tophat, black_tophat, disk


def random_split_patients(unique_patients_ids, validation_split, seed):
   np.random.seed(seed)
   number_of_patients = len(unique_patients_ids)
   number_of_train_patients = int(number_of_patients * (1 - validation_split))
   
   np.random.shuffle(unique_patients_ids)
   train_patients_ids = unique_patients_ids[:number_of_train_patients]
   val_patients_ids = unique_patients_ids[number_of_train_patients:]
   
   return train_patients_ids, val_patients_ids

def copy_all_images_from_one_patient(patient_id, src_folder, dst_folder, metadata):
   try:
      patient_dataset = metadata.loc[metadata['Patient ID'] == patient_id]
      files = patient_dataset['File name']
      for file in files:
         src = src_folder + '\\' + file
         copy(src, dst_folder)
   except Exception as e:
      print('Copy error: {}'.format(e))

def check_dataset(dataset, color_map='viridis'):
   plt.figure(figsize=(10, 10))
   for images, labels in dataset.take(1):
      for i in range(9):
         ax = plt.subplot(3, 3, i + 1)
         plt.imshow(images[i].numpy().astype("uint8"), cmap=color_map)
         plt.title('...' + dataset.file_paths[i][-20:])
         plt.axis("off")

root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

root_modeldir = os.path.join(os.curdir, "my_models")
def get_model_dir():
   import time
   run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
   return os.path.join(root_modeldir, run_id)

def find_target_layer(model):
   # attempt to find the final convolutional layer in the network
   # by looping over the layers of the network in reverse order
   for layer in reversed(model.layers):
      # check to see if the layer has a 4D output
      if len(layer.output_shape) == 4:
         return layer.name
   # otherwise, we could not find a 4D layer so the GradCAM
   # algorithm cannot be applied
   raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

def get_img_array(img_path, size):
   # `img` is a PIL image of size 299x299
   img = keras.preprocessing.image.load_img(img_path, target_size=size)
   # `array` is a float32 Numpy array of shape (299, 299, 3)
   array = keras.preprocessing.image.img_to_array(img)
   # We add a dimension to transform our array into a "batch"
   # of size (1, 299, 299, 3)
   array = np.expand_dims(array, axis=0)
   return array
   
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def build_gradcam(img_path, heatmap, color_map, original_image_colormap, alpha=0.5):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path, color_mode=original_image_colormap)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    # jet = cm.get_cmap("jet")
    jet = cm.get_cmap(color_map)
    # jet = cm.get_cmap("OrRd")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # # Save the superimposed image
    # superimposed_img.save(cam_path)

    # # Display Grad CAM
    # display(Image(cam_path))

    # Return image
    return superimposed_img

def superimpose_gradcam(img_path, image_size, current_model, grad_colormap, original_image_colormap, last_layer_grad_cam):
   preprocess_input = keras.applications.xception.preprocess_input
   # img_path = "PetImages\\Cat\\1157.jpg"
   # Prepare image
   img_array = preprocess_input(get_img_array(img_path, size=image_size))

   # Remove last layer's softmax
   current_model.layers[-1].activation = None

   # Generate class activation heatmap
   last_conv_layer_name = find_target_layer(current_model)
   heatmap = make_gradcam_heatmap(img_array, current_model, last_conv_layer_name=last_layer_grad_cam)
   return build_gradcam(img_path, heatmap, color_map=grad_colormap, original_image_colormap=original_image_colormap)

def test_grad_cam_in_path(current_model, default_images_path, image_size, grad_colormap, original_image_colormap, last_layer_grad_cam):
   plt.figure(figsize=(10, 10))
   list_images = glob.glob(default_images_path + "*")
   for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      img_path = random.choice(list_images)
      superimposed_img = superimpose_gradcam( 
                        current_model=current_model, 
                        img_path=img_path,
                        image_size=image_size,
                        grad_colormap=grad_colormap,
                        original_image_colormap=original_image_colormap,
                        last_layer_grad_cam=last_layer_grad_cam)
      plt.imshow(superimposed_img)
      plt.title('{}'.format(img_path[-20:]))
      plt.axis("off")

def aply_segmentation_to_image(original_image_path, init_level_set='disk', mu=0.25):
   img_path = original_image_path
   img = keras.preprocessing.image.load_img(img_path)
   img_array = keras.preprocessing.image.img_to_array(img)
   
   # from 3d to 2d
   sample_slice = img_array[:, :, 0]

   # from 0-255 to 0-1
   normalize_slice = np.true_divide(sample_slice, [255.0], out=None)

   # aplyng chen algorithm
   lung_mask = chan_vese(
      normalize_slice, 
      mu=mu, 
      lambda1=1.0, 
      lambda2=1.0, 
      tol=0.001, 
      dt=0.5, 
      init_level_set=init_level_set, 
      extended_output=False).astype(int)

   extend_lung_mask = lung_mask[:, :, np.newaxis]

   lung_segmentation_result = extend_lung_mask * img_array[:, :, :]
   return lung_segmentation_result

def aply_segmentation_to_folder(original_folder_path, destination_folder_path, mu=0.25, init_level_set='disk'):
   list_images_paths = glob.glob(original_folder_path + "*")
   for current_image_path in list_images_paths:
      image_name = current_image_path.split('\\')[-1]
      # print("Current_image: {}".format(current_image_path))
      result_image = aply_segmentation_to_image(
         original_image_path=current_image_path, 
         init_level_set=init_level_set,
         mu=mu)
      tf.keras.utils.save_img(
         path="{}seg_version_{}".format(destination_folder_path, image_name),
         x=result_image, 
         data_format=None, 
         file_format='png', 
         scale=True
      )

def find_last_layer(model):
   for layer in reversed(model.layers):
      return layer
   
def aply_white_tophat_to_folder(original_folder_path, destination_folder_path, disk_size):
   list_images_paths = glob.glob(original_folder_path + "*")
   for current_image_path in list_images_paths:
      image_name = current_image_path.split('\\')[-1]
      print("Current_image: {}".format(current_image_path))
      result_image = aply_white_tophat_to_image(original_image_path=current_image_path, disk_size=disk_size)
      tf.keras.utils.save_img(
         path="{}seg_version_{}".format(destination_folder_path, image_name),
         x=result_image, 
         data_format=None, 
         file_format='png', 
         scale=True
      )

def aply_white_tophat_to_image(original_image_path, disk_size):
   img_path = original_image_path
   img = keras.preprocessing.image.load_img(img_path)
   img_array = keras.preprocessing.image.img_to_array(img)
   sample_slice = img_array[:, :, 0]
   normalize_slice = np.true_divide(sample_slice, [255.0], out=None)
   footprint = disk(disk_size)
   w_tophat = white_tophat(normalize_slice, footprint)
   w_tophat = np.uint8(255 * w_tophat)
   return w_tophat[:, :, np.newaxis]

def aply_black_tophat_to_image(original_image_path, disk_size):
   img_path = original_image_path
   img = keras.preprocessing.image.load_img(img_path)
   img_array = keras.preprocessing.image.img_to_array(img)
   sample_slice = img_array[:, :, 0]
   normalize_slice = np.true_divide(sample_slice, [255.0], out=None)
   footprint = disk(disk_size)
   w_tophat = black_tophat(normalize_slice, footprint)
   w_tophat = np.uint8(255 * w_tophat)
   return w_tophat[:, :, np.newaxis]

def aply_black_tophat_to_folder(original_folder_path, destination_folder_path, disk_size):
   list_images_paths = glob.glob(original_folder_path + "*")
   for current_image_path in list_images_paths:
      image_name = current_image_path.split('\\')[-1]
      print("Current_image: {}".format(current_image_path))
      result_image = aply_black_tophat_to_image(original_image_path=current_image_path, disk_size=disk_size)
      tf.keras.utils.save_img(
         path="{}seg_version_{}".format(destination_folder_path, image_name),
         x=result_image, 
         data_format=None, 
         file_format='png', 
         scale=True
      )