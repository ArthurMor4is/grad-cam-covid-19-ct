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
   """
   Randomly splits the patients.

   Args:
       unique_patients_ids (_type_): Unique patients ids.
       validation_split (_type_): Validation split.
       seed (_type_): Seed.

   Returns:
       _type_: Train, validation and test ids.
   """
   np.random.seed(seed)
   number_of_patients = len(unique_patients_ids)
   number_of_train_patients = int(number_of_patients * (1 - validation_split))
   
   np.random.shuffle(unique_patients_ids)
   train_patients_ids = unique_patients_ids[:number_of_train_patients]
   val_patients_ids = unique_patients_ids[number_of_train_patients:]
   
   return train_patients_ids, val_patients_ids

def copy_all_images_from_one_patient(patient_id, src_folder, dst_folder, metadata):
   """
   Copies all images from one patient.

   Args:
       patient_id (_type_): Patient id.
       src_folder (_type_): Source folder.
       dst_folder (_type_): Destination folder.
       metadata (_type_): Metadata.

   Returns:
       _type_: None.
   """
   try:
      patient_dataset = metadata.loc[metadata['Patient ID'] == patient_id]
      files = patient_dataset['File name']
      for file in files:
         src = src_folder + '\\' + file
         copy(src, dst_folder)
   except Exception as e:
      print('Copy error: {}'.format(e))

def check_dataset(dataset, color_map='viridis'):
   """
   Checks the dataset.

   Args:
       dataset (_type_): Dataset.
       color_map (str, optional): Color map. Defaults to 'viridis'.

   Returns:
       _type_: None.
   """
   plt.figure(figsize=(10, 10))
   for images, labels in dataset.take(1):
      for i in range(9):
         ax = plt.subplot(3, 3, i + 1)
         plt.imshow(images[i].numpy().astype("uint8"), cmap=color_map)
         plt.title('...' + dataset.file_paths[i][-20:])
         plt.axis("off")

root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
   """
   Returns the logdir for the current run.

   Returns:
       _type_: str.
   """
   import time
   run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
   return os.path.join(root_logdir, run_id)

root_modeldir = os.path.join(os.curdir, "my_models")
def get_model_dir():
   """
   Returns the model dir for the current run.

   Returns:
       _type_: str.
   """

   import time
   run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
   return os.path.join(root_modeldir, run_id)

def find_target_layer(model):
   """
   Finds the target layer.

   Args:
       model (_type_): Model.

   Returns:
       _type_: Target layer.
   """
   for layer in reversed(model.layers):
      if len(layer.output_shape) == 4:
         return layer.name
   raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

def get_img_array(img_path, size):
   """
   Returns the image array.

   Args:
       img_path (_type_): Image path.
       size (_type_): Size.

   Returns:
       _type_: Image array.
   """
   img = keras.preprocessing.image.load_img(img_path, target_size=size)
   array = keras.preprocessing.image.img_to_array(img)
   array = np.expand_dims(array, axis=0)
   return array
   
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
   """
   Makes the gradcam heatmap.

   Args:
       img_array (_type_): Image array.
       model (_type_): Model.
       last_conv_layer_name (_type_): Last conv layer name.
       pred_index (_type_, optional): Prediction index. Defaults to None.

   Returns:
       _type_: Gradcam heatmap.
   """
   grad_model = tf.keras.models.Model(
      [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
   )

   with tf.GradientTape() as tape:
      last_conv_layer_output, preds = grad_model(img_array)
      if pred_index is None:
         pred_index = tf.argmax(preds[0])
      class_channel = preds[:, pred_index]

   grads = tape.gradient(class_channel, last_conv_layer_output)

   pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

   last_conv_layer_output = last_conv_layer_output[0]
   heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
   heatmap = tf.squeeze(heatmap)

   heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
   return heatmap.numpy()

def build_gradcam(img_path, heatmap, color_map, original_image_colormap, alpha=0.5):
   """
   Builds the gradcam.

   Args:
       img_path (_type_): Image path.
       heatmap (_type_): Heatmap.
       color_map (_type_): Color map.
       original_image_colormap (_type_): Original image colormap.
       alpha (float, optional): Alpha. Defaults to 0.5.

   Returns:
       _type_: Gradcam.
   """
   img = keras.preprocessing.image.load_img(img_path, color_mode=original_image_colormap)
   img = keras.preprocessing.image.img_to_array(img)

   heatmap = np.uint8(255 * heatmap)

   jet = cm.get_cmap(color_map)
   
   jet_colors = jet(np.arange(256))[:, :3]
   jet_heatmap = jet_colors[heatmap]

   jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
   jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
   jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

   superimposed_img = jet_heatmap * alpha + img
   superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

   return superimposed_img

def superimpose_gradcam(img_path, image_size, current_model, grad_colormap, original_image_colormap, last_layer_grad_cam):
   """
   Superimposes the gradcam.

   Args:
       img_path (_type_): Image path.
       image_size (_type_): Image size.
       current_model (_type_): Current model.
       grad_colormap (_type_): Grad colormap.
       original_image_colormap (_type_): Original image colormap.
       last_layer_grad_cam (_type_): Last layer grad cam.

   Returns:
       _type_: Superimposed gradcam.
   """
   preprocess_input = keras.applications.xception.preprocess_input
   img_array = preprocess_input(get_img_array(img_path, size=image_size))

   current_model.layers[-1].activation = None

   last_conv_layer_name = find_target_layer(current_model)
   heatmap = make_gradcam_heatmap(img_array, current_model, last_conv_layer_name=last_layer_grad_cam)
   return build_gradcam(img_path, heatmap, color_map=grad_colormap, original_image_colormap=original_image_colormap)

def test_grad_cam_in_path(current_model, default_images_path, image_size, grad_colormap, original_image_colormap, last_layer_grad_cam):
   """
   Tests the grad cam in path.

   Args:
       current_model (_type_): Current model.
       default_images_path (_type_): Default images path.
       image_size (_type_): Image size.
       grad_colormap (_type_): Grad colormap.
       original_image_colormap (_type_): Original image colormap.
       last_layer_grad_cam (_type_): Last layer grad cam.

   Returns:
       _type_: Superimposed gradcam.
   """
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
         last_layer_grad_cam=last_layer_grad_cam
      )
      plt.imshow(superimposed_img)
      plt.title('{}'.format(img_path[-20:]))
      plt.axis("off")

def aply_segmentation_to_image(original_image_path, init_level_set='disk', mu=0.25):
   """
   Aply segmentation to image.

   Args:
      original_image_path (_type_): Original image path.
      init_level_set (str, optional): Initial level set. Defaults to 'disk'.
      mu (float, optional): Mu. Defaults to 0.25.

   Returns:
       _type_: Segmented image.
   """
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
   """
   Aply segmentation to folder.

   Args:
       original_folder_path (_type_): Original folder path.
       destination_folder_path (_type_): Destination folder path.
       mu (float, optional): Mu. Defaults to 0.25.
       init_level_set (str, optional): Initial level set. Defaults to 'disk'.

   Returns:
       _type_: Segmented folder.
   """
   list_images_paths = glob.glob(original_folder_path + "*")
   for current_image_path in list_images_paths:
      image_name = current_image_path.split('\\')[-1]
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
   """
   Find last layer.

   Args:
       model (_type_): Model.

   Returns:
       _type_: Last layer.
   """
   for layer in reversed(model.layers):
      return layer
   
def aply_white_tophat_to_folder(original_folder_path, destination_folder_path, disk_size):
   """
   Aply white tophat to folder.

   Args:
       original_folder_path (_type_): Original folder path.
       destination_folder_path (_type_): Destination folder path.
       disk_size (_type_): Disk size.

   """
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
   """
   Aply white tophat to image.

   Args:
       original_image_path (_type_): Original image path.
       disk_size (_type_): Disk size.

   Returns:
       _type_: White tophat image.
   """
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
   """
   Aply black tophat to image.

   Args:
       original_image_path (_type_): Original image path.
       disk_size (_type_): Disk size.

   Returns:
       _type_: Black tophat image.
   """
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
   """
   Aply black tophat to folder.

   Args:
       original_folder_path (_type_): Original folder path.
       destination_folder_path (_type_): Destination folder path.
       disk_size (_type_): Disk size.
   """
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
