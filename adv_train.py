from absl import app, flags, logging
from absl.flags import FLAGS

import os
# Suppress ridiculous amount of tf logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import tensorflow as tf
import numpy as np
import cv2
import time
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all, draw_outputs
import yolov3_tf2.dataset as dataset
import my_utils.patch_util as pu
import my_utils.tf_util as tu

flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'eager_tf', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'frozen',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 400, 'number of epochs')
flags.DEFINE_integer('batch_size', 1, 'batch size')
flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_boolean('multi_gpu', False, 'Use if wishing to train with more than 1 GPU.')


def setup_model():
    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    # Configure the model for transfer learning
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    elif FLAGS.transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes
        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        else:
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))
        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                    freeze_all(l)
    else:
        # All other transfer require matching classes
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss,
                  run_eagerly=(FLAGS.mode == 'eager_fit'))

    return model, optimizer, loss, anchors, anchor_masks


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Setup
    if FLAGS.multi_gpu:
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        BATCH_SIZE = FLAGS.batch_size * strategy.num_replicas_in_sync
        FLAGS.batch_size = BATCH_SIZE

        with strategy.scope():
            model, optimizer, loss, anchors, anchor_masks = setup_model()
    else:
        model, optimizer, loss, anchors, anchor_masks = setup_model()

    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    else:
        train_dataset = dataset.load_fake_dataset()
    # train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    # train_dataset = train_dataset.prefetch(
        # buffer_size=tf.data.experimental.AUTOTUNE)

    if FLAGS.mode == 'eager_tf':
        try:
            shutil.rmtree("adv_output")
        except FileNotFoundError:
            pass
        os.makedirs("adv_output", exist_ok=True)


        class_names = [c.strip() for c in open(FLAGS.classes).readlines()]

        # images are loaded in shape (1, 416, 416, 3)
        patch = pu.initialize_patch(patch_frac=0.05) # shape (1, 3, 416, 416)
        dummy_img_shape = (1, 3, 416, 416)
        patch_x, patch_y = 0, 0
        patch_dummy = pu.get_patch_dummy(
            patch, dummy_img_shape, patch_x, patch_y) # shape (1, 3, 416, 416)
        img_mask, patch_mask = pu.get_img_and_patch_masks(patch_dummy)

        # Detection model - to test whether patch attacks successfully
        yolo = YoloV3(classes=FLAGS.num_classes)
        yolo.load_weights(FLAGS.weights)
        
        patch_dummy = pu.get_patch_dummy(patch, dummy_img_shape, patch_x, patch_y)

        train_set = list(train_dataset.take(1))
        images = train_set[0][0]
        labels = train_set[0][1]

        # for batch, (images, labels) in enumerate(train_dataset):
        for epoch in range(0, FLAGS.epochs + 1):
        # patch_dummy shape (1, 3, 416, 416)
        # adv_img shape (1, 416, 416, 3)
            # Note: labels is different from what I expected...
            # lables[1] contain label for chair and cellphone
            # labels[0] contain label for person
            # print(labels[0].shape)
            # print(labels[1].shape)
            # print(labels[2].shape)

            images = pu.patch_on_img(patch_dummy, images, patch_mask)
            images = tf.Variable(images)

            with tf.GradientTape() as tape:
                outputs = model(images, training=False)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                # exit()
                total_loss = tf.reduce_sum(pred_loss) # + regularization_loss
                # total_loss = regularization_loss
                # total_loss = tf.math.log(total_loss)

            grads = tape.gradient(total_loss, images)

            # Non-optimizer
            # grad = tu.channel_to_first(grads)
            # grad = tf.multiply(img_mask, grad)
            # patch_dummy = patch_dummy + (0.5 * grad)

            # Optimizer
            patch_dummy = tu.channel_to_last(patch_dummy)
            patch_dummy = tf.Variable(patch_dummy, trainable=True)
            optimizer.apply_gradients([(grads, patch_dummy)])
            patch_dummy = tu.channel_to_first(patch_dummy)
            patch_dummy = tf.multiply(img_mask, patch_dummy)

            # Clip patch values
            patch_dummy = tf.clip_by_value(patch_dummy, 0.0, 1.0)
            
            if epoch % 25 == 0:
                logging.info("{}_train_{}, {}, {}".format(
                        epoch, 0, total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss))))

            if epoch % 100 == 0:
                boxes, scores, classes, nums = yolo(images, training=False)
                logging.info('\ndetections:')
                for i in range(nums[0]):
                    logging.info('\t{}, {}, {}'.format(
                        class_names[int(classes[0][i])],
                        np.array(scores[0][i]),
                        np.array(boxes[0][i])))
                    
                # cv2 (OpenCV) assumes img to be BGR/BGRA
                save_img = cv2.cvtColor(images.numpy()[0], cv2.COLOR_RGB2BGR)
                save_img *= 255 # transform_images(...) scaled img btwn [0, 1]
                save_img = draw_outputs(save_img, (boxes, scores, classes, nums), class_names)
                cv2.imwrite("./adv_output/output_{}.jpg".format(epoch), save_img)                   

            if epoch == FLAGS.epochs:
                print(" ----- TRAINING IS FINISHED ------ ")
                boxes, scores, classes, nums = yolo(images, training=False)
                # cv2 (OpenCV) assumes img to be BGR/BGRA
                images = cv2.cvtColor(images.numpy()[0], cv2.COLOR_RGB2BGR)
                images *= 255 # transform_images(...) scaled img btwn [0, 1]
                images = draw_outputs(images, (boxes, scores, classes, nums), class_names)
                cv2.imwrite("./adv_output/output_final.jpg", images)

                logging.info('detections:')
                for i in range(nums[0]):
                    logging.info('\t{}, {}, {}'.format(
                        class_names[int(classes[0][i])],
                        np.array(scores[0][i]),
                        np.array(boxes[0][i])))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
