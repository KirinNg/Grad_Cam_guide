import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np
import os
import PIL
import matplotlib.pyplot as plt
import datetime
import cv2
from skimage.transform import resize


nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # 现在
from PIL import Image  # 注意Image,后面会用到
import skimage

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
plt.switch_backend('agg')

_BATCH_SIZE = None

X = tf.placeholder(tf.float32, [_BATCH_SIZE, 224, 224, 3])
Y = tf.placeholder(tf.int32, [_BATCH_SIZE, 8])

sess = tf.InteractiveSession()


def load_image(path, normalize=True):
    img = skimage.io.imread(path)
    if normalize:
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (224, 224), preserve_range=True)
    return resized_img


def VGG16(image):
    preprocessed = tf.multiply(tf.subtract(image / 255, 0.5), 2.0)
    arg_scope = nets.vgg.vgg_arg_scope(weight_decay=5e-4)
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.vgg.vgg_16(preprocessed, 9, is_training=True)  # 留空 什么也不分类的
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


log, pro, end = VGG16(X)
onehot_pro = tf.one_hot(tf.argmax(pro), 9)


def grad_cam(end_point, pre_calss_one_hot, layer_name='vgg_16/pool5'):
    conv_layer = end_point[layer_name]
    signal = tf.multiply(end_point['vgg_16/fc8'], pre_calss_one_hot)
    loss = tf.reduce_mean(signal, 1)
    grads = tf.gradients(loss, conv_layer)[0]
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
    weights = tf.reduce_mean(norm_grads, axis=(1, 2))
    weights = tf.expand_dims(weights, 1)
    weights = tf.expand_dims(weights, 1)
    weights = tf.tile(weights, [1, 7, 7, 1])
    pre_cam = tf.multiply(weights, conv_layer)
    cam = tf.reduce_sum(pre_cam, 3)
    cam = tf.expand_dims(cam, 3)
    cam = tf.reshape(cam, [-1, 49])
    cam = tf.nn.softmax(cam)
    cam = tf.reshape(cam, [-1, 7, 7, 1])
    # cam = tf.nn.relu(cam)
    resize_cam = tf.image.resize_images(cam, [224, 224])
    # resize_cam = resize_cam / tf.reduce_max(resize_cam)
    return resize_cam, grads


cost = tf.losses.softmax_cross_entropy(Y, log)
gb_grad = tf.gradients(cost, X)[0]


def visualize(image, conv_output, conv_grad, gb_viz):
    output = conv_output  # [7,7,512]
    grads_val = conv_grad  # [7,7,512]
    print("grads_val shape:", grads_val.shape)
    print("gb_viz shape:", gb_viz.shape)

    weights = np.mean(grads_val, axis=(0, 1))  # alpha_k, [512]
    cam = np.zeros(output.shape[0: 2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)  # scale 0 to 1.0
    cam = resize(cam, (224, 224), preserve_range=True)

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    # print(img)
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    # cam = np.float32(cam) + np.float32(img)
    # cam = 255 * cam / np.max(cam)
    # cam = np.uint8(cam)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(img)
    ax.set_title('Input Image')

    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(131)
    imgplot = plt.imshow(cam_heatmap)
    ax.set_title('Grad-CAM')

    gb_viz = np.dstack((
        gb_viz[:, :, 0],
        gb_viz[:, :, 1],
        gb_viz[:, :, 2],
    ))
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()

    ax = fig.add_subplot(132)
    imgplot = plt.imshow(gb_viz)
    ax.set_title('guided backpropagation')

    gd_gb = np.dstack((
        gb_viz[:, :, 0] * cam,
        gb_viz[:, :, 1] * cam,
        gb_viz[:, :, 2] * cam,
    ))
    ax = fig.add_subplot(133)
    imgplot = plt.imshow(gd_gb)
    ax.set_title('guided Grad-CAM')

    plt.savefig("test.png")
    # cv2.imshow("", gd_gb)
    # cv2.waitKey(0)


def save():
    saver.save(sess, "model/VGG16_model_" + str(nowTime) + ".ckpt")


def restore():
    saver.restore(sess, "VGG16_model_2019-03-14-11-42-51.ckpt")


# graph
gc, target_conv_layer_grad = grad_cam(end, onehot_pro)

img = load_image('test_img2_Label_3.jpg')

restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('vgg_16/')
]
restore_vars = restore_vars[:30]  # 读预训练的models时候用 读模型参数的前30个

saver = tf.train.Saver(restore_vars, max_to_keep=5, allow_empty=True)  # 最大保留5个Model

sess = tf.Session()
sess.run(tf.global_variables_initializer())
restore()
label = np.zeros([1, 8])
label[0, 1] = 1
gc0, pool5_value, tcl_value, gb0 = sess.run([gc, end['vgg_16/pool5'], target_conv_layer_grad, gb_grad],
                                            feed_dict={X: [img], Y: label})

# heatmap = np.uint8(255 * gc0[0, :, :, 0])  # 将热力图转换为RGB格式
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# img = img + heatmap * 0.4

gb0 = gb0[0, :, :, :]
gradRGB = np.dstack((gb0[:, :, 2], gb0[:, :, 1], gb0[:, :, 0],))

visualize(img, pool5_value[0, :, :, :], tcl_value[0, :, :, :], gradRGB)

