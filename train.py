import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np
import os
import PIL
import matplotlib.pyplot as plt
# import VGG16
import datetime
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') # 现在
from PIL import Image  # 注意Image,后面会用到

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
plt.switch_backend('agg')

_BATCH_SIZE = None

X = tf.placeholder(tf.float32, [_BATCH_SIZE, 224, 224, 3])
Y = tf.placeholder(tf.int32, [_BATCH_SIZE, 8])

sess = tf.InteractiveSession()


def VGG16(image):
    preprocessed = tf.multiply(tf.subtract(image/255, 0.5), 2.0)
    arg_scope = nets.vgg.vgg_arg_scope(weight_decay=5e-4)
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.vgg.vgg_16(preprocessed, 9, is_training=True)#留空 什么也不分类的
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


log, pro, end = VGG16(X)

correct_p = tf.equal(tf.argmax(pro, 1), (tf.argmax(Y, 1)))
accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))
tf.summary.scalar('accuracy', accuracy)

loss = tf.losses.softmax_cross_entropy(Y, log)
tf.summary.scalar('total_loss', loss)

global_steps = tf.Variable(0, trainable=False)

lr = tf.train.exponential_decay(0.001, global_steps, 5000, 0.97, staircase=False)# 5000轮衰减0.97
tf.summary.scalar('lr', lr)

Train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_steps)

train_writer = tf.summary.FileWriter("model/log/train_"+str(nowTime), sess.graph)
summary_op = tf.summary.merge_all()

#读取并编码函数
def read_and_decode(filename, batch_size):
    path = "/home/zhou/data/jiaqi/dataset/weapon/data/"
    filename_queue = tf.train.string_input_producer([os.path.join(path, filename)])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feature = {'label': tf.FixedLenFeature([], tf.int64),
               'test_img': tf.FixedLenFeature([], tf.string)}

    features = tf.parse_single_example(serialized_example, features=feature)
    #读label 并且Onehot
    label = features['label']
    label_o = label
    #print("label is: ",label)
    label = tf.one_hot(label, 8)
    #print("One_hot encoding is: ",label)
    label_onehot = label
    #读取图像
    A = features['test_img']
    images = tf.decode_raw(A, tf.uint8)#
    images = tf.reshape(images, shape=[224, 224, 3])
    batch = tf.train.shuffle_batch([images, label,label_o], batch_size=batch_size, num_threads=256, capacity=10000,
                                   min_after_dequeue=5000)
    return batch


restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('vgg_16/')
]
restore_vars = restore_vars[:30]#读预训练的models时候用 读模型参数的前30个

saver = tf.train.Saver(restore_vars, max_to_keep=5, allow_empty=True)#最大保留5个Model

def save():
    saver.save(sess, "model/VGG16_model_"+str(nowTime)+".ckpt")

#读取预训练的VGG16
def restore():
    saver.restore(sess, "model/pretrain_model/vgg_16.ckpt")


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

train_batch = read_and_decode("train.tfrecords", 8)
test_batch = read_and_decode("test.tfrecords", 32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    restore()#图节点中的运算 才要sess.run()  其他的操作不用
    #载入VGG16model
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)
    print("################################sess.run(train_batch)#######################")
    #检查onehot 类标用的
    # for j in range(781*8//32):
    #     B= sess.run([train_batch])
    #     print("label:{}".format(B[0][2]))
    #     print("label_onehot:{}".format(B[0][1]))
    for j in range(100):
        for i in range(781*8//32):
            B = sess.run([train_batch])
            _, L, s, A, gs = sess.run([Train_op, loss, summary_op, accuracy, global_steps], feed_dict={X: B[0][0], Y: B[0][1]})
            print('Epoach:{},Batch:{},LOSS:{}, ACC:{}'.format(j,i,L, A))
            train_writer.add_summary(s, gs)
        B_test = sess.run([test_batch])
        L_t, A_t = sess.run([loss, accuracy], feed_dict={X: B_test[0][0], Y: B[0][1]})
        print('Epoach:{},test: LOSS:{}, ACC:{}'.format(j,L, A))
        save()

    coord.request_stop()
    coord.join(threads)
