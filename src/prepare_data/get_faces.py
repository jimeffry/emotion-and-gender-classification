import tensorflow as tf
import numpy as np
import os
#from print_ckpt import print_ckpt
from tensorflow.contrib import slim
import sys
from easydict import EasyDict as edict
import argparse
import cv2
import time

config = edict()

config.BATCH_SIZE = 256
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [640,1280,25600,51200]
# 5:test relu, 100: generate for MVtensor
config.train_face = 1
config.r_out = 0
config.P_Num = 3000
config.rnet_wide =0
config.o_out =0
config.Debug =0

def py_nms(dets, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    #order = scores.argsort()
    #print("order, ",order)

    keep = []
    while order.size > 0:
        i = order[0]
        #print("i",i)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        #keep
        #print("len over ",len(ovr))
        #get the opsite of the condition
        inds = np.where(ovr <= thresh)[0]
        #print("inds ",inds+1)
        # inds inlcude the first one : 0, inds+1 is keeping the <thresh;
        # because areas[order[1:]], so the lenth of order[1:] is less one than orignal order. so inds should plus 1
        order = order[inds+1]
    return keep


num_keep_radio = 0.7
#define prelu
test_fg = config.train_face
def prelu(inputs):
    #alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    alphas = 0.25
    pos = tf.nn.relu(inputs)
    if test_fg == 100 or test_fg==5:
        return pos
    else:
        #neg = alphas * (inputs-abs(inputs))*0.5
        neg = 0.25 * (inputs-abs(inputs))*0.5
        return pos +neg

def dense_to_one_hot(labels_dense,num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    #num_sample*num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
#cls_prob:batch*2
#label:batch

def cls_ohem(cls_prob, label):
    zeros = tf.zeros_like(label)
    #label=-1 --> label=0 net_factory
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int = tf.cast(label_filter_invalid,tf.int32)
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    row = tf.range(num_row)*2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    valid_inds = tf.where(label < zeros,zeros,ones)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #set 0 to invalid sample
    loss = loss * valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)
def bbox_ohem_smooth_L1_loss(bbox_pred,bbox_target,label):
    sigma = tf.constant(1.0)
    threshold = 1.0/(sigma**2)
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    abs_error = tf.abs(bbox_pred-bbox_target)
    loss_smaller = 0.5*((abs_error*sigma)**2)
    loss_larger = abs_error-0.5/(sigma**2)
    smooth_loss = tf.reduce_sum(tf.where(abs_error<threshold,loss_smaller,loss_larger),axis=1)
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    smooth_loss = smooth_loss*valid_inds
    _, k_index = tf.nn.top_k(smooth_loss, k=keep_num)
    smooth_loss_picked = tf.gather(smooth_loss, k_index)
    return tf.reduce_mean(smooth_loss_picked)
#label=1 or label=-1 then do regression
def bbox_ohem(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)
    #(batch,)
    square_error = tf.square(bbox_pred-bbox_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    #keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

def landmark_ohem(landmark_pred,landmark_target,label):
    #keep label =-2  then do landmark detection
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)
    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

def cal_accuracy(cls_prob,label):
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(label,tf.int64)
    cond = tf.where(tf.greater_equal(label_int,0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op
#construct Pnet
#label:batch

def print_shape(net,name,conv_num):
    print("the net {} in {}  shape is {} ".format(name,net,[conv_num.get_shape()]))

def P_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    #define common param
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        #print ("Pnet input shape",inputs.get_shape())
        #net = slim.conv2d(inputs, 10, 3, stride=1,scope='conv1')
        net = slim.conv2d(inputs, 8, 3, stride=1,scope='conv1')
        #print ("conv1 shape ",net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope='pool1', padding='SAME')
        #print ("pool1 shape ",net.get_shape())
        net = slim.conv2d(net,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')
        #print ("conv2 shape ",net.get_shape())
        net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')
        #print ("conv3 shape ",net.get_shape())
        #batch*H*W*2
        conv4_1 = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
        #conv4_1 = slim.conv2d(net,num_outputs=1,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.sigmoid)

        #print ("cls shape ",conv4_1.get_shape())
        #batch*H*W*4
        bbox_pred = slim.conv2d(net,num_outputs=4,kernel_size=[1,1],stride=1,scope='conv4_2',activation_fn=None)
        #print ("bbox shape ",bbox_pred.get_shape())
        #batch*H*W*10
        if test_fg:
            landmark_pred = slim.conv2d(net,num_outputs=10,kernel_size=[1,1],stride=1,scope='conv4_3',activation_fn=None)
            #print ("landmark shape ",landmark_pred.get_shape())
        #cls_prob_original = conv4_1
        #bbox_pred_original = bbox_pred
        if training:
            #batch*2
            cls_prob = tf.squeeze(conv4_1,[1,2],name='cls_prob')
            cls_loss = cls_ohem(cls_prob,label)
            #batch
            bbox_pred = tf.squeeze(bbox_pred,[1,2],name='bbox_pred')
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            #batch*10
            if test_fg:
                landmark_pred = tf.squeeze(landmark_pred,[1,2],name="landmark_pred")
                landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)
            else:
                landmark_loss = 0
            accuracy = cal_accuracy(cls_prob,label)
            #L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        #test
        else:
            #when test,batch_size = 1
            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            bbox_pred_test = tf.squeeze(bbox_pred,axis=0)
            if test_fg:
                landmark_pred_test = tf.squeeze(landmark_pred,axis=0)
                return cls_pro_test,bbox_pred_test,landmark_pred_test
            else:
                return cls_pro_test,bbox_pred_test

def R_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        #print_shape('RNet','input',inputs)
        #net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3,3], stride=1, scope="conv1")
        net = slim.conv2d(inputs, num_outputs=16, kernel_size=[3,3], stride=1, scope="conv1")
        print_shape('RNet','conv1',net)
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print_shape('RNet','pool1',net)
        #net = slim.conv2d(net,num_outputs=48,kernel_size=[3,3],stride=1,scope="conv2")
        net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope="conv2")
        print_shape('RNet','conv2',net)
        if config.rnet_wide:
            net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2",padding='SAME')
        else:
            net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2")
        print_shape('RNet','pool2',net)
        if config.rnet_wide:
            net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
            print_shape('RNet','conv3',net)
            net = slim.conv2d(net,num_outputs=128,kernel_size=[3,3],stride=1,scope="conv4")
            print_shape('RNet','conv4',net)
        else:
            net = slim.conv2d(net,num_outputs=64,kernel_size=[2,2],stride=1,scope="conv3")
            print_shape('RNet','conv3',net)
        fc_flatten = slim.flatten(net)
        print_shape('RNet','flatten',fc_flatten)
        if config.rnet_wide:
            fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1", activation_fn=prelu)
        else:
            fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1", activation_fn=prelu)
        print_shape('RNet','fc1',fc1)
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        print_shape('RNet','cls_fc',cls_prob)
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        print_shape('RNet','bbox_fc',bbox_pred)
        #batch*10
        if  test_fg :
            landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
            print_shape('RNet','landmark_fc',landmark_pred)
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            if  test_fg :
                landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)
            else:
                landmark_loss = 0
            #landmark_loss = 0
            #L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            if test_fg:
                return cls_prob,bbox_pred,landmark_pred
            else:
                return cls_prob,bbox_pred
            #return cls_prob,bbox_pred

def O_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        #print_shape('ONet','input',inputs)
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        print_shape('ONet','conv1',net)
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print_shape('ONet','pool1',net)
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
        print_shape('ONet','conv2',net)
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        print_shape('ONet','pool2',net)
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        print_shape('ONet','conv3',net)
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        print_shape('ONet','pool3',net)
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
        print_shape('ONet','conv4',net)
        fc_flatten = slim.flatten(net)
        print_shape('ONet','flatten',fc_flatten)
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1", activation_fn=prelu)
        print_shape('RNet','fc1',fc1)
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        print_shape('ONet','cls_fc',cls_prob)
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        print_shape('ONet','bbox_fc',bbox_pred)
        #batch*10
        if  test_fg:
            landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
            print_shape('RNet','landmark_fc',landmark_pred)
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            if  test_fg:
                landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            else:
                landmark_loss = 0
            #landmark_loss = 0
            #L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            if  test_fg:
                return cls_prob,bbox_pred,landmark_pred
            else:
                return cls_prob,bbox_pred


class FcnDetector(object):
    #net_factory: which net
    #model_path: where the params'file is
    def __init__(self, net_factory, model_path):
        #create a graph
        graph = tf.Graph()
        self.train_face = config.train_face
        with graph.as_default():
            #define tensor and op in graph(-1,1)
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            #self.cls_prob batch*2
            #self.bbox_pred batch*4
            #construct model here
            #self.cls_prob, self.bbox_pred = net_factory(image_reshape, training=False)
            #contains landmark
            if self.train_face:
                self.cls_prob, self.bbox_pred, _ = net_factory(image_reshape, training=False)
            else:
                self.cls_prob, self.bbox_pred = net_factory(image_reshape, training=False)
            #allow
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            net_name = model_path.split('/')[-1]
            print("net name is ",net_name)
            if self.train_face==100:
                logs_dir = "../logs/%s" %(net_name)
                summary_op = tf.summary.merge_all()
                if os.path.exists(logs_dir) == False:
                    os.mkdir(logs_dir)
                writer = tf.summary.FileWriter(logs_dir,self.sess.graph)
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print("restore model path",model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print ("restore models' param")
            saver.restore(self.sess, model_path)
            if self.train_face==100:
                saver.save(self.sess,model_dict+'/resaved/'+net_name+'relu')
            '''
            logs_dir = "../logs/%s" %(net_factory)
            summary_op = tf.summary.merge_all()
            if os.path.exists(logs_dir) == False:
                os.mkdir(logs_dir)
            writer = tf.summary.FileWriter(logs_dir,self.sess.graph)
            #summary = self.sess.run()
            #writer.add_summary(summary,global_step=step)
            '''
    def predict(self, databatch):
        height, width, _ = databatch.shape
        # print(height, width)
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                                           feed_dict={self.image_op: databatch, self.width_op: width,
                                                                      self.height_op: height})
        return cls_prob, bbox_pred


class Detector(object):
    #net_factory:rnet or onet
    #datasize:24 or 48
    def __init__(self, net_factory, data_size, batch_size, model_path):
        graph = tf.Graph()
        self.test_fg = 1
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, 3], name='input')
            #figure out landmark
            if self.test_fg:
                self.cls_prob, self.bbox_pred, self.landmark_pred = net_factory(self.image_op, training=False)
                #self.landmark_pred = tf.identity(self.landmark_pred,name='output')
            else:
                self.cls_prob, self.bbox_pred = net_factory(self.image_op, training=False)
                #self.cls_prob = tf.identity(self.cls_prob,name='cls_out')
                #self.bbox_pred = tf.identity(self.bbox_pred,name='bbox_out')
                #self.landmark_pred = tf.identity(self.landmark_pred,name='out')
                #self.output_op = tf.concat([self.cls_prob, self.bbox_pred], 1)
                #self.net_out = slim.flatten(self.output_op,scope='flatten_1')
                #self.out_put = tf.identity(self.net_out,name='output')

            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            net_name = model_path.split('/')[-1]
            print("net name is ",net_name)
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print ("model_dict is ",model_dict)
            readstate = ckpt and ckpt.model_checkpoint_path
            #assert  readstate, "the params dictionary is not valid"
            print ("restore models' param")
            saver.restore(self.sess, model_path)
            if self.test_fg==100:
                saver.save(self.sess,model_dict+'/resaved/'+net_name+'relu')
            #print_ckpt('./checkpoint')


        self.data_size = data_size
        self.batch_size = batch_size
    #rnet and onet minibatch(test)
    def predict(self, databatch):
        # access data
        # databatch: N x 3 x data_size x data_size
        scores = []
        batch_size = self.batch_size
        minibatch = []
        cur = 0
        #num of all_data
        n = databatch.shape[0]
        while cur < n:
            #split mini-batch
            minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size
        #every batch prediction result
        cls_prob_list = []
        bbox_pred_list = []
        landmark_pred_list = []
        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            #the last batch
            if m < batch_size:
                keep_inds = np.arange(m)
                #gap (difference)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
            #cls_prob batch*2
            #bbox_pred batch*4
            if self.test_fg:
                cls_prob, bbox_pred,landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred,self.landmark_pred], feed_dict={self.image_op: data})
                #num_batch * batch_size*10
                landmark_pred_list.append(landmark_pred[:real_size])
            else:
                cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred], feed_dict={self.image_op: data})
            #num_batch * batch_size *2
            cls_prob_list.append(cls_prob[:real_size])
            #num_batch * batch_size *4
            bbox_pred_list.append(bbox_pred[:real_size])
            #num_of_data*2,num_of_data*4,num_of_data*10
        if config.Debug:
            print("detect shape cls box landmark : ",np.shape(cls_prob_list),np.shape(bbox_pred_list),np.shape(landmark_pred_list))
        if self.test_fg:
            return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0)
        else:
            return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0)

class MtcnnDetector(object):
    def __init__(self,
                 detectors,
                 min_face_size=24,
                 stride=2,
                 threshold=[0.6, 0.6, 0.9],
                 scale_factor=0.79
                 ):

        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.train_face = config.train_face
        if self.train_face:
            self.nms_thresh = [0.4,0.4,0.4]
        else:
            self.nms_thresh = [0.5,0.6,0.6]
        self.scale_factor = scale_factor
        self.r_out = config.r_out

    def convert_to_square(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox
        Returns:
        -------
            square bbox
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox

    def convert_to_rect(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox
        Returns:
        -------
            square bbox
        """
        rect_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        h_n = np.maximum(np.maximum(h, w),2)
        w_n = h_n/2
        rect_bbox[:, 0] = bbox[:, 0] + w * 0.5 - w_n * 0.5
        rect_bbox[:, 1] = bbox[:, 1] + h * 0.5 - h_n * 0.5
        rect_bbox[:, 2] = rect_bbox[:, 0] + w_n - 1
        rect_bbox[:, 3] = rect_bbox[:, 1] + h_n - 1
        return rect_bbox

    def calibrate_box(self, bbox, reg,height,width):
        """
            calibrate bboxes
        Parameters:
        ----------
            bbox: numpy array, shape n x 5
                input bboxes
            reg:  numpy array, shape n x 4
                bboxes adjustment
        Returns:
        -------
            bboxes after refinement
        """
        if config.Debug:
            print("shape ",height,width)
        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        if config.Debug:
            print("x1 ",bbox_c[:,0])
            print("y1 ",bbox_c[:,1])
            print("x2 ",bbox_c[:,2])
            print("y2 ",bbox_c[:,3])
        keep = np.where(bbox_c[:,0] >0)
        bbox_c = bbox_c[keep]
        keep = np.where(bbox_c[:,1] >0)
        bbox_c = bbox_c[keep]
        keep = np.where(bbox_c[:,2] <width)
        bbox_c = bbox_c[keep]
        keep = np.where(bbox_c[:,3] <height)
        bbox_c = bbox_c[keep]
        keep = np.where(bbox_c[:,2] > bbox_c[:,0])
        bbox_c = bbox_c[keep]
        keep = np.where(bbox_c[:,3] > bbox_c[:,1])
        bbox_c = bbox_c[keep]
        return bbox_c

    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
            generate bbox from feature cls_map
        Parameters:
        ----------
            cls_map: numpy array , n x m
                detect score for each position
            reg: numpy array , n x m x 4
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        #stride = 4
        cellsize = 12
        #cellsize = 25

        t_index = np.where(cls_map > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        #offset
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])

        return boundingbox.T
    #pre-process images
    def processed_image(self, img, scale):
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
        img_resized = (img_resized - 127.5) / 128
        return img_resized

    def pad(self, bboxes, w, h):
        """
            pad the the bboxes, alse restrict the size of it
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        keep = np.where(bboxes[:,0]< w)
        bboxes = bboxes[keep]
        keep = np.where(bboxes[:,1]< h)
        bboxes = bboxes[keep]
        keep = np.where(bboxes[:,2] >0)
        bboxes = bboxes[keep]
        keep = np.where(bboxes[:,3] >0)
        bboxes = bboxes[keep]
        keep = np.where(bboxes[:,2] > bboxes[:,0])
        bboxes = bboxes[keep]
        keep = np.where(bboxes[:,3] > bboxes[:,1])
        bboxes = bboxes[keep]
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array
            input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        net_size = 12

        current_scale = float(net_size) / self.min_face_size  # find initial scale
        # print("current_scale", net_size, self.min_face_size, current_scale)
        im_resized = self.processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape
        # fcn
        all_boxes = list()
        while min(current_height, current_width) > net_size:
            #return the result predicted by pnet
            #cls_cls_map : H*w*2
            #reg: H*w*4
            cls_cls_map, reg = self.pnet_detector.predict(im_resized)
            #boxes: num*9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)
            #print("in MtCnnDetector pnet out shape ",cls_cls_map.shape, reg.shape)
            #cls_map = cls_cls_map[:,:,1]
            #print("scale, threshold ",current_scale,self.thresh[0])
            #boxes = self.generate_bbox(cls_map,reg,current_scale,self.thresh[0])
            boxes = self.generate_bbox(cls_cls_map[:,:,1], reg, current_scale, self.thresh[0])

            current_scale *= self.scale_factor
            im_resized = self.processed_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape
            if boxes.size == 0:
                continue
            keep = py_nms(boxes[:, :5], self.nms_thresh[0])
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return  None, None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], self.nms_thresh[0])
        all_boxes = all_boxes[keep]

        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # refine the boxes
        #print('pnet box ',np.shape(all_boxes))
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return  boxes_c,all_boxes
    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        height,width = h,w
        if self.train_face:
            dets = self.convert_to_square(dets)
        else:
            #dets = self.convert_to_rect(dets)
            dets = dets
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        #num_boxes = dets.shape[0]
        num_boxes = tmpw.shape[0]
        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
        if num_boxes <= 0:
            return None,None
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24))-127.5) / 128
        #cls_scores : num_data*2
        #reg: num_data*4
        #landmark: num_data*10
        if self.train_face:
            #cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)
            cls_scores, reg, landmark = self.rnet_detector.predict(cropped_ims)
        else:
            cls_scores, reg = self.rnet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:,1]
        keep_inds = np.where(cls_scores > self.thresh[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            if self.train_face:
                landmark = landmark[keep_inds]
        else:
            return  None, None
        if self.train_face:
            #width
            w = boxes[:,2] - boxes[:,0] + 1
            #height
            h = boxes[:,3] - boxes[:,1] + 1
            landmark[:,0::2] = (np.tile(w,(5,1)) * landmark[:,0::2].T + np.tile(boxes[:,0],(5,1)) - 1).T
            landmark[:,1::2] = (np.tile(h,(5,1)) * landmark[:,1::2].T + np.tile(boxes[:,1],(5,1)) - 1).T
        #"Minimum"
        if self.r_out:
            keep = py_nms(boxes, self.nms_thresh[1],"Minimum")
        else:
            keep = py_nms(boxes, self.nms_thresh[1],"Union")
        boxes = boxes[keep]
        boxes_c = self.calibrate_box(boxes, reg[keep],height,width)
        if self.train_face:
            landmark = landmark[keep]
            return boxes_c,landmark
        else:
            return  boxes_c,None
    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        height,width = h, w
        if self.train_face:
            dets = self.convert_to_square(dets)
        else:
            #dets = self.convert_to_rect(dets)
            dets = dets
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        real_box_num = 0
        for i in range(num_boxes):
            if tmph[i]<=1 or  tmpw[i]<=1:
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48))-127.5) / 128
            real_box_num+=1
        if real_box_num <=0:
            return  None, None
        if self.train_face:
            cls_scores, reg,landmark = self.onet_detector.predict(cropped_ims)
        else:
            cls_scores, reg = self.onet_detector.predict(cropped_ims)
        #prob belongs to face
        cls_scores = cls_scores[:,1]
        keep_inds = np.where(cls_scores > self.thresh[2])[0]
        if len(keep_inds) > 0:
            #pickout filtered box
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            if self.train_face:
                landmark = landmark[keep_inds]
        else:
            return  None, None

        #width
        w = boxes[:,2] - boxes[:,0] + 1
        #height
        h = boxes[:,3] - boxes[:,1] + 1
        if self.train_face:
            landmark[:,0::2] = (np.tile(w,(5,1)) * landmark[:,0::2].T + np.tile(boxes[:,0],(5,1)) - 1).T
            landmark[:,1::2] = (np.tile(h,(5,1)) * landmark[:,1::2].T + np.tile(boxes[:,1],(5,1)) - 1).T
        boxes_c = self.calibrate_box(boxes, reg,height,width)
        keep = py_nms(boxes_c,self.nms_thresh[2], "Minimum")
        boxes_c = boxes_c[keep]
        if self.train_face:
            landmark = landmark[keep]
            return boxes_c,landmark
        else:
            return boxes_c,None
    #use for video
    def detect(self, img):
        """Detect face over image
        """
        boxes = None
        t = time.time()

        # pnet
        t1 = 0
        if self.pnet_detector:
            boxes_c,all_box = self.detect_pnet(img)
            if boxes_c is None:
                return np.array([]),np.array([])

            t1 = time.time() - t
            t = time.time()
            #print("Pnet out ",boxes_c.shape)
            order_idx = np.argsort(boxes_c[:,4])[:-1]
            sel_num = config.P_Num if len(boxes_c) > config.P_Num else len(boxes_c)
            boxes_c = boxes_c[order_idx[:sel_num]]
            #print("Pnet out ",boxes_c.shape)
            boxes_p = boxes_c

        # rnet
        '''
        for i in range(10):
            print("box_c ",map(int,boxes_c[i]))
            print("box",map(int,all_box[i]))
        '''
        t2 = 0
        if self.rnet_detector:
            boxes_c,landmark_r = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([]),np.array([])
            t2 = time.time() - t
            t = time.time()
            bbox_r = boxes_c
        if self.r_out:
            print("time cost " + '{:.3f}'.format(t1 + t2) + '  pnet {:.3f}  rnet {:.3f} '.format(t1, t2))
            return bbox_r,landmark_r

        # onet
        t3 = 0
        if self.onet_detector:
            #boxes, boxes_c,landmark = self.detect_onet(img, boxes_c)
            if config.o_out:
                boxes_c,landmark = self.detect_onet(img, boxes_p)
            else:
                boxes_c,landmark = self.detect_onet(img, boxes_c)
            if boxes_c is None:
                return np.array([]),np.array([])

            t3 = time.time() - t
            t = time.time()
            #print( "time cost " + '{:.3f}'.format(t1 + t2 + t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2,t3))
        return boxes_c,landmark

    def detect_face(self, test_data):
        all_boxes = []#save each image's bboxes
        landmarks = []
        batch_idx = 0
        sum_time = 0
        #test_data is iter_
        data_num = test_data.size
        print("MtcnnDetect image num ",data_num)
        #for databatch in test_data:
        for i in range(data_num):
            databatch = test_data.next()
            #databatch(image returned)
            if batch_idx % 100 == 0:
                print("%d images done" % batch_idx)
            im = databatch
            # pnet
            t1 = 0
            if self.pnet_detector:
                t = time.time()
                #ignore landmark
                boxes_c, landmark = self.detect_pnet(im)
                t1 = time.time() - t
                sum_time += t1
                if boxes_c is None:
                    print("img path: ",test_data.img_path)
                    print("boxes_c is None...")
                    all_boxes.append(np.array([]))
                    #pay attention
                    landmarks.append(np.array([]))
                    batch_idx += 1
                    continue
                order_idx = np.argsort(boxes_c[:,4])[:-1]
                sel_num = config.P_Num if len(boxes_c) < config.P_Num else len(boxes_c)
                boxes_c = boxes_c[order_idx[:sel_num]]
            # rnet
            t2 = 0
            if self.rnet_detector:
                t = time.time()
                #ignore landmark
                boxes_c, landmark = self.detect_rnet(im, boxes_c)
                t2 = time.time() - t
                sum_time += t2
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    batch_idx += 1
                    continue
            # onet
            t3 = 0
            if self.onet_detector:
                t = time.time()
                boxes_c, landmark = self.detect_onet(im, boxes_c)
                t3 = time.time() - t
                sum_time += t3
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    batch_idx += 1
                    continue
                #print("time cost " + '{:.3f}'.format(sum_time) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2,t3))
            all_boxes.append(boxes_c)
            landmarks.append(landmark)
            batch_idx += 1
        #num_of_data*9,num_of_data*10
        return all_boxes,landmarks


# test demo
test_relu =config.train_face

def parameter():
    parser = argparse.ArgumentParser(description='Mtcnn camera test')
    parser.add_argument("--min_size",type=int,default=24,\
                        help='determin the image pyramid and the lest is 12')
    parser.add_argument("--threshold",type=float,default=[0.5,0.7,0.9],nargs="+",\
                        help='filter the proposals according to score')
    parser.add_argument("--slid_window",type=bool,default=False,\
                        help='if true Pnet will use slid_window to produce proposals')
    parser.add_argument('--batch_size',type=int,default=[1,256,32],nargs="+",\
                        help='determin the pnet rnet onet input batch_size')
    parser.add_argument('--epoch_load',type=int,default=[32,2700,25],nargs="+",\
                        help='load the saved paramters for pnet rnet onet')
    parser.add_argument('--file_in',type=str,default='None',\
                        help='input file')
    return parser.parse_args()

def load_model(epoch_load):
    if test_relu==5 or test_relu==100:
        if config.rnet_wide:
            #5,500,60;  5,1700,60
            prefix = ["../data/MTCNN_model/PNet_landmark/PNet", "../data/MTCNN_model/RNet_landmark/rnet_wide/RNet", "../data/MTCNN_model/ONet_landmark/ONet"]
        else:
            # 5,40,60
            prefix = ["../data/MTCNN_model/PNet_landmark/PNet", "../data/MTCNN_model/RNet_landmark/RNet", "../data/MTCNN_model/ONet_landmark/ONet"]
    else:
        #epoch_load = [32,30,25],[32,4400,25]
        #prefix = ["../data/MTCNN_model/PNet_landmark/v1_trained/PNet", "../data/MTCNN_model/RNet_landmark/v1_trained/RNet", "../data/MTCNN_model/ONet_landmark/v1_trained/ONet"]
        #[205,500,200]
        prefix = ["../../trained_models/MTCNN_bright_model/PNet_landmark/PNet", "../../trained_models/MTCNN_bright_model/RNet_landmark/RNet", "../../trained_models/MTCNN_bright_model/ONet_landmark/ONet"]
        #pedestrain [80,360,200],[580,4900,600],[1600,4500,600],[1600,2900,4000]
        #prefix = ["../data/MTCNN_caltech_model/PNet_landmark/PNet", "../data/MTCNN_caltech_model/RNet_landmark/RNet", "../data/MTCNN_caltech_model/ONet_landmark/ONet"]
        #person voc[1600,2900,300]
        #prefix = ["../data/MTCNN_voc_model/PNet_landmark/PNet", "../data/MTCNN_voc_model/RNet_landmark/RNet", "../data/MTCNN_voc_model/ONet_landmark/ONet"]
    print("demo epoch load ",epoch_load)
    model_path = ["%s-%s" %(x,y ) for x, y in zip(prefix,epoch_load)]
    print("demo model path ",model_path)
    return model_path

def process_img():
    param = parameter()
    min_size = param.min_size
    score_threshold = param.threshold
    slid_window = param.slid_window
    if test_relu==100:
        batch_size = [1,1,1]
    else:
        batch_size = param.batch_size
    epoch_load = param.epoch_load
    multi_detector = [None,None,None]
    #load paramter path
    model_path = load_model(epoch_load)
    #load net result
    if slid_window:
        print("using slid window")
        Pnet_det = None
        return [None,None,None]
    else:
        Pnet_det = FcnDetector(P_Net,model_path[0])
    Rnet_det = Detector(R_Net,data_size=24,batch_size=batch_size[1],model_path=model_path[1])
    Onet_det = Detector(O_Net,data_size=48,batch_size=batch_size[2],model_path=model_path[2])
    multi_detector = [Pnet_det,Rnet_det,Onet_det]
    #get bbox and landmark
    Mtcnn_detector = MtcnnDetector(multi_detector,min_size,threshold=score_threshold)
    #bboxs,bbox_clib,landmarks = Mtcnn_detector.detect(img)
    return Mtcnn_detector

def add_label(img,bbox,landmark):
    #print("labe ",bbox.shape)
    num = bbox.shape[0]
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_scale =1
    thickness = 1
    for i in range(num):
        x1,y1,x2,y2 = int(bbox[i,0]),int(bbox[i,1]),int(bbox[i,2]),int(bbox[i,3])
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
        score_label = str('{:.2f}'.format(bbox[i,4]))
        size = cv2.getTextSize(score_label, font, font_scale, thickness)[0]
        if y1-int(size[1]) <= 0:
            cv2.rectangle(img, (x1, y2), (x1 + int(size[0]), y2+int(size[1])), (255, 0, 0))
            cv2.putText(img, score_label, (x1,y2+size[1]), font, font_scale, (255, 255, 255), thickness)
        else:
            cv2.rectangle(img, (x1, y1-int(size[1])), (x1 + int(size[0]), y1), (255, 0, 0))
            cv2.putText(img, score_label, (x1,y1), font, font_scale, (255, 255, 255), thickness)
    if landmark is not None:
        for i in range(landmark.shape[0]):
            for j in range(5):
                #print(int(landmark[i][2*j]),int(landmark[i][2*j+1]))
                cv2.circle(img, (int(landmark[i][2*j]),int(landmark[i][2*j+1])), 2, (0,0,255))

def camera(file_in):
    cv2.namedWindow("result")
    cv2.moveWindow("result",1400,10)
    #camera_cap = cv2.VideoCapture('/home/lxy/Develop/Center_Loss/face_detect/videos/profile_video.wmv')
    if file_in =='None':
        camera_cap = cv2.VideoCapture(0)
    else:
        camera_cap = cv2.VideoCapture(file_in)
    if not camera_cap.isOpened():
        print("failded open camera")
        return -1
    mtcnn_dec = process_img()
    while camera_cap.isOpened():
        ret,frame = camera_cap.read()
        h,w,_ = frame.shape
        if ret:
            bbox_clib,landmarks = mtcnn_dec.detect(frame)
            print("landmark ",bbox_clib.shape)
            if len(bbox_clib):
                bbox_clib = board_img(bbox_clib,w,h)
                add_label(frame,bbox_clib,landmarks)
            if (cv2.waitKey(1)& (0xFF == ord('q'))):
                break
            cv2.imshow("result",frame)
        else:
            print("can not find device")
            break
    camera_cap.release()
    cv2.destroyAllWindows()

def demo_img(file_in):
    cv2.namedWindow("result")
    cv2.moveWindow("result",1400,10)
    if file_in =='None':
        cv2.destroyAllWindows()
        print("please input right path")
        return -1
    else:
        img = cv2.imread(file_in)
    mtcnn_dec = process_img()
    bbox_clib,landmarks = mtcnn_dec.detect(img)
    if len(bbox_clib):
        add_label(img,bbox_clib,landmarks)
        cv2.imshow("result",img)
        cv2.waitKey(0)

def board_img(boxes,wid,height):
    #print ('box shape ',np.shape(boxes))
    #print boxes
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3],
    offset_w = (x2-x1)/5.0
    offset_h = (y2-y1)/5.0
    x1 -= offset_w
    #y1 -= 4*offset_h
    x2 += offset_w
    y2 += offset_h
    x1 = map(int,np.maximum(x1,0))
    #y1 = map(int,np.maximum(y1,0))
    y1 = int(0)
    x2 = map(int,np.minimum(x2,wid-1))
    y2 = map(int,np.minimum(y2,height-1))
    box = [x1,y1,x2,y2,boxes[:,4]]
    #box = [x1,y1,x2,y2]
    box = np.asarray(box)
    #print("box shape",np.shape(box))
    box = np.vstack(box)
    return box.T

def GetFaces(file_in):
    '''
    param = parameter()
    min_size = param.min_size
    score_threshold = param.threshold
    slid_window = param.slid_window
    batch_size = param.batch_size
    epoch_load = param.epoch_load
    multi_detector = [None,None,None]
    '''
    if file_in =='None':
        #cv2.destroyAllWindows()
        print("please input right path")
        return []
    else:
        #img = cv2.imread(file_in)
        img = file_in
    h,w,_ = img.shape
    min_size = 24
    score_threshold = [0.5,0.7,0.9]
    slid_window = False
    batch_size = [1,256,16]
    #epoch_load = [205,500,200]
    epoch_load = [32,2700,25]
    multi_detector = [None,None,None]
    prefix = ["../../trained_models/MTCNN_bright_model/PNet_landmark/PNet", "../../trained_models/MTCNN_bright_model/RNet_landmark/RNet", "../../trained_models/MTCNN_bright_model/ONet_landmark/ONet"]
    print("demo epoch load ",epoch_load)
    model_path = ["%s-%s" %(x,y ) for x, y in zip(prefix,epoch_load)]
    #load net result
    if slid_window:
        print("using slid window")
        Pnet_det = None
        return [None,None,None]
    else:
        Pnet_det = FcnDetector(P_Net,model_path[0])
    Rnet_det = Detector(R_Net,data_size=24,batch_size=batch_size[1],model_path=model_path[1])
    Onet_det = Detector(O_Net,data_size=48,batch_size=batch_size[2],model_path=model_path[2])
    multi_detector = [Pnet_det,Rnet_det,Onet_det]
    #get bbox and landmark
    Mtcnn_detector = MtcnnDetector(multi_detector,min_size,threshold=score_threshold)
    #bboxs,bbox_clib,landmarks = Mtcnn_detector.detect(img)
    bbox_clib,landmarks = Mtcnn_detector.detect(img)
    if len(bbox_clib):
        bbox_clib =board_img(bbox_clib,w,h)
        #add_label(img,bbox_clib,landmarks)
        #cv2.imshow("result",img)
        #cv2.waitKey(0)
        #bbox_clib[:,2] = bbox_clib[:,2] - bbox_clib[:,0]
        #bbox_clib[:,3] = bbox_clib[:,3] - bbox_clib[:,1]
        #bbox_clib[:,0] = map(int,bbox_clib[:,0])
        #bbox_clib[:,1] = map(int,bbox_clib[:,1])
        #bbox_clib[:,2] = map(int,bbox_clib[:,2])
        #bbox_clib[:,3] = map(int,bbox_clib[:,3])
        #bbox_clib = bbox_clib[:,:4]
    else:
        bbox_clib= np.array([])
        landmarks = np.array([])
    return bbox_clib

if __name__ == '__main__':
    #process_img()
    arg = parameter()
    file_in = arg.file_in
    camera(file_in)
    #demo_img(file_in)
    #a = get_faces(file_in)
    #print(a)
