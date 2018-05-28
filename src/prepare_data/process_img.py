#author: lxy
#time: 2018/4/26 11:20
#tool: python3
#version: 0.1
#modify:
#project: age\gender\emotion 
######################################
import numpy as np 
import cv2
import sys
import os
import argparse
import string
from get_faces import MtcnnDetector,FcnDetector,Detector,P_Net,R_Net,O_Net,board_img,add_label

def args():
    parser = argparse.ArgumentParser(description="get face img")
    parser.add_argument('--saved_dir',type=str,default='../../datasets/Aidence_imgs/',\
                        help='images saved dir')
    parser.add_argument('--img_dir',type=str,default='/home/lxy/Downloads/DataSet/face_age/Adience/aligned/',\
                        help='images saved dir')
    parser.add_argument('--anno_file',type=str,default='./fga.txt',\
                        help='annotation files')
    parser.add_argument('--out_file',type=str,default='./train.txt',\
                        help='output annotation file')
    parser.add_argument('--img_size',type=int,default=64,\
                        help='saved img size')
    return parser.parse_args()

def LoadModel():
    min_size = 12
    score_threshold = [0.5,0.5,0.9]
    slid_window = False
    batch_size = [1,256,16]
    #epoch_load = [205,500,200]
    epoch_load = [32,30,25]
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
    return Mtcnn_detector

def GetFaces(file_in,Mtcnn_detector):
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
    #bboxs,bbox_clib,landmarks = Mtcnn_detector.detect(img)
    bbox_clib,landmarks = Mtcnn_detector.detect(img)
    if len(bbox_clib)==1:
        bbox_clib =board_img(bbox_clib,w,h)
    else:
        bbox_clib= np.array([])
        landmarks = np.array([])
    return bbox_clib

def LoadFile(file_path):
    file_r = open(file_path,'r')
    file_lines = file_r.readlines()
    img_path = []
    ages_label = []
    face_ids = []
    gender_labels = []
    cnt_total =0
    cnt_pass =0
    for line_1 in file_lines:
        cnt_total +=1
        line_split = line_1.strip().split(',')
        map(int,line_split[1:])
        #print(line_split[1:])
        if len(line_split) !=4:
            print("len is not 4",line_1)
            cnt_pass+=1
            continue
        if string.atoi(line_split[3]) == 2:
            print("gender laber is 2",line_1)
            cnt_pass+=1
            continue
        if string.atoi(line_split[2]) < 10:
            print("too young",line_1)
            cnt_pass+=1
            continue
        img_path.append(line_split[0])
        gender_labels.append(line_split[3])
        ages_label.append(line_split[2])
        face_ids.append(line_split[1])
    print("total, pass",cnt_total,cnt_pass)
    return dict(zip(img_path,gender_labels))

def SaveImg(img,bbox,img_path,img_size):
    x1,y1 = bbox[0],bbox[1]
    x2,y2 = bbox[2],bbox[3]
    x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
    img_roi = img[y1: (y2 +1),x1: (x2 +1),:]
    crop_img = cv2.resize(img_roi,(img_size,img_size))
    cur_dir = os.path.dirname(img_path)
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)
    cv2.imwrite(img_path,crop_img)

def main():
    param = args()
    img_dir = param.img_dir
    saved_dir = param.saved_dir
    anno_file = param.anno_file
    img_size = param.img_size
    out_file = param.out_file
    dict_img = LoadFile(anno_file)
    img_paths = dict_img.keys()
    anno_out_f = open(out_file,'w')
    total_num = len(img_paths)
    face_detector = LoadModel()
    cnt_pass = 0
    cnt_total = 0
    for i,one_path in enumerate(img_paths):
        cnt_total +=1
        img_path = os.path.join(img_dir,one_path)
        img_array = cv2.imread(img_path)
        sys.stdout.write('\r>> Converting image %d/%d\n' % (i + 1, total_num))
        sys.stdout.flush()
        if img_array is None:
            print("image is failed ",img_path)
            cnt_pass+=1
            continue
        img_array = cv2.resize(img_array,(204,204))
        print("img path", img_path)
        boxes = GetFaces(img_array,face_detector)
        #add_label(img_array,boxes,None)
        #cv2.imshow("show",img_array)
        #cv2.waitKey(50)
        if boxes.shape[0] !=1:
            print("detect more face or no face",boxes.shape[0], img_path)
            cnt_pass+=1
            continue
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        saved_path = os.path.join(saved_dir,one_path)
        SaveImg(img_array,boxes[0],saved_path,img_size)
        anno_out_f.write("{},{}\n".format(one_path,dict_img[one_path]))
    anno_out_f.close()
    print("total,pass ",cnt_total,cnt_pass)

if __name__=='__main__':
    main()