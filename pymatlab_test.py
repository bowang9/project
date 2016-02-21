import sys
import matlab.engine
import time
import timeit 
import numpy as np 
from PIL import Image,ImageDraw

if __name__ == '__main__':
    if len(sys.argv)<2:
        print 'Usage:'
        print '  python %s  input_image_filename' % sys.argv[0]
        sys.exit(-1)
    eng = matlab.engine.start_matlab()

    #filename=sys.argv[1]
    #print filename

    #eng.demo(nargout=0)
    #eng.loadfaceDet(nargout=0)
    #net_path = 'data/vgg_face.mat';
    #face_model_path = 'data/face_model.mat';

    #faceDet = eng.lib.face_detector.dpmCascadeDetector(face_model_path);
    
    start_CPU = time.clock()
    #for i in range(5):
    filelist = []
    for i in range(len(sys.argv)-1):
        filelist.append(sys.argv[i+1])
    print filelist
    
    det_list = eng.faceDet_test_batch(filelist)
    end_CPU = time.clock()
    print("Face detection uses %f CPU seconds" % (end_CPU - start_CPU))
    
    print det_list
    a = np.array(det_list[0])
    a_T = a.T
    a_T_slice = a_T[:,0:4]
    b = np.array(det_list[1])
    b_T = b.T
    b_T_slice = b_T[:,0:4]
    print a,b
    print a_T,b_T
    print a_T_slice,b_T_slice
    
    list_tmp = []
    list_tmp.append(a_T_slice)
    list_tmp.append(b_T_slice)
   
    print list_tmp[0].shape,list_tmp[0][0][0],list_tmp[0][0][1],list_tmp[0][0][2]
    
