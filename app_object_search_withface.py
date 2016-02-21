import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
import cStringIO as StringIO
import urllib
import exifutil
import commands
import math
import matplotlib.pyplot as plt

caffe_root = '/home/user/softwares/caffe-master/'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.insert(0,caffe_root+'python')
import caffe

# import matlab.engine
import timeit 


REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/home/user/project/210_DH_project/tmp_images/object_search_uploads_src'
UPLOAD_FOLDER_IMAGESET = '/home/user/project/210_DH_project/tmp_images/object_search_uploads_gallery'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])
ALLOWED_UPLOAD_FILE_EXTENSIONS = set(['zip'])

# Make sure that caffe is on the python path:
caffe_root = '/home/user/softwares/caffe-master/'

max_image_num = 2000
feature_dim = 1000
class_dim = 1000

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index_object_searching.html', has_result=False)

@app.route('/team-member')
def show_team():
    photos = []
    
    filename1 = './templates/team-members-photo/wangbo.jpg'
    image1 = exifutil.open_oriented_im(filename1)
    member1=embed_image_html(image1)
    photos.append(member1)
        
    filename2 = './templates/team-members-photo/zdh.jpg'
    image2 = exifutil.open_oriented_im(filename2)
    member2=embed_image_html(image2)
    photos.append(member2)    
    
    filename3 = './templates/team-members-photo/zhuhao.jpg'
    image3 = exifutil.open_oriented_im(filename3)
    member3=embed_image_html(image3)
    photos.append(member3)
    
    return flask.render_template('index_team.html', has_result=False,imagesrc=photos)



@app.route('/image_set_upload', methods=['POST'])
def image_set_upload():
    batchsize = 64
    app.is_gallery_feature_extract = False
    try:
        # We will save the file to disk for possible data collection.
        logging.info('Come into image_set_upload')
        compress_file = flask.request.files['imageset_file']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(compress_file.filename)
        logging.info('filename_: %s.', filename_)
        filename = os.path.join(UPLOAD_FOLDER_IMAGESET, filename_)
        logging.info('Saving to %s.', filename)
        #image = exifutil.open_oriented_im(filename)
        command = 'rm ' + UPLOAD_FOLDER_IMAGESET + '/* -fr'
        (status, output) = commands.getstatusoutput(command)
        compress_file.save(filename)
        command = 'unzip ' + filename +  ' -d  ' + UPLOAD_FOLDER_IMAGESET + '/'
        (status, output) = commands.getstatusoutput(command)
        
        #delete  the *.zip file
        command = 'rm ' +   filename 
        (status, output) = commands.getstatusoutput(command)
                
        #delete invalid image file
        files = os.listdir(UPLOAD_FOLDER_IMAGESET) 
        total_image_num = len(files)
        invalid_image_num = 0
        imagefile_list = []
        app.gallery_image_name = []
        for f in files:
            if not allowed_file(f):
                invalid_image_num += 1
                command = 'rm ' +  UPLOAD_FOLDER_IMAGESET + '/' + f 
                (status, output) = commands.getstatusoutput(command)
            else:
                imagefile_list.append(UPLOAD_FOLDER_IMAGESET + '/' + f )   
                        
        app.upload_gallery = 1
        
        print  'Upload imagefile_list:',imagefile_list
        app.gallery_image_name.extend(imagefile_list)
        
        starttime = time.time()
        tmp_features = image_feature_extract(imagefile_list,len(imagefile_list),batchsize)
        
        if len(imagefile_list) > max_image_num:
            feat_num = max_image_num
        else:
            feat_num = len(imagefile_list)
        app.gallery_features[0:feat_num,:] = tmp_features[0:feat_num,:]
        app.gallery_image_num = feat_num
        
        app.is_gallery_feature_extract = True
        
        endtime = time.time()
        
        logging.info('Feaure rxtract for upload %d images, runtime %d',feat_num, endtime-starttime)
        return flask.render_template(
            'index_object_searching.html', has_result=True,
            result=[-1, 'Uploaded imageset successfully:%d valid images and %d invalid images.'%(total_image_num-invalid_image_num,invalid_image_num)]
        )
    except Exception as err:
        logging.info('Uploaded image or image handle error: %s', err)
        return flask.render_template(
            'index_object_searching.html', has_result=True,
            result=[-1, 'Cannot open uploaded image.']
        )

@app.route('/image_search_url', methods=['GET'])
def image_search_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        # string_buffer = StringIO.StringIO(
        #    urllib.urlopen(imageurl).read())
        #image_filename = imageurl.split('/')[-1]  
        image_filename = str(datetime.datetime.now()).replace(' ', '_')+'.jpg'
        filename = os.path.join(UPLOAD_FOLDER, image_filename)
        data = urllib.urlopen(imageurl).read()  

        f = file(filename,"wb")  
        f.write(data)  
        f.close()  


        #image = caffe.io.load_image(string_buffer)
        #filename_ = str(datetime.datetime.now()).replace(' ', '_')+'.jpg'
        #filename = os.path.join(UPLOAD_FOLDER, filename_)
        #image.save(filename)
        #logging.info('Saving to %s.', filename)
    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index_object_searching.html', has_result=True,
            result=[-1, 'Cannot open image from URL.']
        )

    logging.info('Image: %s', imageurl)
    
    try:
        starttime = time.time()
        
        image = exifutil.open_oriented_im(filename)
        
        tmp_list = [filename]
        target_feature = image_feature_extract(tmp_list,1,1)
        
        L2_distance_feature = []
        res_images = []
        if app.is_gallery_feature_extract == True:
            logging.info('Begin to caculate L2_distance and sort.')
            for i in range(app.gallery_image_num):
                #caculate the L2 distance for f8 feature 
                tmp_distance = Euclidean_distance(app.gallery_features[i],target_feature[0])
                L2_distance_feature.append(tmp_distance)
            print "L2_distance_feature:",L2_distance_feature
            index = np.argsort(np.array(L2_distance_feature))
            
            for i in index:
                if L2_distance_feature[i] < app.threshold:
                    tmp_image = exifutil.open_oriented_im(app.gallery_image_name[i])
                    print 'Prepare output image, index: %d, distance:%d, name : %s \n'%(i,L2_distance_feature[i],app.gallery_image_name[i])
                    res_images.append(embed_image_html(tmp_image))
                else:
                    break
            endtime = time.time()
            logging.info('Finish searching, output %d similar images.',len(res_images)) 
            return flask.render_template(
        'index_object_searching.html', has_result=True, result=[len(res_images),'%.3f' % (endtime - starttime)],result_images=res_images,imagesrc=embed_image_html(image) )

        else:
            endtime = time.time()
            return flask.render_template(
        'index_object_searching.html', has_result=True, result=[0,'%.3f' % (endtime - starttime)],result_images=res_images,imagesrc=embed_image_html(image) )
        
    except Exception as err:
            logging.info('Image searching error: %s', err)
            return flask.render_template(
            'index_object_searching.html', has_result=True,
            result=[-1, 'Something went wrong when searching image. '] )
     


@app.route('/image_search_upload', methods=['POST'])
def image_search_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        #image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index_object_searching.html', has_result=True,
            result=[-1, 'Cannot open uploaded image.']
        )
    try:
        starttime = time.time()        
        image = exifutil.open_oriented_im(filename)        
        tmp_list = [filename]
        target_feature = image_feature_extract(tmp_list,1,1)
        
        
        L2_distance_feature = []
        res_images = []
        if app.is_gallery_feature_extract == True:
            logging.info('Begin to caculate L2_distance and sort.')
            for i in range(app.gallery_image_num):
                #caculate the L2 distance for f8 feature 
                tmp_distance = Euclidean_distance(app.gallery_features[i],target_feature[0])
                L2_distance_feature.append(tmp_distance)
            print "L2_distance_feature:",L2_distance_feature
            index = np.argsort(np.array(L2_distance_feature))                               
            for i in index:
                if L2_distance_feature[i] < app.threshold:
                    print 'Prepare output image, index: %d, distance:%d, name : %s \n'%(i,L2_distance_feature[i],app.gallery_image_name[i])
                    tmp_image = exifutil.open_oriented_im(app.gallery_image_name[i])
                    
                    res_images.append(embed_image_html(tmp_image))
                else:
                    break
            endtime = time.time()
            logging.info('Finish searching, output %d similar images.',len(res_images)) 
            return flask.render_template(
        'index_object_searching.html', has_result=True, result=[len(res_images),'%.3f' % (endtime - starttime)],result_images=res_images,imagesrc=embed_image_html(image) )

        else:
            endtime = time.time()
            return flask.render_template(
        'index_object_searching.html', has_result=True, result=[0,'%.3f' % (endtime - starttime)],result_images=res_images,imagesrc=embed_image_html(image) )
        
    except Exception as err:
            logging.info('Image searching error: %s', err)
            return flask.render_template(
            'index_object_searching.html', has_result=True,
            result=[-1, 'Something went wrong when searching image. '] )
                              

def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )
    

def image_feature_extract(imagefile_list,input_image_num,batchsize):
    
    logging.info('Come into image_feature_extract, %d images input.',input_image_num)    
    
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]  
    # Now, the net.blobs['data'].data.shape is (10,3,227,227), which is read from 'models/bvlc_reference_caffenet/deploy.prototxt'
    transformer = caffe.io.Transformer({'data': app.imagenet.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    print np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1).shape
    
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        

    logging.info('Come into image_feature_extract, image num:%d,  batch size : %d .',input_image_num,batchsize)
    app.imagenet.blobs['data'].reshape(batchsize,3,227,227)
        
    #The return of transformer.preprocess is (K x H x W) ndarray for input to a Net. The preprocess() function will resize input image to (227,227) 
    #Now, net.blobs['data'].data.shape is (50,3,227,227)
    #net.blobs['data'].data[0,:,:,:] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/dog.jpg'))
    #net.blobs['data'].data[1,:,:,:] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/dog1.jpg'))
    
    feature_output = np.zeros((input_image_num,feature_dim),dtype=float)
    #class_output = np.zeros((input_image_num,class_dim),dtype=float)
    
    residue = input_image_num%batchsize
    batch_num = input_image_num/batchsize 
           
    for i in range(batch_num):
        starttime = time.time()
        
        baseindex = i*batchsize
        for j in range(batchsize):
            tmp_iter = baseindex+j
            app.imagenet.blobs['data'].data[j,:,:,:] = transformer.preprocess('data', caffe.io.load_image(imagefile_list[tmp_iter]))
        
        time1 = time.time()
        logging.info('Caffe.io.load_image and preprocess time:%d,batchsize: %d',time1-starttime,batchsize ) 
        
        out = app.imagenet.forward()                
        
        time2 = time.time()
        logging.info('app.imagenet.forward time:%d,batchsize: %d',time2-time1,batchsize )             
        
        for j in range(batchsize):
            tmp_iter = baseindex+j
            feature_output[tmp_iter] = app.imagenet.blobs['fc8'].data[j].flatten()

                #print "\nFor image",input_image[i]," :extract feature of layer fc8. "        
                
                #tmp_class = app.imagenet.blobs['prob'].data[j].flatten()
                #class_output[tmp_iter] = tmp_class
                
                #top_k = tmp_class.argsort()[-1:-6:-1]
                #print  "For image",input_image[i]," :extract feature of layer softmax."
                #print "The top_k class index is:",top_k
                #print  "The prob is:",tmp_class[top_k]
                #print("Predicted class is #{}.".format(out['prob'][0].argmax()))
        
        time3 = time.time()
        logging.info('Extract features time:%d,batchsize: %d',time3-time2,batchsize )             
        
        feat = app.imagenet.blobs['fc8'].data[0].flatten()
        logging.info('Batch:%d/%d, The output of layer fc8 len is:%d',i,batch_num,len(feat) )
        #tmp_class = app.imagenet.blobs['prob'].data[0].flatten()
        #logging.info('The output of layer prob len is:%d', len(tmp_class) )    
        
    if residue != 0:
        baseindex = batch_num*batchsize
        for j in range(residue):
            tmp_iter = baseindex+j
            app.imagenet.blobs['data'].data[j,:,:,:] = transformer.preprocess('data', caffe.io.load_image(imagefile_list[tmp_iter]))
        out = app.imagenet.forward()
        for j in range(residue):
            tmp_iter = baseindex+j
            feature_output[tmp_iter] = app.imagenet.blobs['fc8'].data[j].flatten()                        
    return feature_output


def Euclidean_distance(a,b):
    if a.shape !=  b.shape :
        print "a.shape !=  b.shape"
        return
    c = a-b
    d = np.dot(c,c)
    return math.sqrt(d)
    

def start_tornado(app, port=10005):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=10005)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    #ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    #app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    #app.clf.net.forward()
    #app.eng = matlab.engine.start_matlab()
                    
    caffe.set_mode_gpu()
    caffe.set_device(0)
    app.imagenet = caffe.Net('/home/user/softwares/caffe-master/models/bvlc_reference_caffenet/deploy.prototxt',
                    '/home/user/project/200_caffe_cudnnV4/caffe-caffe-0.14/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    caffe.TEST)
                    
    app.imagenet.forward()
    
    app.is_gallery_feature_extract = False    
    #app.eng = matlab.engine.start_matlab()   
    app.is_new_upload_gallery = True
    app.upload_gallery = 0
    
    app.gallery_features = np.zeros((max_image_num,feature_dim),dtype=float)
    app.gallery_image_num = 0
    app.gallery_image_name = []
    #For image not including face, the threshold should set to 60~70. But for image of face, this value should be smaller, may be 50
    app.threshold = 66
    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
