function det =faceDet_test(imagename)
config.paths.face_model_path = 'data/face_model.mat';

faceDet = lib.face_detector.dpmCascadeDetector(config.paths.face_model_path);

img = imread(imagename);
det = faceDet.detect(img);
