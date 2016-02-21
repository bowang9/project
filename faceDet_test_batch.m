function det_list =faceDet_test_batch(imagelist)
config.paths.face_model_path = 'data/face_model.mat';

faceDet = lib.face_detector.dpmCascadeDetector(config.paths.face_model_path);
for i=1:size(imagelist,2)
    img = imread(imagelist{1,i});
    det_list{1,i} = faceDet.detect(img);
end
