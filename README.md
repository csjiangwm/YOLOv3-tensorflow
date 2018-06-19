An YOLOv3 implention in tensorflow


If you want to test the model, you should:

1. First download the weights file using "wget https://pjreddie.com/media/files/yolov3.weights"

2. Put the yolov3.weights into your path and run "python convert --ckpt 1 --weights_path yourpath" to convert the weights into .ckpt
        For example, run "python convert.py --ckpt 1 --weights_path ~/darknet/yolov3.weights"
   Also, you can directly download the converted ckpt files from "" with extract code: ""

3. If your PC has GPUs, run "python test.py --ckpt_dir your_ckpt_dir --ckpt 1" for testing,
   otherwise, run "python test.py --ckpt_dir your_ckpt_dir --ckpt 1 --GPU_options 0"
        For example, run "python test.py --ckpt_dir experiments/ckpt --ckpt 1 --GPU_options 0"


If you want to train your own data, you should:

1. First download the weights file using "wget https://pjreddie.com/media/files/darknet53.conv.74"

2. Put the yolov3.weights into your path and run "python convert --cvt2npz 1 --weights_path yourpath" to convert the weights into .npz
        For example, run "python convert.py --cvt2npz 1 --weights_path ~/darknet/darknet53.conv.74"
   Also, you can directly download the converted npz file from "" with extract code: ""

3. Change the value of __C.trained_img_num to your total number of training images in configs/config.py and change __C.batch_size if necessary

4. If you want to train your model from darknet53.conv.74, use "python train.py --npz_path your_npz_path\
                                                                                --imgs_path your_data_path\
                                                                                --xml_path your_xml_path\
                                                                                --classes_path your_classes_file_path"
   If you want to train your model from your last checkpoint, use "python train.py --imgs_path your_data_path\
                                                                                   --xml_path your_xml_path\
                                                                                   --classes_path your_classes_file_path\
                                                                                   --restart 0"
   Note the parameter GPU_options should be also altered if necessary (whether your PC has GPUs or not)

5. Run "python test.py --ckpt 1 --ckpt_dir your_ckpt_dir" for testing
    For example, run "python test.py --ckpt 1 --ckpt_dir experiments/ckpt"


references:
https://pjreddie.com/media/files/papers/YOLOv3.pdf
https://pjreddie.com/darknet/yolo
https://github.com/qqwweee/keras-yolo3
https://github.com/Robinatp/YOLO_Tensorflow
https://github.com/raytroop/YOLOv3_tf
