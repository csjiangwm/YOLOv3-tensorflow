A tensorflow implementation of YOLOv3.


#test<br>
If you want to test the model, you should:

1. First download the weights file using "wget https://pjreddie.com/media/files/yolov3.weights"<br>
2. Put the yolov3.weights into your path and run "python convert --ckpt 1 --weights_path yourpath" to convert the weights into .ckpt  <br>        ---For example, run
```Bash
python convert.py --ckpt 1 --weights_path ~/darknet/yolov3.weights
```
Also, you can directly download the converted ckpt files from "https://pan.baidu.com/s/1mBxcNwFZW-JEOZOiu73JfA" with extract code: "6d3s"<br>
3. If your PC has GPUs, run
```Bash
python test.py --ckpt_dir your_ckpt_dir --ckpt 1//Bash
```
for testing, otherwise, run
```Bash
python test.py --ckpt_dir your_ckpt_dir --ckpt 1 --GPU_options 0 #Bash
```
---For example, run
```Bash
python test.py --ckpt_dir experiments/ckpt --ckpt 1 --GPU_options 0 #Bash
```

##results:<br>
![](https://github.com/csjiangwm/YOLOv3-tensorflow/blob/master/prediction.jpg) 

#train<br>
If you want to train your own data, you should:

1. First download the weights file using 
```Bash
wget https://pjreddie.com/media/files/darknet53.conv.74 #Bash
``` 
<br>
2. Put the yolov3.weights into your path and run "python convert --cvt2npz 1 --weights_path yourpath" to convert the weights into .npz <br>        ---For example, run "python convert.py --cvt2npz 1 --weights_path ~/darknet/darknet53.conv.74" <br>    Also, you can directly download the converted npz file from "https://pan.baidu.com/s/13u38HIclp0iPoP1JVmZYsQ" with extract code: "swyf" <br>
3. Change the value of __C.trained_img_num to your total number of training images in configs/config.py and change __C.batch_size if necessary <br>
4. If you want to train your model from darknet53.conv.74, use <br>
```Bash
python train.py --npz_path your_npz_path\
                --imgs_path your_data_path\
		--xml_path your_xml_path\
		--classes_path your_classes_file_path  #Bash
```
										
   If you want to train your model from your last checkpoint, use <br>
```Bash
python train.py --imgs_path your_data_path\
		--xml_path your_xml_path\
		--classes_path your_classes_file_path\
		--restart 0  #Bash
```
   Note the parameter GPU_options should be also altered if necessary (whether your PC has GPUs or not) <br>
5. Run
```Bash
python test.py --ckpt 1 --ckpt_dir your_ckpt_direxperiments/ckpt #Bash
```
 for testing <br>        ---For example, run
```Bash
python test.py --ckpt 1 --ckpt_dir #Bash
```

references:<br>
https://pjreddie.com/media/files/papers/YOLOv3.pdf <br>
https://pjreddie.com/darknet/yolo <br>
https://github.com/qqwweee/keras-yolo3 <br>
https://github.com/raytroop/YOLOv3_tf <br>
https://github.com/Robinatp/YOLO_Tensorflow
