# yolov5_tensorrt_python
网上搜索的很多yolov5的Tensorrt部署版本都是基于C++开发的，在此提供一个python版本的yolov5部署的，也是基于tensorrt环境的部署，亲测有效。

# 环境要求

CUDA 11.3

CUDNN 8.6

TensorRT-8.5.1.7.Windows10.x86_64.cuda-11.8.cudnn8.6

Opencv452

**注意：**Tensorrt下载上面要求的cudnn版本一定要与cuda下的cudnn一致，否则可能会报错

# 注意事项

（1）使用的onnx模型是三通道分别输出的模型，未将三个层concat在一起，使用build_engine函数生成engine

（2）程序中包含了等比例缩放、前向推理和坐标还原

（3）安装的tensorrt包为8.5.1.7，torch为 1.12.1+cu116，因为不同版本的trt库，生成engine代码不一样

测试结果如下图：

![](.\images\bus_show.jpg)



## 联系

**有问题可以提issues 或者加qq群:722237896 询问**

