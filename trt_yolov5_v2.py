import torch
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
from Processor_v2 import Processor

# 加载YOLOv5模型
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()

# 定义TensorRT引擎生成器
#TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = []
print('trt version', trt.__version__)
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
#onnx 转 trt
trt_engine_path = "./weights/yolov5s-simple.trt"
def build_engine(model_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        #builder.max_workspace_size = 1 << 30 # 1GB
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        # with open(model_path, 'rb') as model_file:
        #     parser.parse(model_file.read())
        with open(model_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        #engine = builder.build_cuda_engine(network)
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.add_optimization_profile(profile)
        engine = builder.build_engine(network, config)
        with open(trt_engine_path, "wb") as f:
            f.write(engine.serialize())
        return engine
    
# 读取trt模型
def get_engine():
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    # TRTbin = '{0}/models/{1}'.format(os.path.dirname(__file__), model_path)
    # print('trtbin', TRTbin)
    with open(trt_engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine 

# 生成TensorRT引擎
#engine = build_engine('./weights/yolov5s-simple.onnx')
#engine = get_engine()

processor = Processor(model=trt_engine_path)

img = cv2.imread('./images/bus.jpg')
# inference
nHei = img.shape[0]
nWid = img.shape[1]
output = processor.detect(img) 

imgShow = img.copy()
boxes, confs, classes = processor.post_process(output, nHei,nWid,conf_thres=0.5)
for box, conf, cls in zip(boxes, confs, classes):
    cv2.rectangle(imgShow, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255),thickness=1)
    cv2.putText(imgShow, '%s %f' % (cls, conf), org=(int(box[0]), int(box[1]+10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255))


cv2.imwrite('./images/bus_show.jpg',imgShow)
k = 1






