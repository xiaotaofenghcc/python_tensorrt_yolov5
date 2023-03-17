import cv2 
import sys
import os 
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import math
import time

class Processor():
    def __init__(self, model):
        # load tensorrt engine
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        TRTbin = model
        print('trtbin', TRTbin)
        with open(TRTbin, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        # allocate memory
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append({ 'host': host_mem, 'device': device_mem })
            else:
                outputs.append({ 'host': host_mem, 'device': device_mem })
        # save to class
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream
        # post processing config
        filters = (80 + 5) * 3
        self.output_shapes = [
            (1, 3, 80, 80, 85),
            (1, 3, 40, 40, 85),
            (1, 3, 20, 20, 85)
        ]
        self.strides = np.array([8., 16., 32.])
        anchors = np.array([
            [[10,13], [16,30], [33,23]],
            [[30,61], [62,45], [59,119]],
            [[116,90], [156,198], [373,326]],
        ])
        self.inpWidth = 640
        self.inpHeight = 640

        #坐标换算
        self.newh = 640
        self.neww = 640
        self.top = 0
        self.left = 0
        self.nl = len(anchors)
        self.nc = 80 # classes
        self.no = self.nc + 5 # outputs per anchor
        self.na = len(anchors[0])
        a = anchors.copy().astype(np.float32)
        a = a.reshape(self.nl, -1, 2)
        self.anchors = a.copy()
        self.anchor_grid = a.copy().reshape(self.nl, 1, -1, 1, 1, 2)

    def pre_process(self, img):
        print('original image shape', img.shape)
        img = cv2.resize(img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.transpose((2, 0, 1)).astype(np.float16)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        return img
    
    def detect(self,img):
        shape_orig_WH = (img.shape[1], img.shape[0])
        #预处理
        img,imgFloat, newh, neww, top, left = self.resize_image(img)
        self.newh = newh;self.neww = neww;self.top = top;self.left = left
        #img = self.pre_process(img)
        outputs = self.inference(imgFloat) #25200 * 85
        # reshape from flat to (1, 3, x, y, 85)
        reshaped = []
        for output, shape in zip(outputs, self.output_shapes):
            reshaped.append(output.reshape(shape))
        return reshaped
    
    def resize_image(self, srcimg, keep_ratio=True, dynamic=False):
        top, left, newh, neww = 0, 0, self.inpWidth, self.inpHeight
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                if not dynamic:
                    left = int((self.inpWidth - neww) * 0.5)
                    img = cv2.copyMakeBorder(img, 0, 0, left, self.inpWidth - neww - left, cv2.BORDER_CONSTANT,
                                             value=(114, 114, 114))  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                if not dynamic:
                    top = int((self.inpHeight - newh) * 0.5)
                    img = cv2.copyMakeBorder(img, top, self.inpHeight - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                             value=(114, 114, 114))
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        #BGR2RGB  Float类型
        imgFloat = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgFloat = imgFloat.transpose((2, 0, 1)).astype(np.float32)
        imgFloat /= 255.0
        return img,imgFloat, newh, neww, top, left

    def inference(self, img):
        # copy img to input memory
        # self.inputs[0]['host'] = np.ascontiguousarray(img)
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        start = time.time()
        self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle)
       # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        end = time.time()
        print('execution time:', end-start)
        return [out['host'] for out in self.outputs]

    def extract_object_grids(self, output):
        """
        Extract objectness grid 
        (how likely a box is to contain the center of a bounding box)
        Returns:
            object_grids: list of tensors (1, 3, nx, ny, 1)
        """
        object_grids = []
        for out in output:
            probs = self.sigmoid_v(out[..., 4:5])
            object_grids.append(probs)
        return object_grids

    def extract_class_grids(self, output):
        """
        Extracts class probabilities
        (the most likely class of a given tile)
        Returns:
            class_grids: array len 3 of tensors ( 1, 3, nx, ny, 80)
        """
        class_grids = []
        for out in output:
            object_probs = self.sigmoid_v(out[..., 4:5])
            class_probs = self.sigmoid_v(out[..., 5:])
            obj_class_probs = class_probs * object_probs
            class_grids.append(obj_class_probs)
        return class_grids

    def extract_boxes(self, output,nHei,nWid, newh, neww ,conf_thres=0.5):
        """
        Extracts boxes (xywh) -> (x1, y1, x2, y2)
        """
        ratioh, ratiow = nHei*1.0 / newh, nWid*1.0 / neww
        scaled = []
        grids = []
        for out in output:
            out = self.sigmoid_v(out)
            _, _, width, height, _ = out.shape
            grid = self.make_grid(width, height)
            grids.append(grid)
            scaled.append(out)
        z = []
        for out, grid, stride, anchor in zip(scaled, grids, self.strides, self.anchor_grid):
            _, _, width, height, _ = out.shape
            out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
            out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor
            # out[..., 0] =  out[..., 0] * ratioh
            # out[..., 1] =  out[..., 1] * ratiow
            # out[..., 2] =  out[..., 2] * ratioh
            # out[..., 3] =  out[..., 3] * ratiow
            out[..., 5:] = out[..., 4:5] * out[..., 5:]
            out = out.reshape((1, 3 * width * height, 85))
            z.append(out)
        pred = np.concatenate(z, 1)
        xc = pred[..., 4] > conf_thres
        pred = pred[xc]
        boxes = self.xywh2xyxy(pred[:, :4])
        return boxes

    def post_process(self, outputs,nOriHei,nOriWid, conf_thres=0.5):

        ratioW = nOriWid*1.0/self.neww
        ratioH = nOriHei*1.0/self.newh
        """
        Transforms raw output into boxes, confs, classes
        Applies NMS thresholding on bounding boxes and confs
        Parameters:
            output: raw output tensor
        Returns:
            boxes: x1,y1,x2,y2 tensor (dets, 4)
            confs: class * obj prob tensor (dets, 1) 
            classes: class type tensor (dets, 1)
        """
        scaled = []
        grids = []
        for out in outputs:
            out = self.sigmoid_v(out)
            _, _, width, height, _ = out.shape
            grid = self.make_grid(width, height)
            grids.append(grid)
            scaled.append(out)
        z = []
        for out, grid, stride, anchor in zip(scaled, grids, self.strides, self.anchor_grid):
            _, _, width, height, _ = out.shape
            out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
            out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor

            out[..., 0] = (out[..., 0] - self.left) * ratioW
            out[..., 1] = (out[..., 1] - self.top) * ratioH
            out[..., 2] = (out[..., 2] ) * ratioW
            out[..., 3] = (out[..., 3] ) * ratioH
            
            out = out.reshape((1, 3 * width * height, 85))
            z.append(out)
        pred = np.concatenate(z, 1)
        xc = pred[..., 4] > conf_thres
        pred = pred[xc]
        return self.nms(pred)
    
    def make_grid(self, nx, ny):
        """
        Create scaling tensor based on box location
        Source: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
        Arguments
            nx: x-axis num boxes
            ny: y-axis num boxes
        Returns
            grid: tensor of shape (1, 1, nx, ny, 80)
        """
        nx_vec = np.arange(nx)
        ny_vec = np.arange(ny)
        yv, xv = np.meshgrid(ny_vec, nx_vec)
        grid = np.stack((yv, xv), axis=2)
        grid = grid.reshape(1, 1, ny, nx, 2)
        return grid

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_v(self, array):
        return np.reciprocal(np.exp(-array) + 1.0)
    def exponential_v(self, array):
        return np.exp(array)
    
    def non_max_suppression(self, boxes, confs, classes, iou_thres=0.6):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
        order = confs.flatten().argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where( ovr <= iou_thres)[0]
            order = order[inds + 1]
        boxes = boxes[keep]
        confs = confs[keep]
        classes = classes[keep]
        return boxes, confs, classes

    def nms(self, pred, iou_thres=0.6):
        boxes = self.xywh2xyxy(pred[..., 0:4])
        # best class only
        confs = np.amax(pred[:, 5:], 1, keepdims=True)
        classes = np.argmax(pred[:, 5:], axis=-1)
        return self.non_max_suppression(boxes, confs, classes)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
