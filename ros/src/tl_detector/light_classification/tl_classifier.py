import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json, load_model
import numpy as np
from PIL import Image

from styx_msgs.msg import TrafficLight
import sys
import os
import tensorflow as tf
import rospy
import time

import cv2

# trained model path
MODEL_PATH = os.path.join(os.getcwd(), '../../..', 'train_classifier')


# add tensorflow models path (https://github.com/tensorflow/models)
ROOT_PATH = os.path.join(os.getcwd(), '../../..')
sys.path.append(os.path.join(ROOT_PATH, 'models/research/'))
sys.path.append(os.path.join(ROOT_PATH, 'models/research/object_detection/utils'))


CLASS_TO_TRAFFIC_LIGHT = {
    2: TrafficLight.RED,
    3: TrafficLight.YELLOW,
    1: TrafficLight.GREEN,
    4: TrafficLight.UNKNOWN
}


##################################
# Traffic Light Classifier Class #
##################################
class TLClassifier(object):

    def __init__(self):
        # load classifier
        model_path = os.path.join(MODEL_PATH, "frozen_inference_graph.pb")
        NUM_CLASSES = 4
        
        #### Build network
        self.image_np_deep = None
        self.detection_graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)

        # Define input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        
        run_network = True

        # For timing single frame detection
        #start = time.time()
        
        def load_image_into_numpy_array(image):
            (im_width, im_height) = image.size
            return np.array(image).reshape(
                    (im_height, im_width, 3)).astype(np.uint8)

        if run_network is True:
            # We are now capturing in rgb so no need to color convert
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Actual detection.
            with self.detection_graph.as_default():
                (boxes, scores, classes, num) = self.sess.run(
                    [self.detection_boxes, self.detection_scores,
                     self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: image_np_expanded})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            min_score_thresh = .50
            light = TrafficLight.UNKNOWN

            for i in range(boxes.shape[0]):
                if scores[i] > min_score_thresh:
                        c_l = CLASS_TO_TRAFFIC_LIGHT[classes[i]]
                        if c_l < light:
                            light = c_l
            return light


class TLKeras(object):
    def __init__(self):
        # Normally, you'd think that we can just load a keras model here and
        # use it elsewhere, but you'd be wrong when running in ROS.
        # This is a most bizarre bug... it turns out because of the threading of 
        # the process that runs the tl detector node, if we read in the model here
        # and just try to use it, it will error out in the get_classification method
        # saying that it cannot find the final softmax layer.
        # So we have to do the extra steps below, including calling an internal
        # method, to make it work.  See
        # https://github.com/keras-team/keras/issues/2397#issuecomment-338659190
        model_path = os.path.join(MODEL_PATH, "full_model.hd5")
        self.model = load_model(model_path)
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # First resize the image to something smaller
        x = np.array(cv2.resize(image, (400, 300), cv2.INTER_AREA), dtype=np.float64)
        x = np.expand_dims(x, axis=0) / 255.
        
        with self.graph.as_default():
            preds = self.model.predict(x)

        min_score_thresh = 0.50
        light = TrafficLight.UNKNOWN

        for (i, prob) in enumerate(preds[0]):
            if prob > min_score_thresh:
                c_l = CLASS_TO_TRAFFIC_LIGHT[i+1]
                if c_l < light:
                    light = c_l
        
        return light
        
