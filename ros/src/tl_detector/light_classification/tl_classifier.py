from styx_msgs.msg import TrafficLight
import sys
import os
import tensorflow as tf
import numpy as np
import rospy
import time

import cv2
from PIL import Image

# trained model path
MODEL_PATH = os.path.join(os.getcwd(), '../../..', 'train_classifier')

# add tensorflow models path (https://github.com/tensorflow/models)
ROOT_PATH = os.path.join(os.getcwd(), '../../..')
sys.path.append(os.path.join(ROOT_PATH, 'models/research/'))
sys.path.append(os.path.join(ROOT_PATH, 'models/research/object_detection/utils'))

# import tensorflow models functions
#from label_map_util import load_labelmap, convert_label_map_to_categories, create_category_index
#from visualization_utils import visualize_boxes_and_labels_on_image_array

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
        #TODO load classifier
        model_path = os.path.join(MODEL_PATH, "frozen_inference_graph.pb")
        #label_path = os.path.join(MODEL_PATH, "train_data/label_map.pbtxt")
        NUM_CLASSES = 4
        
        #label_map = load_labelmap(label_path)
        #categories = convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
        #                                                            use_display_name=True)
        #self.category_index = create_category_index(categories)
        
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

        # Definite input and output Tensors for detection_graph
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
        #TODO implement light color prediction
        
        run_network = True

        # For timing single frame detection
        #start = time.time()
        
        def load_image_into_numpy_array(image):
            (im_width, im_height) = image.size
            return np.array(image).reshape(
                    (im_height, im_width, 3)).astype(np.uint8)

        if run_network is True:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

