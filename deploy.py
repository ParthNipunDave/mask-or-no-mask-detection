from flask import Flask, render_template, request,send_from_directory,send_file

from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

app=Flask(__name__)
app.config['UPLOAD_FOLDER']="image/input/"
@app.route("/")
def index():
	return render_template("index.html")
@app.route("/upload",methods=["POST"])
def upload():
	if request.method=="POST":
		f=request.files['image']
		fn=secure_filename(f.filename)
		f.save("static/images/input/"+f.filename)
	image="static/images/input/"+str(fn)
	return "sucess"

@app.route("/object_detection",methods=["POST"])
def object_dectection():
	if request.method=="POST":
		f=request.files['image']
		fn=secure_filename(f.filename)
		f.save("static/image/input/"+f.filename)
	image="static/image/input/"+str(fn)
	MODEL_NAME = 'faster_masks'
	CWD_PATH = os.getcwd()

	# Path to frozen detection graph .pb file, which contains the model that is used
	# for object detection.
	PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

	# Path to label map file
	PATH_TO_LABELS = os.path.join(CWD_PATH,'label.pbtxt')

	# Path to image
	PATH_TO_IMAGE = os.path.join(CWD_PATH,image)

	# Number of classes the object detector can identify
	NUM_CLASSES = 2
	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	# Load the Tensorflow model into memory.
	detection_graph = tf.Graph()
	with detection_graph.as_default():
	    od_graph_def = tf.GraphDef()
	    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
	        serialized_graph = fid.read()
	        od_graph_def.ParseFromString(serialized_graph)
	        tf.import_graph_def(od_graph_def, name='')

	    sess = tf.Session(graph=detection_graph)

	# Define input and output tensors (i.e. data) for the object detection classifier

	# Input tensor is the image
	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

	# Output tensors are the detection boxes, scores, and classes
	# Each box represents a part of the image where a particular object was detected
	detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

	# Each score represents level of confidence for each of the objects.
	# The score is shown on the result image, together with the class label.
	detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
	detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

	# Number of objects detected
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

	# Load image using OpenCV and
	# expand image dimensions to have shape: [1, None, None, 3]
	# i.e. a single-column array, where each item in the column has the pixel RGB value

	image = cv2.imread(PATH_TO_IMAGE)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image_expanded = np.expand_dims(image_rgb, axis=0)

	# Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
	[detection_boxes, detection_scores, detection_classes, num_detections],
	feed_dict={image_tensor: image_expanded})

	# Draw the results of the detection (aka 'visulaize the results')

	vis_util.visualize_boxes_and_labels_on_image_array(
	image,
	np.squeeze(boxes),
	np.squeeze(classes).astype(np.int32),
	np.squeeze(scores),
	category_index,
	use_normalized_coordinates=True,
	line_thickness=2,
	min_score_thresh=0.60)

	image=Image.fromarray(image)
	image.save("static/image/output/temp.png")
	filepath="static/image/output/temp.png"	
	return send_file('static/image/output/temp.png',as_attachment=True)

@app.route("/detected_image")
def detected_image():
	return send_file("static/image/output/temp.png")
if __name__=="__main__":
	app.run(debug=True,use_reloader=False)
