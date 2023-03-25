import os
from flask import Flask, render_template, request
import cv2
import face_recognition
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app=Flask(__name__,static_folder='static')

# Load known face encodings from static folder
known_faces = []
known_names = []
for filename in os.listdir('static'):
    image = face_recognition.load_image_file(os.path.join('static', filename))
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)
    known_names.append(os.path.splitext(filename)[0])

# Initialize attendance DataFrame
attendance = pd.DataFrame(columns=['Name', 'Attendance'])

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST': 
	 imgg = request.files['image']
	 img_path = "static/" + imgg.filename	
	 imgg.save(img_path)
	 img=cv2.imread(img_path)
	 img=cv2.resize(img, (416,416))
	 label,bbox,confidence=functions.yolo(img_path)
	 print(label)
	#  try:
	# 	 os.remove(img_path)
	#  except:
	# 	 pass
	 #return jsonify({"label":label},{"bbox":bbox},{"confidence":confidence})
	 print(imgg.filename)
	 return render_template("index.html", Prediction = label, img_name=imgg.filename)
if __name__=='__main__':
    app.run(debug=True)
