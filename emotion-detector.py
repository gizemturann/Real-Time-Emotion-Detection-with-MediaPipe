import mediapipe as mp
import cv2 
import csv
import os
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from PIL import ImageFont, ImageDraw, Image

def main():
	mpDraw = mp.solutions.drawing_utils
	mediapipeHolistic = mp.solutions.holistic
	
	with open('emotion-detection-eng.pkl', 'rb') as f:
		model = pickle.load(f)

	# press q to quit
	cap = cv2.VideoCapture(0)

	with mediapipeHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

		while cap.isOpened():
			ret, frame = cap.read()
			b, g, r, a = 0, 0, 0, 0

			image = np.zeros((640, 480), np.uint8)
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image.flags.writeable = False

			results = holistic.process(image)

			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			try:
				# Extract Pose landmarks
				coordinate_pose = results.pose_landmarks.landmark
				seqPose = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in
										 coordinate_pose]).flatten())

				# Extract Face landmarks
				coordinate_face = results.face_landmarks.landmark
				seqFace = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in
										 coordinate_face]).flatten())

				line = seqPose + seqFace
				X = pd.DataFrame([line])
				face_pose_class = model.predict(X)[0]

				font = ImageFont.truetype("arial.ttf", 50)
				PILImage = Image.fromarray(image)
				draw = ImageDraw.Draw(PILImage)
				draw.text((15, 10), face_pose_class, font=font, fill=(b, g, r, a))
				image = np.array(PILImage)

			except:
				pass

			cv2.imshow('Emotion Detection', image)

			if cv2.waitKey(10) & 0xFF == ord('q'):
				break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()