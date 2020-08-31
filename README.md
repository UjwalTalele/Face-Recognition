# Face-Recognition

A face detection application written in python with OpenCV. 

# Prerequisites
After cloning the project , go to the project location in command prompt and install necessary software.
`python -m pip install requirements.txt`

# Run Project

### 1. Extract embedddigs from faces in dataset
In project folder run 
`python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7`
This command will extract 128-d feature vector from all images in dataset and store them in embeddings.pickle file in outputs

### 2. Train face fecognition model
This will train the svm classifier which will inturn recognize the unknown images in later stage.
`python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle`

### 3. Recognize Faces 
Now we are ready to recognize the faces.
`python recognize.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle --image *PATH TO IMAGE*`

### 4. Recognize in video
This command will open the webcam and perform simultaneous detection on the videofeed from the machine webcam.
`python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle` 

# Needed improvements:
 - A api for web deployment so that the embedding and model training can be done in background.Also should allow user to 
  -Add new faces
  -Delete current faces
  -Recognize faces with webcam in browser
 - The model currently predicts the similarity probability, need a logic such that the if probability is under certain threshold, the image will be classified as unknown
 
