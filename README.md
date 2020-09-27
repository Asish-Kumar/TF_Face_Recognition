Project Requirements:
===
1. All the library dependencies are mentioned in [**Requirements file**](requirements.txt) file.
Please be sure for their correct version installation.
2. Operating system : Windows 10 or ubuntu 18.04
3. Python 3.8 (remember to set path in system environment variables for windows)
4. pip for python
5. tensorflow [**Installation guide**](https://www.tensorflow.org/install/pip)
6. CUDA Core enabled GPU


How to run the project?
===
Open cmd or powershell or terminal (for ubuntu) in the project directory and follow the following guide. 
#### For training:
1. Put all the (minimum 5) face photos in the _Dataset/train/person_name_ folder. 
2. Where replace person_name with the actual name of the person whose face is in the photos. 
3. Only one face should be present in a photo during training.
4. Run the command `python train.py`.
5. After when training is complete you will see two new files _training_dataset.npz_ and _training_face_embaddings.npz_
6. Now you should remove the photos used for training from the _Dataset/train/person_name_ folder.

**[Note]** Training will not remove any duplicate face data,
        so in order to add new suspect to the dataset,
        first remove the folders containing already added suspects 
        and then follow above steps again (do not delete _training_dataset.npz_ and _training_face_embaddings.npz_ ),
        and inorder to add new faces for already added suspect,
        first remove previously added photos of the suspect and then run this file,
        while the training folders contain only new photos.

#### For executing the main program:
1. Train first if never trained before.
2. Make sure _training_dataset.npz_ and _training_face_embaddings.npz_ files are present.
3. We are using the built in laptop camera, if you are using another camera change the value from 0 to 1 or 2 or 3 or 4 etc. in line number 15 (`capture_video = cv2.VideoCapture(0)`) of `main.py` file. 
4. Uncomment line number 64 (`# cv2.imwrite(file_name, frame)`) in `main.py` file if you want the recognised faces to be stored in the _Recognised_faces_ folder.
5. Run the command `python main.py`
6. After a few seconds a window will appear which shows live face from the camera.
7. For every recognised face a box will appear around the face with person_name on it and confidence of prediction.
8. Press 'a' to exit.