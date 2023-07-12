# Real-Time Image Based Attendance System

This is a face recognition project that utilizes the MTCNN (Multi-task Cascaded Convolutional Networks) algorithm for face detection and the SVM (Support Vector Machines) classifier for face recognition. It also incorporates the FaceNet model for generating face embeddings.

## Installation

1. Clone the repository or download the code files.
2. Make sure you have the following dependencies installed:
   - TensorFlow
   - PIL
   - numpy
   - mtcnn
   - keras-facenet
   - scikit-learn
   - OpenCV
   - pickle5
3. Create a folder named "SVM_MTCNN" in your local directory.
4. Place the code files inside the "SVM_MTCNN" folder.

## Usage

### Training the Face Recognition Model

1. Create a folder named "database" inside the "SVM_MTCNN" folder.
2. Inside the "database" folder, create subfolders for each person and place their respective face images inside those folders. Each person's folder should contain multiple face images for training.
   - Example structure:
     ```
     SVM_MTCNN/
     ├── database/
     │   ├── person1/
     │   │   ├── image1.jpg
     │   │   ├── image2.jpg
     │   │   └── ...
     │   ├── person2/
     │   │   ├── image1.jpg
     │   │   ├── image2.jpg
     │   │   └── ...
     │   └── ...
     └── ...
     ```
3. Open a terminal or command prompt and navigate to the directory containing the "verify.py" file.
4. Run the command `python verify.py` to train the face recognition model.
5. The training process will load the face images, detect faces using MTCNN, generate embeddings using FaceNet, and train an SVM classifier. The trained model will be saved in a file named "model.pkl".
6. The training data and embeddings will be compressed and saved in a file named "data.npz" for future use.

### Face Classification

1. Run the command `python fclassification.py` to start the face classification script.
2. The script will access the webcam and start capturing frames.
3. It will detect faces in the frames using MTCNN and generate embeddings using FaceNet.
4. The SVM model will be loaded from the "model.pkl" file.
5. The script will classify the faces based on the embeddings using the SVM model.
6. If a face is recognized, the person's name and classification probability will be displayed on the frame.
7. If a face is not recognized, the frame will display "face not recognized".
8. Press 'q' to exit the classification.

### Capturing Images for Training

1. Run the command `python cv.py` to capture and save face images for the training dataset.
2. Follow the prompts to enter the name of the person for whom you want to capture images.
3. The webcam will start capturing frames and saving images in the corresponding folder inside the "database" directory.
4. Press 'q' to stop capturing images.

## Modules and libraries used

- [MTCNN](https://github.com/ipazc/mtcnn) - Multi-task Cascaded Convolutional Networks for face detection.
- [keras-facenet](https://github.com/nyoki-mtl/keras-facenet) - Pretrained FaceNet model for face recognition.
- [scikit-learn](https://scikit-learn.org/) - Machine learning library for SVM classifier.
- [OpenCV](https://opencv.org/) - Computer vision library for webcam access and image manipulation.
- [pickle5](https://github.com/marcusvaltonen/python-pickle5) - Pickle module supporting protocol version 5 for model serialization.
- [PIL](https://pillow.readthedocs.io/) - Python Imaging Library for image processing.
- [numpy](https://numpy.org/) - Library for numerical operations and array manipulation.

## License

This project is licensed under the [MIT License](LICENSE).
