# Image Forgery Detection

This project focuses on developing a deep learning-based image forgery detection system using the CASIA dataset. The goal is to accurately classify images as either authentic or forged, with the ability to localize the tampered regions in the forged images.

### Dataset
The dataset used in this project is the CASIA Image Tampering Detection dataset:
1. **Image Splicing Detection Dataset**:
   - Total images: 5,123
   - Authentic images: 3,683
   - Spliced images: 1,440 (generated by splicing multiple authentic images)
   - Covers various challenging scenarios: object insertion, background replacement, composite scenes.

### Methodology
The project explores three different approaches for image forgery detection:

1. **Convolutional Neural Network (CNN) with Adam Optimizer**:
   - The CNN model is built using Keras and TensorFlow.
   - The model consists of multiple convolutional, max-pooling, and fully connected layers.
   - The Adam optimizer is used for training the model.

2. **Convolutional Neural Network (CNN) with RMSProp Optimizer**:
   - Similar to the CNN with Adam optimizer, but using the RMSProp optimizer.

3. **Random Forest Classifier**:
   - A Random Forest classifier is implemented using scikit-learn.

### Results
The performance of the three models is evaluated on the CASIA dataset. The key results are:

1. **CNN with Adam Optimizer**:
   - Validation Accuracy: 94.05%

2. **CNN with RMSProp Optimizer**:
   - Validation Accuracy: 92.36%

3. **Random Forest Classifier**:
   - Validation Accuracy: 89.21%

The CNN models, particularly the one with the Adam optimizer, outperformed the Random Forest classifier, demonstrating the effectiveness of deep learning techniques for image forgery detection.

### Usage
To use the project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/image-forgery-detection.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Download the CASIA Image Tampering Detection dataset and extract it to the appropriate directory.
   - Update the dataset paths in the code accordingly.

4. Train the models:
   - Run the provided scripts to train the CNN models and the Random Forest classifier.

5. Evaluate the models:
   - The scripts will provide the validation and test accuracies for each model.

6. Test the models:
   - Use the provided functions to test the models on individual images or a dataset.

### Conclusion
This project demonstrates the effectiveness of deep learning techniques, particularly CNNs, in the task of image forgery detection. The CNN models with Adam and RMSProp optimizers outperformed the Random Forest classifier, highlighting the ability of deep learning to capture complex patterns in image data.

### References
1. [IEEE Paper on Image Forgery Detection](https://ieeexplore.ieee.org/document/10429021)
2. [Springer Article on Image Forgery Detection](https://link.springer.com/article/10.1007/s11063-024-11448-9)

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/16121160/c78c4f31-d84e-491d-9ffb-9ed06bff01d7/MLT Lab Project Report.pdf
