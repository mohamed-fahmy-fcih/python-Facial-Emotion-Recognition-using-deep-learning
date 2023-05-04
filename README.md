# python-Facial-Emotion-Recognition-using-deep-learning
# Facial Emotion Recognition | VGG19 Model
![alt text](https://res.cloudinary.com/dzd2k9k9a/image/upload/v1683172419/FER-2013-sample-images-for-facial-emotion-recognition_uz3k4f.jpg)

A typical dataset for face emotion detection is the Fer2013 dataset. The dataset includes 35,887 grayscale portraits of faces displaying 7 various emotions. 
(anger, disgust, fear, happiness, neutral, sad, and surprise).

Convolutional Neural Network (CNN) design is the name of the deep learning model VGG19. Convolutional layers make up 16 of the 19 layers in the VGG19 model,
while completely linked layers make up the remaining 3 layers. Utilizing filters, convolutional layers extract features from the input image.

With the help of the VGG19 model's pre-trained weights, transfer learning can be accomplished. 
The process of solving an issue in a new dataset using the weights of a learned model is known as transfer learning.

Pre-trained weights are frequently used to categorize pictures in the ImageNet collection for the VGG19 model. 
In order to adapt the VGG19 model for face emotions in the fer2013 dataset, the final layer is retrained.

The VGG19 model's final layer is made up of levels that are completely linked. 
The facial emotions in the collection are categorized using these levels.
The final layer of the VGG19 model is retrained to have a classifier with seven outputs because the Fer2013 dataset contains seven distinct mood classes
