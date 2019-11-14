# ELLIPSE - LIMB DETECTION
Using OpenCV and MPI Caffe Model

### ABSTRACT
In recent advancements in fields like Machine Learning and Computer Vision, Person detection has become one of the most widely used applications. Person detection only provides you only a limited amount of data. Knowing the position of a person is not enough to determine the nature of his actions. 


### INTRODUCTION
In this project, we take on the challenge to identify the action and nature of a person by using pose estimation models. We used the pre-trained weights of the MPI Caffe Model to identify Key-Points of the human body. We used the coordinates of the Key-Points to wrap the limbs and head inside ellipses. We then accumulated data from the movement of the ellipses caused by the moving subject. We used the data to analyze the behavior of the person.


### PROPOSED MODEL

#### Key-Points Detection:
Downloaded ProtoFile and WeightsFile and made a Deep Neural Network using OpenCV.
Syntax: net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
Captured Video Input using OpenCV’s  cv2.VideoCapture(:PATH:) 
Used every ith  (i = 10 in this case) and ran it through the loaded Neural Network Model which would return coordination of the Key-Points. We have made pairs of all Key-Points manually such that area between two Key-Points represents some part of the human body.

#### Forming Ellipse:
Used the coordinates of Key-Points as a reference to find out Centroid, Major Axis (Vertical), Minor Axis and Angle of the Ellipse.

##### Centroid:
A (x, y) = First Point in the Pair
B (p, q) = Second Point in the Pair

![centroid](https://latex.codecogs.com/gif.latex?Centroid%20%28C_%7B1%7D%2C%20C_%7B2%7D%29%20%3D%20%5Cleft%20%28%20%5Cleft%20%28%20%5Cfrac%7Bx&plus;p%7D%7B2%7D%20%5Cright%20%29%2C%20%5Cleft%20%28%20%5Cfrac%7By&plus;q%7D%7B2%7D%20%5Cright%20%29%20%5Cright%20%29)

##### Major Axis (Vertical):
A (x, y) = First Point in the Pair
B (p, q) = Second Point in the Pair

![major-axis](https://latex.codecogs.com/gif.latex?Length%20of%20Major%20Axis%20%28L%29%20%3D%20%5Csqrt%7B%5Cleft%20%28%5Cfrac%7B%283x&plus;p%29%7D%7B4%7D-%5Cfrac%7B%283y&plus;q%29%7D%7B4%7D%20%5Cright%20%29%5E%7B2%7D%20&plus;%20%5Cleft%20%28%5Cfrac%7B%283p&plus;x%29%7D%7B4%7D-%5Cfrac%7B%283q&plus;y%29%7D%7B4%7D%20%5Cright%20%29%5E%7B2%7D%7D)


##### Minor Axis:
A (x, y) = First Point in the Pair
B (p, q) = Second Point in the Pair

![minor-axis](https://latex.codecogs.com/gif.latex?Length%20of%20Minor%20Axis%20%28M%29%20%3D%20%5Cfrac%7B%5Csqrt%7B%5Cleft%20%28%5Cfrac%7B%283x&plus;p%29%7D%7B4%7D-%5Cfrac%7B%283y&plus;q%29%7D%7B4%7D%20%5Cright%20%29%5E%7B2%7D%20&plus;%20%5Cleft%20%28%5Cfrac%7B%283p&plus;x%29%7D%7B4%7D-%5Cfrac%7B%283q&plus;y%29%7D%7B4%7D%20%5Cright%20%29%5E%7B2%7D%7D%7D%7B2.5%7D)

##### Angle:
A (x, y) = First Point in the Pair
B (p, q) = Second Point in the Pair

![angle](https://latex.codecogs.com/gif.latex?Angle%20%28%5Ctheta%20%29%20%3D%20tan%5E%7B-1%7D%28%5Cfrac%7BA%7D%7BB%7D%29)

Saving Processed Video Sample:
We used OpenCV’s video writer along with MJPG codec format to initiate an AVI file and we had written each and every frame to the output.avi file at end of the algorithm.

### ARCHITECTURE

![arch](https://raw.githubusercontent.com/iam-abbas/ellipse-limb-detection/master/arch.png)

### CONCLUSION
We have collected and analyzed the data and we will be using it for further improvement of model to use it for applications like:
- Action Detection
- Fight Detection

