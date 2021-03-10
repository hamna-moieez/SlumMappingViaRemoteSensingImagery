# SlumMappingViaRemoteSensingImagery
 This repository contains code for learning slum segmentation and localization using satellite imagery.
This web demo uses a Mask R-CNN deployed on a server and displays the inference done by the model on the test data. Here are some of the images of how the demo works. The initial web interface available to the user once they visit the website.

![Home page](images/home.jpg?raw=true "Home")

### Random Image Selector
You can choose the model results based on locations namely, Islamabad, Karachi Central, Karachi South. This will choose a random image from the chosen location and display the pre-saved model prediction on that image. The random image results generated based on selected location. This feature lets the user select a location from the dropdown list. A random image and the segmented results are displayed from that area.

![Select page](images/upload.jpg?raw=true "Select")

### Image Chooser
The website also allows you to pick your own image in which a slum needs to be mapped. An image chooser utility will help you select an image from your device and runs a live model inference on the image selected. Upload your own image for live network testing. This allows the user to upload a satellite image of a slum to get the prediction from the network. The results along with the image, the ground truth and the predicted mask are displayed.

![Upload page](images/upload2.jpg?raw=true "Upload")

### Inference
The result generated by the underlying model showing network prediction as well as ground truth. These results are displayed when the user uploads a slum image to be fed to the network running in the inference mode.

![Inference page](images/test.jpg?raw=true "inference")

### Demo Video
For better undersrtanding of the research work, visit https://youtu.be/3DmVfxSCvAk for the web demo.
