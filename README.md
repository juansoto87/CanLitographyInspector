# CanLitographyInspector
Lithographic inspection on aluminum cans


Lithography Inspector

To use the application, the Inspector class must be imported from Inspector.py
The images contained in a folder are prepared using the Crop () method which reads and cuts the images to be analyzed.

The Show_can () method shows the records in a folder.
The Inspect () method compares the evaluated can against the reference can and returns a folder with the defects found.
## How Program Works? ##
1. An image of the lithograph to be analyzed is taken and its MSE is calculated against the reference images and the one with the lowest error is selected.
     ![image](https://github.com/juansoto87/CanLitographyInspector/assets/70484982/25c77099-ae49-43d9-9602-9db8ff49134d)

2. Common points of interest are found using ORB and SIFT.
   ![image](https://github.com/juansoto87/CanLitographyInspector/assets/70484982/20dd04cd-46ff-476c-8009-89ff68a8a37f)
   
3. The homographies are found and the error between the homographies and the reference image is recalculated and the one with the lowest error is selected.
   ![image](https://github.com/juansoto87/CanLitographyInspector/assets/70484982/27a9eea3-8d3f-4eb2-a6fc-5fd276235467)
   
4. The difference between the images is found by comparing them after applying an OTSU thresholding.
   ![image](https://github.com/juansoto87/CanLitographyInspector/assets/70484982/5e1040a1-7a54-4a7a-abc9-94e232f916be)

![lito_analysis_gf](https://github.com/juansoto87/CanLitographyInspector/assets/70484982/98c0da9b-c2b1-4635-8fe8-8bd738b64636)

#########
## Image acquisition ##
With can_inspectgor.py the images were obtained using a Jetson Nano to control a camera and a rotating base. Watch video.

![can_pic_gf](https://github.com/juansoto87/CanLitographyInspector/assets/70484982/adc6e7b9-0744-4aac-9588-efdafdebae6f)
