# Report - B657 Assignment 1: Image Processing and Recognition Basics

## Introduction
----
In this assignment, we presented a semi-dynamic solution to detect and extract students' answers from an OMR sheet. Also, we devised a way to inject the correct answer into the sheet, which can only be decoded using our algorithm and can't be interpreted by the students.

## Setup
----

If you are using a basic python environment with version 3.11 then you can install the requirements by running, `pip install -r requirements.txt`.

You can also install everything using poetry. Make sure you have poetry in your system and run, `poetry install`. It will install everything along with the Python version.

## Methodology
----
We discuss our approach in three parts. In the first part, we will talk about our approach to finding the student's answers from the given image and extracting them. The second part will consist of marking the answers and in the third part, we will talk about injecting and extracting correct answers in the blank form.

### Extracting Answer
----
#### Handling Image
Our main issue with processing the images was with the variance in intensity of the images. Since we wanted to process images with each of the pixels, the differences proved to be an obstacle to generalizing an approach. After trying with multiple thresholds we decided that converting the given image into its corresponding binary space with a threshold value of 128 is the best way to handle it. This proved to be very effective as now we were able to devise a generalized approach that was working for all the images.
#### Finding Regions With Questions
Next, we wanted to find the regions that have the questions (e.g. 1. A B C D E), in the image. We wanted to traverse the image pixel by pixel and wanted to look for white pixels in our binary image assuming that it is the starting of the questions. However, at first, we needed to decide on where to start the traversing as the top part of the image consists of instructions and our injected answer which we will talk about later. Now, since the top part is not needed so wanted to avoid it and based on the given images we decided that the top 30% of the image could be ignored and started processing the image after that.

While traversing the image we encounter another issue. Our goal was to look for white pixels assuming that it is the start of the question, the written answer and some anomaly which we assume is a student just marking the question with a pen/pencil was causing an issue for our algorithm to detect where the question is starting. We tried multiple ways, even with alternating pixel traversing but failed with each method. Finally, we came up with a solution to find the region by traversing from the right side of the image. Based on our analysis, the the questions do not contain such anomalies on the right sides (except for the written answer which we will mention how it has been handled). We do agree that our current approach works under the assumption that no such anomaly is present in the right and starts processing the image in this manner.

Now, we assumed that it would work efficiently on each of the images a further tweaking was necessary. We find that some marked answers by the students do go outside of the box. This was causing our algorithm to incorrectly detect the start of the questions. As our algorithm is only dependent on finding a single pixel to detect the start of a question an anomaly like that was triggering issues. So to fix this issue we checked for a series of white pixels in the binary image. At first, we were looking for a random number of pixels in a straight line but it was not working. We only saw an acceptable result when we used a ratio between white and black pixels in a range of a straight line. This fixed our question detection issue. One thing we felt like to mention is the way we were handling the traversing method. At first, we were going with a manner of all the pixels in the `y`-axis given an `x`-point, which was obviously not working to find the questions in the correct order. So we switched the points for all `x`-axis given a `y`-point and then we were effectively able to find the right coordinate value.

Then, it came down to effectively finding the bounding box over the found question. We tried random sizes to determine the region space and have found that a (0.197, 0.0181) portion of an image actually is comprised of each of the questions. So from there on it was easy to mark the region and we decided to shift the `y`-point by the height of the bounding region so that we can avoid overlapping the bounding region. Now, after getting the bounding region it was pretty straightforward to find the marked region in a question. Our initial algorithm of splitting the bounding region into 6 pieces worked well and we found that a marked region contains more white pixels, approximately 500+ than not marked region. Setting this as a threshold, we were getting accurate results for all the regions.

#### Detecting Written Answer
Since, we already had the bounding region points and were ignoring the left part of the question by setting an offset of 0.047 percent of the image width while detecting questions, finding the written answer region was very easy. Our arbitrary range of (80x30) which is then modified to (widthxheight)=(0.047, 0.0159) portion of the original image proves to be effective in finding the written answer beside the questions. We set a threshold by running the images multiple times and finding that anything more than 100 is safer to assume that the has written an answer by hand. So, our algorithm just takes that into account and adds `x` if it detects anything above the threshold.

### Marking Student's Answers
----
As we are splitting the detected question region into equally smaller images to find the region with marked answers, this approach makes it easier for us to draw over the region. Our algorithm goes through only the image portion of the marked answer and tries to find the white pixel assuming that that is the starting point of the marked region. We opted for this approach as our equal part would contain some whitespace that is not associated with the marked region. For the size of the bounding box, we went through multiple images to get the appropriate ratio given the size of the image.


### Injection & Extraction Of Correct Answers
----
For this problem, we wanted to use with easy injection and extraction method for faster computation. Based on our research, such idea of hiding secret messages in images is known as Steganography. We used a slightly modified version of the algorithm and drew a color bar on the blank form to hide the correct answers. Since it is written inside the pixel it is not possible to decode the answer by hand and also you would require a specific decoding algorithm to get the answers. Our method converts the answer to its appropriate binary value and based on the 16-digit binary value a line is drawn over the image on a specific offset. It is to be noted that our algorithm works under the assumption that the form doesn't go through rigorous changes after the printing and rescanning process. However, changes in pixel values are handled as a JPEG image is known to alter the pixel value after saving to compress the size of the image.

## Evaluation
----
Our approach achieved 100% accuracy on most of the images. However, for the images a-30, a-48, b-13 we found that some of the detections were wrong, mainly for question 30 to 34. On close inspection we found that for these images we were getting wrong bounding boxes of the question but was getting right for most of them. We are currently unsure as to why that might be and in time of writing this report trying to figure out a fix for it. Our best guess is that there are some anomalies on the right side of the questions and as mentioned about the assumptions before our algorithm is throwing wrong values.

## Contributions of the Authors
----
Nilambar Halder Tonmoy:
- Formulating algorithm to process the image on a generalize manner
- Designing algorithm to traverse the image for finding questions on the sheet
- Processing and splitting questions and then re-organization to get the final output
- Finding manual writing of the student and associating the answer with the detected answer

Bhanuprakash Narayana:
- Processing each questions with the found bounded region
- Finding students marked region and drawing marker to show detected answer by the algorithm

Himanshi Kushwaha:
- Research on finding dense region on an image
- Formulating algorithm for injection and extraction work

All member contributed equally on the preparation of the report with each writing up their research findings and reason to choose a method to tackle their part. Also, all member helped with evaluating the algorithm and manually checking the answers for all the test images.