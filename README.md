# Fighting Fake News: A Picture is Worth a Thousand Words

  This is a capstone project for the [Winter 2018 Galvanize Data Science Immersive](https://www.galvanize.com/austin) program.

  The project focuses on addressing the problem of fake news and misinformation in the media. Specifically, it aims to aid users in rooting out the usage of fake images on the internet by automatically scraping web pages related to a topic of interest, cross referencing the images from each each web page with a directory of known fake images, and identifying which images match the fakes with high accuracy.
  
  The project used Scrapy for web scraping, AWS (S3, EC2) for storage and computing, and various Python packages (OpenCV, NumPy, Pandas, scikit learn) for image processing & analysis, data wrangling, modeling, and interpretation.

  See a video presentation of this project here: [Fighting Fake News](https://www.youtube.com/watch?v=HFXuHqcMj8I&list=PLxtztEze-DRfCd2LY4IRytALcjpJQp0WC&index=6)

# Background & Motivation

 Due to the proliferation of social media tools and the ease with which one can set up a website today, anyone can 'publish' content with the potential to garner widespread engagement. As the following figure shows, it can often be difficult to distinguish truth from fiction when it comes to this information:

  ![Fake News Statistics](https://github.com/stooblie/capstone_project/blob/master/capstone_project/image_project/images/project/most_americans_believe_fake_news.jpg)

  Since these stories are believable, they can cause harm by inciting panic, proliferating misinformation, destroying reputations, and more. Just as an article can be false, misleading images are circulated on the internet and they can be just as harmful. As the idiom goes, 'A picture is worth a thousand words.' Even a legitimate, confirmed news article could still insidiously use a false image, and there is no automated way to know if it is real or not. This particular problem was the motivation for the project.

# Methodology

The goal of the project was to maximize the image matching recall, or the percentage of the actual fake images that the model correctly identifies. A secondary goal was to maximize precision, or the percentage of the time an image that the model labels as fake is correctly identified. This priority in accuracy measurements is due to the fact that in the context of the application, the cost of missing a fake image that exists is higher than accidentally labeling real images as false.

An important initial step was dividing the target datasets into 'full frame' and 'out of frame' images. A 'full frame' image is one in which the photo takes up the full frame and is not cropped, surrounded by a border, mixed with other images, etc. This was necessary as there is a lot more complexity involved in finding 'out of frame' fake images, and I had to temper expectations and incrementally grow the model.

![Airport Frames](https://github.com/stooblie/capstone_fighting_fake_news/blob/master/images/project/full_frame_out_of_frame.jpg) 

There were five key stages to the project:

### **1. Data Collection & Storage**

  In order to collect the images for analysis, I constructed a web scraper using the Python library Scrapy. This framework made it possible to easily build a fast web scraper with automatic image and file pipelines into AWS S3. I built unique scrapers with distinct start urls and AWS S3 folder feeds for each 'topic' a user might be interested in.

  The data includes images, image urls, text, and page urls from the web pages. Having the page urls allows the user to return to the websites and collect additional information as necessary.

### **2. Collect Test Images for Measurement**

  One key challenge the project presented was how to measure the success of the system when it was unknown whether the images from the internet were fake or not. This required me to manually seek out samples of the fake images and the different formats they could appear in on the internet, and use those as 'test' images to ensure that my system would capture a sufficient range of formats that the fake images could appear in. For example, this fake image of a shark swimming on the highway during Hurricane Harvey can manifest in the following ways:
  
  ![Shark Collage](https://github.com/stooblie/capstone_project/blob/master/capstone_project/image_project/images/project/test_image_collage_shark.jpg) 

### **3. Oversampling Correction**

  A second significant challenge was managing an unbalanced dataset. The system is essentially finding a 'needle in a haystack', meaning only a small percentage of the images I am analyzing would be positive targets. This is problematic for classification in machine learning, because the models will lean too heavily towards a negative classification if the vast majority of the dataset falls under that category.

  In order to correct for this, I deployed a technique called Synthetic Minority Over-sampling Technique (SMOTE). This involved taking my test images that I know should be detected as true positives by the system, and synthetically creating new observations from that dataset that are effectively weighted averages of similar test images. This allowed me to expand my sample of positives and ensure my model would not overfit and predict too many images as real.

### **4. Image Comparison**

  There are many different ways to 'measure' an image, and it was important for me to research some of the most popular methods for image matching in order to ensure that I was efficient with the progression of my experiment. There are four general measurement techniques I tried for the comparison with mixed results. For the purpose of computation speed, I converted all the images to grayscale for the analysis.

  **Histogram**
  
  This involves converting the image into a histogram representation of the pixel distribution. Since the images were grayscale, this showed the intensity distribution of the pixels and proved to be a fast and indicative feature for the model. The depiction below shows an example of how this technique works to represent each pixel of the photo:
  
  ![Histogram Sample](https://github.com/stooblie/capstone_fighting_fake_news/blob/master/images/project/histogram_sample.jpg)  

  **Structural Similarity Index (SSIM)**
  
  One shortcoming of the histogram method is it does not take into account structural information about the image outside of the overall distribution of pixel intensities. The SSIM metric automatically calculates an index for each picture based on the contrast, luminance, and structure of the picture and helps offset the one-dimensional nature of just using the histogram.

  **Template Matching**
  
  Template matching is critical for finding matches in 'out of frame' images, where the fake image might be in a tweet, cropped, or combined in a frame with other pictures. Since this is a deeper problem than matching cleaner, full frame images, it was not my focus but I did have success using a very specific 'tweet matching' template. Tweets have a specified structure for their embedded images, the pictures are ~93% the width of the tweet and have a 2-1 ratio of width to length. Using these parameters, I was able to create a template that checked specifically  whether the target fake image was inside of the picture of the tweet.

  There is ample opportunity to expand this metric and create a more generalized template matching system.

  **Image Hashing**
  
  Image hashing involves representing the gradient directions from the image's changes in contrast as a hash. You can then compare two images using the similarity of their hashes. This metric was accurate when hashes matched or were very similar, but it failed to help the goal of a high recall rate, it did not score highly on feature importance, and it required 5x the computation time as SSIM, the most intensive of the other measurements. For my application's goals, it was not an ideal fit.

### **5. Modeling**

  I elected to use a Random Forest Classifier to predict which images matched based on their comparison results. Random Forest classifiers include a 'feature importance' measurement that allowed me to see which comparison metric was making the largest impact on determining which classification the image would fall under. Since I was experimenting with several different features for image comparison, 'feature importance' was important for guiding my decision making on wether to keep or exclude certain measurements.

  Before testing the comparison results in the model, I split the data into a training and test set  (75/25) to ensure that my results of the model training would be generalizable to unseen data.

# Results & Analysis

The model ended up performing well, achieving a 94% recall and 93% precision rate on 'full frame' images, and a 57% recall and 83% precision rate on the more ambitious 'out of frame images'. The template matching functionality will need further development to boost performance on the 'out of frame' images.

# Future Improvements

There are several opportunities for future improvement of the project:

1. Automated System for Parameter Tuning and Measuring the Accuracy vs Computation Speed Trade-Off

2. Improved Template Matching

3. Text Analysis

4. Trends in Domains using Fake Images

# Acknowledgements

I would like to thank the entire team from the Winter 2018 Galvanize Data Science Immersive for support related to the project.
