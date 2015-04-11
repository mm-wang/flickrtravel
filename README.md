[cb2004]: /images/camerabrands2004.jpg
[cb2005]: /images/camerabrands2005.jpg
[cb2006]: /images/camerabrands2006.jpg
[cb2007]: /images/camerabrands2007.jpg
[cb2008]: /images/camerabrands2008.jpg
[cb2009]: /images/camerabrands2009.jpg
[cb2010]: /images/camerabrands2010.jpg
[cb2011]: /images/camerabrands2011.jpg
[cb2012]: /images/camerabrands2012.jpg
[cb2013]: /images/camerabrands2013.jpg
[cb2014]: /images/camerabrands2014.jpg

[us]: /images/us.jpg
[kmclusters]: /images/kmeans_15clusters.jpg
[kmcenters]: /images/kmeans_15clustercenters.jpg
[kmsil]: /images/kmeans_silhouette.jpg

[cl2000]: /images/clusters_2000color.jpg
[cl2001]: /images/clusters_2001color.jpg
[cl2002]: /images/clusters_2002color.jpg
[cl2003]: /images/clusters_2003color.jpg
[cl2004]: /images/clusters_2004color.jpg
[cl2005]: /images/clusters_2005color.jpg
[cl2006]: /images/clusters_2006color.jpg
[cl2007]: /images/clusters_2007color.jpg
[cl2008]: /images/clusters_2008color.jpg
[cl2009]: /images/clusters_2009color.jpg
[cl2010]: /images/clusters_2010color.jpg
[cl2011]: /images/clusters_2011color.jpg
[cl2012]: /images/clusters_2012color.jpg
[cl2013]: /images/clusters_2013color.jpg
[cl2014]: /images/clusters_2014color.jpg

[pc0]: /images/prediction_cluster0.jpg
[pc1]: /images/prediction_cluster1.jpg
[pc2]: /images/prediction_cluster2.jpg
[pc3]: /images/prediction_cluster3.jpg
[pc4]: /images/prediction_cluster4.jpg
[pc5]: /images/prediction_cluster5.jpg
[pc6]: /images/prediction_cluster6.jpg
[pc7]: /images/prediction_cluster7.jpg
[pc8]: /images/prediction_cluster8.jpg
[pc9]: /images/prediction_cluster9.jpg
[pc10]: /images/prediction_cluster10.jpg
[pc11]: /images/prediction_cluster11.jpg
[pc12]: /images/prediction_cluster12.jpg
[pc13]: /images/prediction_cluster13.jpg
[pc14]: /images/prediction_cluster14.jpg


# Predicting Travel Patterns Using Flickr
Using the 100 Million Photos and Videos database from Flickr to predict travel patterns.

## The Hypothesis:

People typically travel to take photographs, or go to a specific place to take photographs. Even if it is their backyard, it is a place that has meaning and visual attraction. I am interested in looking at photography as a predictor of ideal locations to travel to. Where do people like to take photographs? Where _will_ people like to take photographs?

### *Where will people travel?*

## The Preprocessing/Cleaning/Manipulation

The Flickr database consists of the following: 
	> Photo/video ID, User NSID, User nickname, Date taken, Date uploaded, Capture device, Title, Description, User tags (comma-separated), Machine tags (comma-separated). Longitude, Latitude, Accuracy, Photo/video page URL, Photo/video download URL, License name, License URL, Photo/video server identifier, Photo/video farm identifier, Photo/video secret, Photo/video secret original, Photo/video extension original, Photos/video marker (0 = photo, 1 = video)

Cleaning consisted of the following steps:
- Taking out any cameras with "scan" in the name
- Binning the rest of the camera brands, putting any that occur less than 1% of the time into a category "Other"

## Visual Explorations

Through explorations of the camera brands apparent in the dataset, it is clear that there is a growth of Canon cameras over time, although the introduction of the Apple iPhone in 2007 quickly brings Apple into the ring for contention. 

2006                             | 2007
:-------------------------------:|:-------------------------------:
![cb2006]                        | ![cb2007]


## Final Analysis

My final analysis involved using K-Means clustering to develop a set of regions to analyze the United States and part of Central America in particular. To determine the number of clusters, I used a silhouette score and picked a higher performing number of clusters that still gave me more varied regions than just 3. I then grouped points into each cluster by year, and used that information to create a time series and perform a linear regression.