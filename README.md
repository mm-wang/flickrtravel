<!-- jQuery library (served from Google) -->
<script src="//ajax.googleapis.com/ajax/libs/jquery/1.8.2/jquery.min.js"></script>
<!-- bxSlider Javascript file -->
<script src="/js/jquery.bxslider.min.js"></script>
<!-- bxSlider CSS file -->
<link href="/css/jquery.bxslider.css" rel="stylesheet" />

<!-- Slider Ready -->
$(document).ready(function(){
  $('.cbslider').bxSlider();
});

<!-- Camera Brands Slider -->
<ul class="cbslider">
  <li><img src="/images/camerabrands2000.jpg" /></li>
  <li><img src="/images/camerabrands2001.jpg" /></li>
  <li><img src="/images/camerabrands2002.jpg" /></li>
  <li><img src="/images/camerabrands2003.jpg" /></li>
  <li><img src="/images/camerabrands2004.jpg" /></li>
  <li><img src="/images/camerabrands2005.jpg" /></li>
  <li><img src="/images/camerabrands2006.jpg" /></li>
  <li><img src="/images/camerabrands2007.jpg" /></li>
  <li><img src="/images/camerabrands2008.jpg" /></li>
  <li><img src="/images/camerabrands2009.jpg" /></li>
  <li><img src="/images/camerabrands2010.jpg" /></li>
  <li><img src="/images/camerabrands2011.jpg" /></li>
  <li><img src="/images/camerabrands2012.jpg" /></li>
  <li><img src="/images/camerabrands2013.jpg" /></li>
  <li><img src="/images/camerabrands2014.jpg" /></li>
</ul>

<ul class="clslider">
  <li><img src="/images/clusters_2000color.jpg" /></li>
  <li><img src="/images/clusters_2001color.jpg" /></li>
  <li><img src="/images/clusters_2002color.jpg" /></li>
  <li><img src="/images/clusters_2003color.jpg" /></li>
  <li><img src="/images/clusters_2004color.jpg" /></li>
  <li><img src="/images/clusters_2005color.jpg" /></li>
  <li><img src="/images/clusters_2006color.jpg" /></li>
  <li><img src="/images/clusters_2007color.jpg" /></li>
  <li><img src="/images/clusters_2008color.jpg" /></li>
  <li><img src="/images/clusters_2009color.jpg" /></li>
  <li><img src="/images/clusters_2010color.jpg" /></li>
  <li><img src="/images/clusters_2011color.jpg" /></li>
  <li><img src="/images/clusters_2012color.jpg" /></li>
  <li><img src="/images/clusters_2013color.jpg" /></li>
  <li><img src="/images/clusters_2014color.jpg" /></li>
</ul>

<ul class="pcslider">
  <li><img src="/images/prediction_cluster0.jpg" /></li>
  <li><img src="/images/prediction_cluster1.jpg" /></li>
  <li><img src="/images/prediction_cluster2.jpg" /></li>
  <li><img src="/images/prediction_cluster3.jpg" /></li>
  <li><img src="/images/prediction_cluster4.jpg" /></li>
  <li><img src="/images/prediction_cluster5.jpg" /></li>
  <li><img src="/images/prediction_cluster6.jpg" /></li>
  <li><img src="/images/prediction_cluster7.jpg" /></li>
  <li><img src="/images/prediction_cluster8.jpg" /></li>
  <li><img src="/images/prediction_cluster9.jpg" /></li>
  <li><img src="/images/prediction_cluster10.jpg" /></li>
  <li><img src="/images/prediction_cluster11.jpg" /></li>
  <li><img src="/images/prediction_cluster12.jpg" /></li>
  <li><img src="/images/prediction_cluster13.jpg" /></li>
  <li><img src="/images/prediction_cluster14.jpg" /></li>
</ul>

[us]: /images/us.jpg
[kmclusters]: /images/kmeans_15clusters.jpg
[kmcenters]: /images/kmeans_15clustercenters.jpg
[kmsil]: /images/kmeans_silhouette.jpg


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

$('.cbslider').bxSlider({
  auto: true,
  autoControls: true
});

## Final Analysis

My final analysis involved using K-Means clustering to develop a set of regions to analyze the United States and part of Central America in particular. To determine the number of clusters, I used a silhouette score and picked a higher performing number of clusters that still gave me more varied regions than just 3. I then grouped points into each cluster by year, and used that information to create a time series and perform a linear regression.
