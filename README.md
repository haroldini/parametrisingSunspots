# Parametrising Transient Sunspot Properties

## MPhys Astrophysics Dissertation Project

## Received 75% overall, 82% in report.

### Retrieves specified sequences of visible continuum intensitygram solar imagery captured by SDO/HMI using helioviewer API. Images within the sequence are centred and flattened using skimage and numpy. Using intensity thresholding, sunspot boundaries are identified and stored as lists of coordinate pairs. Sunspots are parametrised by their areas, centroid positions, perimeter length, and roundness. Umbrae and penumbrae polygons are isolated and parametrised, allowing U/P ratios to be determined.

## Abstract

### An intensity thresholding technique for autonomous sunspot detection was developed using Python and applied to visible continuum intensitygram images captured by SDO/HMI. Using multiple threshold values, the program detects both umbrae and penumbrae and pairs them accordingly. Various procedures were developed to ensure anomalous detections were avoided, including an initial correction of the image to eliminate the limb-darkening effect which occurs towards the edges of the solar disc. Using the shoelace algorithm, methods for calculating sunspots’ areas and centroids were then developed, with uncertainty arising from pixel resolution correctly propagated.

### The technique allowed the Sun’s whole visible hemisphere to be parametrised according to its mean U/P ratio, as well as its overall umbral & penumbral coverage, the total sunspot number, and the latitude and area distributions of sunspots across the image. These solar properties were then tracked across time by passing a sequence of images through the program, allowing averages to be calculated and trends to be observed. Several image sequences were considered, including a 141 image sequence showing the transit of solar cycle 24’s largest sunspot, AR12192, and a 4,042 image sequence detailing the majority of solar cycle 24, from HMI’s first observations in 2010 to the end of 2021. The latter of which resulted in a computed mean U/P ratio of 0.165 ± 1.61 × 10−4, differing 21.4% from the commonly adopted constant value of 0.21. By using the 2014 image sequence, with a resolution of 12 images per day, the monthly smoothed sunspot number during the peak of solar cycle 24 was computed at 174.5 for February, differing 19.4% from the expected 146.1 sunspots recorded by the NOAA during the month. The results also verified that the latitude distribution of sunspots is independent of hemisphere, drifts toward the equator as the solar cycle progresses, and averaged 23.1° and 22.6° for penumbra and umbra respectively.

## Sample Generated Results

### An overlay of detected sunspots over a region of the solar disc:

![Generated sunspot overlay](/README_content/sunspots_overlay.gif)

![Generated sunspot overlay](/README_content/sunspots_overlay2.gif)

### An overlay of detected sunspots over the whole solar disc during 2014:

![Generated sunspot overlay](/README_content/sunspots_2014.gif)

### Area and latitude distribution of detected sunspots across solar cycle 24:

![Generated sunspot overlay](/README_content/distribution_areas_ss24.png)

![Generated sunspot overlay](/README_content/distribution_latitudes_ss24.png)

### Parameters tracked across 2014:

![Generated sunspot overlay](/README_content/plots_2014.png)

### Parameters tracked across solar cycle 24:

![Generated sunspot overlay](/README_content/plots_ss24.png)

### Parameters tracked across transit of AR12192, the largest active region of solar cycle 24:

![Generated sunspot overlay](/README_content/plots_AR12192.png)
