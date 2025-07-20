# Task-SustainabilityLab
# Feedback

## Question 1: Grid Processing (3 marks total)

### a. Mark Grid Corners and Centers (1 mark)

The png file is located at Task1/grid.png

As the output of the leafmap was html file and would turn into static file when uploaded, i have added screenshot of the marked grid cells

```python
leaf_map.to_html("test_marker_map.html")
```

### b. Filter Images Based on Center Coordinates (1 mark)

Images are filtered like:
- grid_geo_df_wgs having the coordinates of the grids calculated from the coordinates of the delhi ncr shapeline bounds 
- img_geo_df is calculated from the rgb images

```python
filtered_geo_df = geopandas.sjoin(img_geo_df, grid_geo_df_wgs, how='inner', predicate='within')
```

This in the code will calculate intersection(or overlapping regions) of both files

### c. Count Images Before and After Filtering (1 mark)

| Count Type | Number |
|------------|--------|
| Total images before filtering | 9216 |
| Total images after filtering | 9216 |

## Question 2: Class Balance

### Class Imbalance:

| Label | Count |
|-------|-------|
| Barren | 3 |
| Cropland | 3606 |
| Forest | 469 |
| Grassland | 169 |
| Shrubland | 295 |
| Urban | 1357 |
| Water | 7 |
| Wetland | 2 |

To provide nearly equal samples for model to train on, i have augmented the data, using transformation like horizontal flip, vertical flip, and few rotations.

### After Augmentation

| Label | Count |
|-------|-------|
| Barren | 999 |
| Cropland | 6010 |
| Forest | 812 |
| Grassland | 932 |
| Shrubland | 882 |
| Urban | 1357 |
| Water | 999 |
| Wetland | 999 |

### In case of multi class and no pixel:

If there are multiple modes returned then calculate scores based on texture. More detailed edge cases can be formed. More details about texture of each can be learnt and then the texture can be calculated. The **get_label_using_texture()** function returns the class with highest texture in case of multiple labels. Returning the one from the modes could create biasness so I think thats not appropriate.

**get_patch_label()** return mode and handles both no pixel values and multiple labels cases using the above functions

Removed No Data labelled values as it had no pixels

## Question 3: Confusion Matrix and Evaluation Metrics

The explanation of the **confusion matrix** was missing.

While testing, the model performs well on Barren, Grassland, forest, shrubland, urban and wetland. whereas wrongly classifies water and cropland.

Matrix in the form of png is added in the same direcctory

### F1 Score:

#### F1 Custom Scores Per class:

| Class | F1 Score |
|-------|----------|
| Barren | 0.9901 |
| Cropland | 0.7026 |
| Forest | 0.5711 |
| Grassland | 0.8624 |
| Shrubland | 0.7534 |
| Urban | 0.5971 |
| Water | 0.0441 |
| Wetland | 1.0000 |

#### TorchMetrics Per-Class F1 Scores:

| Class | F1 Score |
|-------|----------|
| Barren | 0.9901 |
| Cropland | 0.7026 |
| Forest | 0.5711 |
| Grassland | 0.8624 |
| Shrubland | 0.7534 |
| Urban | 0.5971 |
| Water | 0.0441 |
| Wetland | 1.0000 |

### Each metric should be explained in context to support the interpretation of the model's performance:

- **Barren, wetland** - The model is nearly perfect in identifying this
- **Cropland, shrubland, grassland** - misclassifies few of its samples thus reduces f1 score
- **Urban, forest** - forest and urban recall is good but the precision is poor, thus f1 score is relatively poor
- **Water** - poor recall and precision, thus least f1 score
