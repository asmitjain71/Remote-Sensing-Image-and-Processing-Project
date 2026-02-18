# Theme 4: Urban Solar Energy Mapping - Complete Guide

This is a comprehensive, end-to-end guide to executing Theme 4: Urban Solar Energy Mapping. This guide is broken down into a "Recipe" format so your group can follow it step-by-step.

## Prerequisites (The Setup)

Before starting, your group needs a Python environment. Install these libraries:

```bash
pip install torch torchvision segmentation-models-pytorch rasterio geopandas opencv-python numpy pandas matplotlib shapely
```

## Phase 1: Data Acquisition (The Raw Material)

You need two types of data for the same area:

- **Optical Imagery (RGB)**: For finding buildings.
- **Elevation Data (DSM/LiDAR)**: For analyzing roof slant.

### Where to get it:

**Easier Route (Open Data)**: Use the "Inria Aerial Image Labeling Dataset" or "SpaceNet 2". These datasets already come with high-res images and building footprints (ground truth) which saves you months of labeling work.

**Harder Route (Your Own City)**:
- **RGB**: Use SASPlanet or Google Earth Engine to export a GeoTIFF of your study area (e.g., a university campus).
- **DSM**: Use OpenTopography or JAXA AW3D30 (30m resolution is too low for individual roofs, so you might need to synthesize a "fake" DSM for the project demonstration if you can't get drone data).

## Phase 2: Student 1 (Building Detection)

**Goal**: Input an RGB Image → Output a Shapefile of Building Footprints.

### Step 1: The Model (Python Code)

We will use a library called `segmentation_models_pytorch` which has pre-trained models. This saves you from training from scratch.

```python
import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np

# 1. Load a pre-trained U-Net model
# 'encoder_weights="imagenet"' means it already knows how to see edges/shapes
model = smp.Unet(
    encoder_name="resnet34",        # The "backbone" to extract features
    encoder_weights="imagenet",     # Pre-trained on millions of photos
    in_channels=3,                  # Red, Green, Blue
    classes=1                       # Output: 1 channel (Building Probability)
)

# 2. Load your satellite image
image_path = "my_city_tile.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Preprocessing (Resize to standard size, e.g., 512x512)
# In a real project, you would slice a large image into 512x512 chips here.
img_tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).float().unsqueeze(0)

# 4. Inference (The "Magic")
model.eval()
with torch.no_grad():
    prediction = model(img_tensor)

# 5. Convert prediction to a binary mask (0 or 1)
# Sigmoid converts raw output to probability (0-1). We cut off at 0.5.
mask = prediction.sigmoid().cpu().numpy()[0, 0]
binary_mask = (mask > 0.5).astype(np.uint8)

# Save the mask to verify
cv2.imwrite("building_mask.png", binary_mask * 255)
```

### Step 2: Vectorization (Mask to Shapefile)

Now you have a black-and-white image. You need to turn the white blobs into "Polygons" with coordinates.

```python
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, Polygon
import geopandas as gpd

# Load the mask you just saved
mask = cv2.imread("building_mask.png", 0)

# Extract shapes
# This finds the contours of the white blobs and converts them to coordinates
my_shapes = list(shapes(mask, mask=(mask > 0)))

# Convert to Geometry Objects
polygons = []
for geom, value in my_shapes:
    polygons.append(shape(geom))

# Create a GeoDataFrame (The "Excel sheet" of maps)
gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326") 
gdf.to_file("buildings.shp")  # <-- THIS IS THE DELIVERABLE FOR STUDENT 1
```

## Phase 3: Student 2 (Roof Classification)

**Goal**: Input Building Polygons + DSM → Output "Flat" or "Slanted" label.

If you don't have a DSM, stick to Option A (Visual Classification). If you have elevation data, use Option B (Geometric).

### Option B: Geometric Method (The "Pro" Way)

This code calculates the "slope" of the roof. High variation in slope usually means it's slanted/gabled.

```python
import rasterio
import numpy as np
from rasterio.mask import mask

# 1. Load the Digital Surface Model (DSM)
dsm_path = "city_dsm.tif"
src = rasterio.open(dsm_path)

# 2. Load the buildings from Student 1
gdf = gpd.read_file("buildings.shp")

roof_types = []

for idx, row in gdf.iterrows():
    # Crop the DSM to just this building
    geom = [row['geometry']]
    out_image, out_transform = mask(src, geom, crop=True)
    
    # Remove "No Data" values (usually -9999 or 0)
    valid_pixels = out_image[out_image > 0]
    
    if len(valid_pixels) == 0:
        roof_types.append("Unknown")
        continue

    # 3. Calculate Statistics
    # A flat roof has very low standard deviation in height/slope.
    # A slanted roof has high variance (low at eaves, high at ridge).
    height_std = np.std(valid_pixels)
    
    # Threshold: You must tune this number based on your data units (meters vs feet)
    if height_std < 0.5: 
        roof_types.append("Flat")
    else:
        roof_types.append("Gable/Slanted")

# 4. Save the result
gdf['roof_type'] = roof_types
gdf.to_file("buildings_classified.shp") # <-- DELIVERABLE FOR STUDENT 2
```

## Phase 4: Student 3, 4, 5 (Solar Analysis)

**Goal**: Calculate Energy (kWh) and Money ($$).

This part is pure calculation using the attributes you've created.

```python
# Constants for Physics
SOLAR_IRRADIANCE = 1800  # kWh/m2/year (Example for Rajasthan)
PANEL_EFFICIENCY = 0.20  # 20% efficient panels
PERFORMANCE_RATIO = 0.75 # 75% (Accounts for dust, heat, wire losses)
ENERGY_COST = 8.0        # Rupees per kWh

# Load the file from Student 2
gdf = gpd.read_file("buildings_classified.shp")

# 1. Calculate Area (Student 3)
# Project to a metric CRS (like UTM) to get area in meters, not degrees
gdf = gdf.to_crs(epsg=32643) # Example for India/Rajasthan
gdf['area_sqm'] = gdf.geometry.area

# Define "Usable Area" (We can't cover 100% of the roof)
# Flat roofs use 70% area; Slanted roofs might only use 50% (one side)
gdf['usable_area'] = np.where(gdf['roof_type'] == 'Flat', 
                              gdf['area_sqm'] * 0.7, 
                              gdf['area_sqm'] * 0.5)

# 2. Calculate Energy Potential (Student 4)
# Formula: E = A * r * H * PR
gdf['annual_energy_kwh'] = (gdf['usable_area'] * SOLAR_IRRADIANCE * PANEL_EFFICIENCY * PERFORMANCE_RATIO)

# 3. Calculate Economics (Student 5)
gdf['annual_savings_inr'] = gdf['annual_energy_kwh'] * ENERGY_COST

# Save Final Report
gdf.to_file("final_solar_map.shp")
gdf.to_csv("solar_report.csv")
```

## Summary of Student Deliverables

- **Student 1**: Runs the `segmentation_models_pytorch` code. Delivers `buildings.shp`.
- **Student 2**: Runs the `rasterio` masking code. Delivers `buildings_classified.shp` (adds `roof_type` column).
- **Student 3**: Runs the area calculation code. Delivers `buildings_with_area.shp`.
- **Student 4**: Runs the physics formula. Delivers `energy_potential.csv`.
- **Student 5**: Runs the financial math and creates the final presentation/charts showing "Total City Potential in MW".

## How to make this a "Great" Project

1. **Visuals**: Use QGIS to color-code the final map. Red buildings = High Solar Potential, Blue = Low Potential.

2. **Validation**: Use Google Earth to manually check 10-20 buildings and see if your model correctly identified them as "Flat" or "Slanted." Report your accuracy (e.g., "Our model is 85% accurate").
