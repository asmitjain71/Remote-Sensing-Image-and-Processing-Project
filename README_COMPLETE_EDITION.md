# 🌞 Final Project Theme 4: Urban Solar Energy Mapping (Complete Edition)

## 1. Project Structure

Create a folder `Solar_Project` with this exact structure:

```
/Solar_Project
    /scripts
        0_slice_images.py          # Prepares data
        1_check_viability.py       # (Student 3) Night Light Filter
        2_train_detector.py        # (Student 1) Building Detection AI (U-Net)
        3_train_classifier.py      # (Student 2 - NEW) Roof Type AI (ResNet) - *For No DSM*
        4_main_pipeline.py         # (Student 4/5) The Master Script (Detection + Analysis)
    /data
        /training_buildings        # INRIA/SpaceNet Dataset (For Script 2)
            /images
            /masks
        /training_roofs            # (For Script 3 - If No DSM)
            /flat                  # Put 50 crop images of flat roofs here
            /gable                 # Put 50 crop images of slanted roofs here
        /project_area
            my_city.tif            # Your high-res map
            my_city_dsm.tif        # (Optional) Height map
            viirs_nightlight.tif   # Night light data
    /models
        unet_detector.pth
        resnet_classifier.pth
    /output
        final_solar_map.shp
```

## 2. The Code

### Script 0: Data Slicer (Preprocessing)

**Does not change.** Slices big images into 512x512 chunks.

(Use the code from the previous response for this step).

### Script 1: Viability Check

**Does not change.** Checks Night Lights.

(Use the code from the previous response for this step).

### Script 2: Train Building Detector (U-Net)

**Does not change.** Trains the model to find building footprints.

(Use the code from the previous response for this step).

### Script 3: Train Roof Classifier (The "No DSM" Solution)

**NEW**: If you do not have height data (DSM), you must train a second, smaller AI to look at the roof image and guess if it is flat or slanted.

**Role**: Student 2

**Prerequisite**: Manually crop ~50 images of flat roofs and ~50 of gable roofs from your city map and put them in `data/training_roofs/flat` and `data/training_roofs/gable`.

```python
# scripts/3_train_classifier.py
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# CONFIG
DATA_DIR = '../data/training_roofs'
MODEL_SAVE_PATH = '../models/resnet_classifier.pth'

def train_classifier():
    # 1. Image Augmentation
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Data
    dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    class_names = dataset.classes # ['flat', 'gable']
    print(f"Classes found: {class_names}")

    # 3. Load Pre-trained ResNet18
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Binary Classification
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 4. Train
    print("🧠 Training Roof Classifier...")
    model.train()
    for epoch in range(10): # 10 Epochs is usually enough for 100 images
        running_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {running_loss/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("✅ Classifier Saved!")

if __name__ == "__main__":
    train_classifier()
```

### Script 4: The Master Pipeline (Detection + Classification + Solar Analysis)

**NEW**: This script combines everything. It detects buildings, checks if you have a DSM, and if not, uses the AI from Script 3. It then performs the full physics and economic calculation.

**Role**: Students 1, 2, 4, 5 (Joint execution)

```python
# scripts/4_main_pipeline.py
import os
import torch
import cv2
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.windows import Window
from rasterio.features import shapes
from shapely.geometry import shape
from torchvision import transforms, models
import torch.nn as nn
import segmentation_models_pytorch as smp
from rasterio.mask import mask

# --- CONFIGURATION ---
CITY_MAP = '../data/project_area/my_city.tif'
DSM_MAP = '../data/project_area/my_city_dsm.tif' # Optional
OUTPUT_SHP = '../output/final_solar_map.shp'

# Solar Constants (Student 4)
SOLAR_IRRADIANCE = 1800  # kWh/m2/year (Avg for India)
PANEL_EFFICIENCY = 0.20  # 20%
PERFORMANCE_RATIO = 0.75 # System losses
ELEC_TARIFF = 8.50       # INR per kWh
CO2_FACTOR = 0.82        # kg CO2 per kWh (Coal baseline)

def main():
    # ---------------------------------------------------------
    # PART 1: DETECT BUILDINGS (The Architect)
    # ---------------------------------------------------------
    print("🚀 Step 1: Detecting Buildings...")
    
    # Load U-Net
    detector = smp.Unet(encoder_name="resnet18", classes=1, in_channels=3)
    detector.load_state_dict(torch.load('../models/unet_detector.pth'))
    detector.eval()
    
    polygons = []
    
    with rasterio.open(CITY_MAP) as src:
        H, W = src.height, src.width
        # Sliding Window
        for row in range(0, H, 512):
            for col in range(0, W, 512):
                window = Window(col, row, 512, 512)
                chip = src.read(window=window, infinite=False)
                if chip.shape[1] < 512 or chip.shape[2] < 512: continue
                
                # Predict
                tensor = torch.from_numpy(chip[:3]/255.0).float().unsqueeze(0)
                with torch.no_grad():
                    mask_pred = detector(tensor).sigmoid().numpy()[0,0]
                
                binary = (mask_pred > 0.5).astype(np.uint8)
                
                # Vectorize
                tfm = src.window_transform(window)
                for geom, val in shapes(binary, transform=tfm):
                    if val == 1: polygons.append(shape(geom))

    gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=src.crs)
    gdf['total_area'] = gdf.area
    gdf = gdf[gdf['total_area'] > 20] # Filter noise
    print(f"   > Found {len(gdf)} buildings.")

    # ---------------------------------------------------------
    # PART 2: CLASSIFY ROOFS (The Classifier)
    # ---------------------------------------------------------
    print("🚀 Step 2: Classifying Roof Types...")
    roof_types = []
    
    # CHECK: Do we have a DSM?
    if os.path.exists(DSM_MAP):
        print("   > DSM found! Using Geometric Method (Accurate).")
        with rasterio.open(DSM_MAP) as dsm:
            for idx, row in gdf.iterrows():
                try:
                    out_img, _ = mask(dsm, [row['geometry']], crop=True)
                    valid = out_img[out_img > -100]
                    std_dev = np.std(valid) if len(valid) > 0 else 0
                    
                    if std_dev < 1.0: roof_types.append("Flat")
                    else: roof_types.append("Gable")
                except: roof_types.append("Flat") # Default

    else:
        print("   > No DSM found! Using Visual AI Method (Fallback).")
        
        # Load ResNet Classifier
        classifier = models.resnet18()
        classifier.fc = nn.Linear(classifier.fc.in_features, 2)
        classifier.load_state_dict(torch.load('../models/resnet_classifier.pth'))
        classifier.eval()
        
        # Preprocessing for ResNet
        tfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        with rasterio.open(CITY_MAP) as src:
            for idx, row in gdf.iterrows():
                try:
                    # Crop image to building
                    out_img, _ = mask(src, [row['geometry']], crop=True)
                    # Convert (C,H,W) -> (H,W,C) for PIL
                    img_array = np.moveaxis(out_img[:3], 0, -1) 
                    
                    input_tensor = tfms(img_array).unsqueeze(0)
                    
                    with torch.no_grad():
                        outputs = classifier(input_tensor)
                        _, preds = torch.max(outputs, 1)
                        
                    roof_types.append("Flat" if preds.item() == 0 else "Gable")
                except:
                    roof_types.append("Flat") # Default

    gdf['roof_type'] = roof_types

    # ---------------------------------------------------------
    # PART 3: SOLAR ANALYSIS (The Engineer & Planner)
    # ---------------------------------------------------------
    print("🚀 Step 3: Performing Solar Analysis...")

    # Logic: Usable Area
    # Flat roofs can use ~70% of area (minus AC units/water tanks)
    # Gable roofs can use ~50% (only the South-facing side)
    gdf['usable_area'] = np.where(gdf['roof_type'] == 'Flat', 
                                  gdf['total_area'] * 0.70, 
                                  gdf['total_area'] * 0.50)

    # Logic: Energy Generation (E = A * r * H * PR)
    gdf['annual_energy_kwh'] = (gdf['usable_area'] * SOLAR_IRRADIANCE * PANEL_EFFICIENCY * PERFORMANCE_RATIO)

    # Logic: Economic Viability
    gdf['annual_savings_inr'] = gdf['annual_energy_kwh'] * ELEC_TARIFF
    
    # Logic: Environmental Impact
    gdf['co2_saved_kg'] = gdf['annual_energy_kwh'] * CO2_FACTOR

    # Save
    gdf.to_file(OUTPUT_SHP)
    
    # ---------------------------------------------------------
    # FINAL REPORT
    # ---------------------------------------------------------
    total_mw = gdf['annual_energy_kwh'].sum() / 1000
    total_savings = gdf['annual_savings_inr'].sum() / 10000000 # In Crores
    
    print("\n" + "="*40)
    print(f"🌞 FINAL CITY REPORT")
    print("="*40)
    print(f"Buildings Detected:   {len(gdf)}")
    print(f"Total Potential:      {total_mw:.2f} MWh/year")
    print(f"Economic Value:       ₹ {total_savings:.2f} Crores/year")
    print(f"CO2 Offset:           {gdf['co2_saved_kg'].sum()/1000:.2f} Tonnes")
    print("="*40)
    print(f"✅ Map saved to: {OUTPUT_SHP}")

if __name__ == "__main__":
    main()
```

## 3. Execution Checklist (How to Run)

### Prepare Data:
1. Download Inria dataset images → `data/training_buildings/`
2. Download your city map → `data/project_area/my_city.tif`
3. **(If No DSM)**: Manually create folders `data/training_roofs/flat` and `/gable` and fill them with 50 small images each.

### Run Preprocessing:
```bash
python scripts/0_slice_images.py
```

### Train Models:
```bash
python scripts/2_train_detector.py  # Wait ~1 hour
```
**(If No DSM)**:
```bash
python scripts/3_train_classifier.py  # Wait ~10 mins
```

### Execute Pipeline:
```bash
python scripts/4_main_pipeline.py
```

### Visualize:
Open `output/final_solar_map.shp` in QGIS.
- Change color style to categorize by `roof_type` or graduate by `annual_savings_inr`.
