# Ankle Sensor Data for Freezing of Gait Detection

## Author
J.D. Delgado-Terán  
Biomedical Signals and Systems, EEMCS, University of Twente  

**Contact Information:**  
Email: j.d.delgadoteran@utwente.nl  
Address:  
University of Twente  
Faculty of Electrical Engineering, Mathematics and Computer Science  
Faculty Office  
P.O. Box 217  
7500 AE Enschede, The Netherlands  

---

## Overview
This dataset contains preprocessed accelerometer and gyroscope data collected from a Movisens Move 4 sensor mounted on the right ankle. The data was recorded during a study involving **21 participants with Parkinson’s Disease** to investigate freezing of gait (FOG) detection and prediction using wearable sensors and machine learning.  

The data is provided in two distinct datasets for machine learning research:  
1. **All-Activities and FOG:** Includes data from various activities, including FOG episodes.  
   - **File names:** `PD0XX_AllActivities_FOG.npz`  
2. **Walking-Turning and FOG:** Focuses on walking, turning, and FOG episodes.  
   - **File names:** `PD0XX_WalkingTurning_FOG.npz`  

Each dataset includes time-series data, binary FOG labels, and associated metadata describing the samples.

---

## Data Description

### Dataset 1: All-Activities and FOG
- **xTensor:** A 3D tensor containing accelerometer and gyroscope data.  
  - **Shape:** (9096, 120, 6)  
    - 9096: Number of samples.  
    - 120: Time steps per sample.  
    - 6: Channels (3 for accelerometer, 3 for gyroscope).  
  - **Data Type:** `float64`  
- **yTensor:** A 1D tensor with binary labels indicating FOG presence.  
  - **Shape:** (9096,)  
  - **Values:** `1` (FOG), `0` (no FOG).  
  - **Data Type:** `int32`  
- **Metadata:** A 1D array with descriptive metadata for each sample.  
  - **Shape:** (9096,)  
  - **Data Type:** `str`  
  - **Content:** Activity type (e.g., Walking, Turning, Standing), participant ID, and trial details.  

### Dataset 2: Walking-Turning and FOG
- **xTensor:** A 3D tensor containing accelerometer and gyroscope data.  
  - **Shape:** (2796, 120, 6)  
    - 2796: Number of samples.  
    - 120: Time steps per sample.  
    - 6: Channels (3 for accelerometer, 3 for gyroscope).  
  - **Data Type:** `float64`  
- **yTensor:** A 1D tensor with binary labels indicating FOG presence.  
  - **Shape:** (2796,)  
  - **Values:** `1` (FOG), `0` (no FOG).  
  - **Data Type:** `int32`  
- **Metadata:** A 1D array with descriptive metadata for each sample.  
  - **Shape:** (2796,)  
  - **Data Type:** `str`  
  - **Content:** Indicates whether the sample involves Walking or Turning, participant ID, and trial details.  

---

## Data Collection and Preprocessing

### Sensor Data
- Data collected from a right ankle-mounted **Movisens Move 4 IMU** with integrated accelerometers and gyroscopes.  
- **Sampling rate:** 60 Hz  

### Preprocessing
1. **Filtering:**  
   - A 3rd-order Butterworth filter was applied.  
     - Low cutoff: 0.3 Hz  
     - High cutoff: 15 Hz  
2. **Segmentation:**  
   - Signals segmented into windows of **120 time steps**.  
   - Overlap:  
     - FOG: 87.5%  
     - Non-FOG (All-Activities or Walking-Turning): 50%  
3. **Annotation:**  
   - Labels were manually annotated based on [1].  
   - Synchronized with visual data from a GoPro (positioned at participants’ feet) and six overhead cameras. Synchronization was ensured by tapping five times in front of the GoPro at the start and end of measurements.  

---

## Usage

### Applications
This dataset is ideal for:  
- Machine learning-based FOG detection and classification.  
- Time-series analysis of mobility patterns.  
- Validating algorithms in neurology and mobility research.  

### Loading the Data
The data is stored in `.npz` files and can be loaded using libraries such as `numpy`, `torch`, or `tensorflow`.  

**Example in Python:**  
```python
import numpy as np

# Load All-Activities and FOG dataset
with np.load('PD0XX_AllActivities_FOG.npz') as data:
    xTensor = data['xTensor']
    yTensor = data['yTensor']
    Metadata = data['Metadata']

# Load Walking-Turning and FOG dataset
with np.load('PD0XX_WalkingTurning_FOG.npz') as data:
    xTensor_wt = data['xTensor']
    yTensor_wt = data['yTensor']
    Metadata_wt = data['Metadata']

Citation
1. Gilat, M. How to Annotate Freezing of Gait from Video: A Standardized Method Using Open-Source Software. JPD 2019, 9, 821–824, doi:10.3233/JPD-191700.


