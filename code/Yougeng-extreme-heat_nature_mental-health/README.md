# Extreme Heat, Nature, and Mental Health

## Green View Index (GVI) Calculation Using Google Street View (GSV) Images

This repository contains code to calculate the **Green View Index (GVI)** based on GPS coordinates and Google Street View images. The workflow is designed to process Urban Mind GPS data and extract vegetation information using a deep learning segmentation model.

---

## Workflow Overview

### 1. Retrieve Google Street View Metadata from GPS Coordinates

- Input: Urban Mind GPS data.
- For each GPS point:
  - Check if a Google Street View (GSV) image is available within a 30-meter radius.
  - If **no image is available**, mark the GVI value as `null` for that point.
  - If **an image is available**, retrieve and store the following metadata:
    - `panoID`
    - `panoTime`
    - `panoLatitude`
    - `panoLongitude`
  - Export the metadata to a text file for image download.

### 2. Download GSV Images

- Use the metadata file to download the corresponding GSV images.
- If multiple GPS points are close together and share the same GSV image, the script avoids downloading duplicates.

### 3. Calculate GVI Using HRNet Segmentation

- Model: [HRNet (v2-w48)](https://www.kaggle.com/models/google/hrnet/TensorFlow2/v2-w48/1), pretrained on the **Berkeley Deep Drive (BDD)** dataset.
- Semantic segmentation is used to identify vegetation.
- GVI is calculated based on pixels classified as:
  - **Category 9**: Vegetation
  - **Category 10**: Terrain

---

## Citation

If you use this code, please cite the related Urban Mind project and acknowledge the HRNet model and the BDD dataset.

---

## License

[MIT License](LICENSE)
