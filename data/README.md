# Dataset Directory

This directory stores the satellite datasets used in the benchmark:

1. **EuroSAT (RGB)**: Sentinel-2 optical satellite imagery with 10 Land Use Land Cover (LULC) classes (27,000 images, 64x64 pixels).
2. **Geographic Shift Dataset**: Out-of-domain Sentinel-2 TOA imagery collected via Google Earth Engine (`COPERNICUS/S2_HARMONIZED`) and labeled with Dynamic World (`GOOGLE/DYNAMICWORLD/V1`) across 7 geographic regions (Europe vs India).

## Reproducibility Splits
The leak-free stratified 70/15/15 train/val/test split indices are tracked in:
- `splits.json`

## Building Datasets
- EuroSAT is automatically downloaded when running `src/train.py` or `src/dataset.py`.
- Out-of-domain geographic shift tiles can be synthesized using `src/build_shift_dataset.py`.
