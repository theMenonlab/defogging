# Spatial synthetic fog GUI

Run the tuner from this folder:

```bash
python spatial_fog_gui.py
```

The preview columns are:

1. clear input image
2. synthetic fog image
3. smooth random fog field
4. final fog amount map after depth and spatial modulation

Use `Open folder` to point the GUI at a clear-image dataset. Use `Save preset` to export a JSON parameter file after tuning.

For a headless test preview:

```bash
python spatial_fog_model.py \
  --input data/example_clear.jpg \
  --output outputs/spatial_fog_default_preview.jpg \
  --save-preset-json outputs/spatial_fog_default_preset.json
```
