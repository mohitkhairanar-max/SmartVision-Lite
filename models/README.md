# Models

SmartVision Lite downloads its lightweight pretrained checkpoint through
Ultralytics on first use. Downloaded `*.pt` files are intentionally ignored by
Git.

To use a custom model, copy `best.pt` into this directory and select **Custom**
in the app sidebar. Model weights can be large and may have separate licenses,
so they must not be committed to this repository.
