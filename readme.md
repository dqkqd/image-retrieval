# Image Retrieval

Retrieve Image on Oxford5k dataset using [Hessian Affine SIFT](https://github.com/perdoch/hesaff) and [VLAD](https://hal.inria.fr/inria-00548637/file/jegou_compactimagerepresentation_slides.pdf)

## Requirements
```
pip install -r requirements.txt
```

## Running
```
python app.py
```

## Evaluating
    - 100: mAP = 0.4830098928127191
    - 200: mAP = 0.5053583428520593
    - 400: mAP = 0.5251670987207762
    - 800: mAP = 0.5404967361383671
```
python eval.py
```
