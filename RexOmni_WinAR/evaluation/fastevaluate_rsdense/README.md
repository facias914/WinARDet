# fastevaluate_rsdense

This evaluator keeps the category-aware matching, IoU thresholds, and area
ranges of `fastevaluate_dota`, but sets `maxDets=INT_MAX`. Every prediction
loaded for each image/category therefore participates in matching and metric
accumulation. At each IoU threshold it pools TP, FP, and FN across categories
within each image, computes precision, recall, and F1 for that image, then
averages each metric across all images. A metric with a zero denominator is
defined as zero. In particular, the reported F1 is the direct mean of per-image
F1 values, not F1 recomputed from the mean precision and mean recall.

The evaluator does not truncate its input. Filtering performed while creating
the prediction TSV (for example `--positive_only`) still applies before the
evaluator receives the boxes.

Build the extension with the Python interpreter that will run the metric:

```bash
cd evaluation/fastevaluate_rsdense
python setup.py build_ext --inplace
```

Run the per-image mean metric with:

```bash
PYTHONPATH=evaluation/fastevaluate_rsdense python \
  evaluation/metrics/coco_lvis_metric_rsdense.py \
  --gt /path/to/annotations.json \
  --pred_tsv /path/to/predictions.tsv \
  --eval_type coco
```

`fastevaluate_rsdense` and `fastevaluate_dota` use different Python package
and extension-module names, so they can be imported in the same process.
