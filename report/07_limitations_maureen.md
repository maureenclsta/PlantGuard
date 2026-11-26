# Limitations (Maureen's work)

> **Key takeaways.** Results are limited by the source labels, severe class imbalance, lack of group metadata, one random seed, and a short scratch-CNN run.

The 41 labels combine species and conditions and retain source spelling errors; the model cannot produce separate species and disease outputs without additional verified annotations. The folder names do not provide validated disease ground truth beyond the dataset's labeling. Class frequencies range widely, including small classes with limited test support.

No plant/session grouping metadata was supplied, so residual leakage among related but non-identical images is possible. Only one split and seed were used. The baseline is a small CNN trained for five epochs, and its validation scores were still increasing at the final epoch. The reported test metrics apply to one checkpoint selected by validation macro F1. No robustness evaluation, external field validation, or multi-seed uncertainty analysis is included. Confidence values are not calibrated probabilities. The frontend is contract-ready but has not been integrated with Disca's real backend.
