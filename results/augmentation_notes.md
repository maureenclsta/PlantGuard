# Augmentation experiment notes

Both runs used the same 41-class mapping, fixed split, BaselineCNN, seed 42, five epochs, batch size 16, AdamW (lr 0.001, weight decay 0.0001), inverse-frequency class weights, and RTX 4050 CUDA. Training augmentation was the intended variable.

| Run | Best validation macro F1 | Best epoch | Train + validation time |
|---|---:|---:|---:|
| No augmentation | 0.4863 | 5 | 786.5 s |
| Training augmentation | 0.3525 | 5 | 808.8 s |

The observed validation difference (augmentation minus no augmentation) was -0.1338 macro F1 in this single seed and five-epoch run. This is descriptive, not evidence that augmentation generally helps or hurts. Both runs reached their best validation score at the final epoch and were still improving, so the epoch limit did not establish convergence. No additional training was started.

Test-set evaluation was performed once, after choosing the no-augmentation checkpoint by validation macro F1. Its metrics are recorded separately in `results/test_evaluation/metrics.json`; they are not a second augmentation comparison because the augmented model was not selected for test evaluation.
