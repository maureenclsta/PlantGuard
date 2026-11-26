# Baseline CNN notes

## Runs

Both runs used the same split, class mapping, four-block CNN, seed 42, batch size 16, AdamW (learning rate 0.001, weight decay 0.0001), inverse-frequency class weights, and CUDA on the RTX 4050. The only intended difference was training augmentation. Each completed five epochs; the best validation macro F1 occurred at epoch 5 for both.

| Run | Best validation macro F1 | Training plus validation time |
|---|---:|---:|
| Augmentation off | 0.48631442704987005 | 786.4538493 s |
| Augmentation on | 0.3525222310064326 | 808.8393639999995 s |

These are single-seed validation results. They do not establish that augmentation is generally harmful. In both runs validation macro F1 was still increasing at epoch 5, so the five-epoch limit did not demonstrate a plateau. The validation loss also declined overall across the five epochs. These curves do not show clear overfitting within this short run, but they are not enough to claim convergence.

The test set has not been evaluated. The selected checkpoint must be chosen using validation results, then evaluated on the test set once. The test classification report, confusion matrix, test metrics, and inference timings are **NOT YET AVAILABLE**.
