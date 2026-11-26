# Baseline CNN notes

## Runs

Both runs used the same split, class mapping, four-block CNN, seed 42, batch size 16, AdamW (learning rate 0.001, weight decay 0.0001), inverse-frequency class weights, and CUDA on the RTX 4050. The only intended difference was training augmentation. Each completed five epochs; the best validation macro F1 occurred at epoch 5 for both.

| Run | Best validation macro F1 | Training plus validation time |
|---|---:|---:|
| Augmentation off | 0.48631442704987005 | 786.4538493 s |
| Augmentation on | 0.3525222310064326 | 808.8393639999995 s |

These are single-seed validation results. They do not establish that augmentation is generally harmful. In both runs validation macro F1 was still increasing at epoch 5, so the five-epoch limit did not demonstrate a plateau. The validation loss also declined overall across the five epochs. These curves do not show clear overfitting within this short run, but they are not enough to claim convergence.

The no-augmentation checkpoint was selected using validation macro F1 (0.48631442704987005) and then evaluated once on the 1,419-image test set. Test accuracy was 0.5264270613107822, macro precision 0.5102368305302046, macro recall 0.5807870721949809, and macro F1 0.49178949883961426. The model has 83,161 parameters and a 0.3172340393066406 MB float32 parameter footprint. Model-forward time was 2.5544682170524573 ms/image; the full test data pipeline including ZIP image read and preprocessing took 19.162226779424408 ms/image. Details are in `results/test_evaluation/metrics.json` and `classification_report.csv`.

The confusion matrix PNG was saved. The evaluator did not save its underlying matrix counts or test predictions, so exact most-confused class pairs are **NOT YET AVAILABLE** from the committed artifacts. The test set must not be run again to reconstruct them.
