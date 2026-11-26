# Preprocessing and split

> **Key takeaways.** The 9,459-image modeling pool was stratified by its 41 exact labels into fixed 70/15/15 train, validation, and test CSVs; path and SHA-256 overlap were checked as zero.

Evaluation uses RGB, resize the shorter edge to 256, center crop to 224x224, then ImageNet mean/std normalization. The shared `models/preprocessing_config.json` remains 224x224. Training-only augmentation uses a mild 0.95-1.0 random scale crop, horizontal flip, and restrained color jitter; validation/test transforms do not augment. Horizontal flips preserve left/right orientation-independent leaf symptoms.

The split uses seed 42 and exact folder class stratification: 6,621 train, 1,419 validation, and 1,419 test images. The ten files in five exact duplicate groups were removed before splitting. Since the archive has no plant/session IDs, group-aware splitting was not possible; visually related but non-identical images may still cross splits. Inverse-frequency class weights were used in both full baseline runs.
