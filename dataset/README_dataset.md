# Dataset inspection (Phase 0)

## Source and provenance

- Dataset title: *Plant Species and Disease Detection using Deep Learning on Plant Images Captured in Realistic Natural Field Conditions*.
- Source specified by the project: [Mendeley Data, version 2](https://data.mendeley.com/datasets/bsr2vzhrzr/2), DOI [10.17632/bsr2vzhrzr.2](https://doi.org/10.17632/bsr2vzhrzr.2).
- License: CC BY 4.0, as stated in the project specification and the bundled dataset README. The archive README separately identifies itself as v1.0 and says the related paper DOI is pending; this differs from the project’s Mendeley v2 citation.
- Local archive inspected: `data/raw/Custom Dataset for Plant Species and Disease Detec/Ranveer_Realistic_dataset_2000_1126.zip`.
- Archive contains 9,469 JPG images (5,421,708,267 uncompressed image bytes), 41 label folders, and three non-image files (README, crop-information DOCX, and resize script). All images are RGB JPEGs.

## Verified labels and distribution

Folder names are the actual labels. Each label combines a plant/species prefix with a condition name, and some names combine multiple conditions. The data provides one folder-level categorical label per image; it does not provide separate species and disease annotation fields. Keep the original folder names (including spelling) as canonical labels unless the dataset authors provide an authoritative correction.

| Species prefix | Images | Healthy folder present | Healthy images | Label folders |
|---|---:|---|---:|---:|
| grape | 2,197 | Yes | 691 | 9 |
| groundnut | 2,484 | Yes | 137 | 13 |
| guava | 2,360 | Yes | 1,325 | 10 |
| soyabean | 2,428 | Yes | 363 | 9 |
| **Total** | **9,469** |  | **2,516** | **41** |

Per-class counts are in [`class_distribution.csv`](class_distribution.csv). The largest class is `guava_healthy` (1,325); the smallest is `grape_mineral_dificiency` (23), a max/min class ratio of 57.6. Counts by class are exact archive folder counts.

The 41 folder labels are:

- `grape_anthracnose` (113), `grape_guignardia_bidwellii` (73), `grape_healthy` (691), `grape_insect_eating` (793), `grape_mineral_dificiency` (23), `grape_plasmopara_viticola` (199), `grape_plasmopara_viticola_guignardia_bidwellii` (62), `grape_plasmopara_viticola_insect_eating` (189), `grape_powdery_mildew_insect_eating` (54)
- `groundnut_cercospora_arachidicola` (336), `groundnut_cercospora_arachidicola_chlorosis` (103), `groundnut_cercospora_arachidicola_insect_eating` (44), `groundnut_cercospora_arachidicola_millibug` (92), `groundnut_cercospora_personata` (476), `groundnut_cercospora_personata_chlorosis` (527), `groundnut_cercospora_personata_insect_eating` (343), `groundnut_cercospora_personata_millibug` (30), `groundnut_chlorosis` (149), `groundnut_chlorosis_insect_eating` (88), `groundnut_healthy` (137), `groundnut_inseact_eating_mineral_deficiency` (75), `groundnut_mineral_deficiency` (84)
- `guava_cephaleuros_parasiticus` (46), `guava_chlorosis` (114), `guava_chlorosis_scab` (26), `guava_cholrophyl_sucking_insect` (25), `guava_healthy` (1,325), `guava_leaf_eating_insect` (225), `guava_manganese_dificiency` (230), `guava_scab` (251), `guava_scab_insect_eating` (94), `guava_scab_manganese_dificiency` (24)
- `soyabean_cercospora_sojina` (53), `soyabean_cercospora_sojina_insect_eating` (111), `soyabean_cercospora_sojina_mineral_dificiency` (143), `soyabean_healthy` (363), `soyabean_insect_eating_leaf_decay` (822), `soyabean_insect_eating_mineral_dificiency` (365), `soyabean_leaf_eating_caterpillar` (322), `soyabean_leaf_eating_caterpillar_mineral_dificiency` (34), `soyabean_mineral_dificiency` (215)

## Integrity, duplicates, and metadata

Every one of the 9,469 JPGs was opened and fully decoded with Pillow. Result: 9,469 decoded; 0 unreadable/corrupt images. Dimensions are not perfectly uniform: 6,578 are 2000x1126, 2,698 are 634x1126, 190 are 1126x2000, two are 633x1126, and one is 2000x1125. All decoded as RGB.

Exact duplicates were checked by SHA-256 of each image’s uncompressed file bytes. Five exact duplicate groups (10 files) were found. Each group has the same `IMG_20240923_175458.jpg` through `IMG_20240923_175502.jpg` file in both `groundnut_cercospora_arachidicola` and `groundnut_cercospora_personata`. This is both a cross-split leakage risk and a conflicting-label issue. Hash matching does not detect visually similar images encoded differently; no perceptual near-duplicate analysis was done.

No per-image plant ID, capture-session ID, or group manifest was found. Filenames include camera-like date/time names for the duplicated sequence, but the dataset does not document them as grouping metadata. The bundled `PlantsBasicInfo.docx` contains general crop and cultivation notes, not per-image metadata. The archive README describes field capture and a smartphone camera. There are no separate annotation files or structured image-level metadata files in the archive.

The archive is named `2000_1126` and the bundled README describes images at 2000x1126, but dimensions above show multiple orientations/resolutions. No distinct pre-resized training copy was included in the inspected archive. The bundled resize script targets 888x500, not a provided resized dataset.

## Mismatches with documentation and proposed label scheme

The project specification reports four species and 9,469 total images; those totals agree with the archive when prefixes are grouped as grape, groundnut, guava, and soyabean. The specification’s common names use “Soybean”; the source folder spelling is `soyabean`.

The bundled README folder tree lists the same 41 folders as the archive. Its Groundnut table, however, lists 12 classes and omits `groundnut_chlorosis_insect_eating`, which is present in both its tree and the archive. The README table and the project specification still agree with the archive per-species image totals. The bundled crop DOCX mentions Papaya, but there are no Papaya image folders in this archive.

**Label scheme applied in Phase 1:** use the 41 exact folder names as mutually exclusive combined species/condition classes. This preserves supplied labels and supports a Healthy/Diseased status grouping only if all non-healthy folders are mapped to Diseased. It does not support separate species and disease prediction heads without a new annotation mapping. No label spelling normalization or folder merging was applied. The deterministic duplicate handling is documented below.

## Phase 1 data preparation

The raw `class_distribution.csv` records all 9,469 source images. The inventory flags ten files from five exact duplicate SHA-256 groups as excluded from modeling. Both files in each group were dropped because the copies carried conflicting Groundnut folder labels. The prepared pool contains 9,459 images; no arbitrary label was selected for those repeated pixels.

The 9,459 images were stratified by the exact 41 folder labels into train (6,621), validation (1,419), and test (1,419), using seed 42. Split CSVs have zero path overlap and zero SHA-256 overlap. There are no plant/session/group IDs, so related but non-identical images may still cross splits. The percentages are 70/15/15 at the dataset level, with integer class counts stratified. `prepared_class_distribution.csv` reports counts after duplicate-group exclusion. The class mapping is based on those 41 included classes.

The shared preprocessing contract remains RGB 224 x 224, resize the shorter edge to 256 pixels, center crop for evaluation, and ImageNet mean/std normalization. It was not changed during this phase. Training alone adds a mild 0.95-1.0 scale crop, horizontal flip, and mild brightness/contrast/saturation/hue jitter. Horizontal flips are appropriate because leaf symptom labels do not depend on left/right orientation. Vertical flips are omitted. Validation and test use deterministic transforms only. The archive-backed dataset accepts a custom transform for later robustness evaluation and reads images from the ZIP without extracting a second full copy.

Class imbalance remains substantial (raw max/min ratio 57.61). The data module supports inverse-frequency class weights and weighted sampling; the smoke run used neither. The EDA module uses Matplotlib's Agg backend when Matplotlib is installed. The augmentation preview samples the actual `ArchiveImageDataset` with its PyTorch training transform and also exercises a real DataLoader batch.

Reproduce the data preparation and figures with `python scripts/prepare_data.py`, `python scripts/run_eda.py`, and `python scripts/visualize_augmentations.py`. Execute both notebooks with `jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_eda.ipynb notebooks/02_preprocessing.ipynb`.

## How to obtain

Download version 2 from the [Mendeley Data record](https://data.mendeley.com/datasets/bsr2vzhrzr/2), DOI 10.17632/bsr2vzhrzr.2, and place/extract the archive under `data/raw/`. Do not commit the dataset files. Confirm the downloaded archive and license terms against the record before redistribution.

## Reproduction of Phase 0 checks

From the repository root, run `python scripts/prepare_data.py`. It hashes every image member, fully decodes it with Pillow, writes inventory and split CSVs, and asserts there is no split path or content-hash overlap. Then run `python scripts/run_eda.py` and `python scripts/visualize_augmentations.py` to rebuild the figures.
