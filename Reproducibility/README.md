# Guide for reproducing results of manually labelled thyroids

1. Download the US SegThy Dataset, which can be found under the folliowing link:<br> https://www.cs.cit.tum.de/camp/publications/segthy-dataset/ <br>
   Under
   ```
   US_data/US_volunteer_dataset/ground_truth_data/US_thyroid_label
   ```
   you'll find the manual annotations of the specific scans.
2. As the complete neck area has been annotated, the next step involves extracting the thyroid shape, labeled as '1'. Subsequently, thyroid meshes can be generated using the Marching Cubes       algorithm.
3. Due to different versions of the dataset, one has to rename the files in the following way, so that the 4-fold cross validation is consistent with ours.
    ```
    005_P1_1_left.nii   ->     000.ply
    001_P1_1_left.nii   ->     001.ply
    007_P3_1_left.nii   ->     002.ply
    008_P1_1_left.nii   ->     003.ply
    015_P1_1_left.nii   ->     004.ply
    029_P2_1_left.nii   ->     005.ply
    002_P2_2_left.nii   ->     006.ply
    014_P3_1_left.nii   ->     007.ply
    020_P2_1_left.nii   ->     008.ply
    003_P3_1_left.nii   ->     009.ply
    024_P3_1_left.nii   ->     010.ply
    011_P2_3_left.nii   ->     011.ply
    026_P2_2_left.nii   ->     012.ply
    006_P2_1_left.nii   ->     013.ply
    021_P3_1_left.nii   ->     014.ply
    028_P3_1_left.nii   ->     015.ply
    ```
    These meshes are then stored in the data directory.

5. The training can be started with:

    ```
    python run_s3m.py Reproducibility/config_thyroids_manual.yml
    ```
