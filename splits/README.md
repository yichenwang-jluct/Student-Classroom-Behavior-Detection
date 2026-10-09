# Split lists

`train.txt`, `val.txt` and `test.txt` list the image files of each partition:
2,061 / 197 / 98 lines. No images are included.

Filenames follow the Roboflow export pattern `<frame>_jpg.rf.<hash>.jpg`, where
`<frame>` is the name of the annotated video frame.

The dataset contains 982 annotated frames: 687 for training, 197 for validation
and 98 for testing. Each training frame appears three times in `train.txt`,
once for each of the three augmented versions produced by the export; the
versions share the frame name and differ in the hash. Validation and test frames
were not augmented and appear once. All versions of a frame are in the same
partition, and no frame occurs in more than one partition.
