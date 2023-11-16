# cattleid

**Project description**
First, observe the drone flight speed, shooting angles, and altitude of the original video to evaluate the quality of the footage and set the keyframe rate.
From the original video, use the diff algorithm to extract keyframes, reducing redundancy during frame extraction and increasing the diversity of cattle postures, back patterns, and shooting angles.
For data preprocessing, use the Segment Anything Model (SAM) to perform image segmentation on 95 cows, extracting the outlines of the cows. Store the data in separate folders, with each folder named "cattle_ID". There are 95 categories of cows, including standing, lying down, and various other postures. Use pixel value calculations to estimate the pose of the segmented cow outlines, which helps determine the orientation of the cow's head in a markless manner. Then, apply a cyan filter to the data with the determined cow head orientation. This is done because the cows in the dataset are of black, dark brown, brown, light brown, and white colors, and adding this filter enhances the recognition effect.
Utilize multiple algorithms to perform fine-grained image classification tasks on the 95 categories of cow images, training a model that achieves state-of-the-art (SOTA) performance. Finally, train the model to differentiate whether the cow images in the prediction dataset exist in the training dataset. If they do, assign the corresponding existing label ID; if they don't, assign a new ID label.

### Usage

`weights/`needs to be downloaded from the original repository <https://github.com/CWOA/MetricLearningIdentification/tree/master/weights>. Our data is in `datasets/cattleid/`. To train the model, use `python train.py -h` to get help with setting command line arguments. A minimal example would be `python train.py --out_path=output/ --folds_file=datasets/cattleid/splits/10-90.json`.&#x20;

To train on your own dataset, write your own dataset class for managing loading the data, import it into `utilities/utils.py` and add the case to the `def selectDataset(args)` method.

To test a trained model by inferring embeddings and using KNN to classify them, use `python test.py -h` to get help with setting command line arguments. A minimal example would be `python test.py --model_path=output/fold_0/best_model_state.pkl --folds_file=datasets/cattleid/splits/10-90.json --save_path=output/fold_0/`.

To visualise inferred embeddings using T-SNE, use `python utilities/visualse_embeddings.py -h` to get help with setting relevant command line arguments. A minimal example would be `python utilities/visualise_embeddings.py --embeddings_file=output/fold_0/test_embeddings.npz`
