#create_training_dataset

The code in this folder creates the training dataset that can be used to train the Random Forest Regression to create the proxy sea ice thickness product. Run in the below order:
- 1_create_training_dataset.py creates dataset_X from the ice charts and dataset_y from observed CryoSat-2 sea ice thickness. Can be run for months November-April and locations Eastern Arctic and Western Arctic.
- 2_add_scatC_trainingdata.py and 2_add_scatKu_trainingdata.py add scatterometer backscatter, from C or Ku band respectively, to dataset_X.
- 3_combine_trainingdata_locations.py combines Eastern and Western Arctic into one dataset.