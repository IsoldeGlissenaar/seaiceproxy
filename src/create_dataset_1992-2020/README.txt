#create_dataset_1992_2020

The code in this folder creates the dataset that can be used by the Random Forest Regression to create the proxy sea ice thickness product. Run in the below order:
- 1_create_dataset_1992-2020.py creates dataset_X from the ice charts. Can be run for months November-April and locations Eastern Arctic and Western Arctic.
- 2_add_scatC_1992-2020.py and 2_add_scatKu_1992-2020.py add scatterometer backscatter, from C or Ku band respectively, to dataset_X.
- 3_combine_dataset_locations_1992-2020.py combines Eastern and Western Arctic into one dataset.
- 4_remove_notused_params.py removes icebergs and combines old ice types for months SYI not used in training. 