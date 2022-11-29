# seaiceproxy

Code creates a Random Forest Regression model that creates a proxy product for Canadian Arctic sea ice thickness. The dataset that is used contains parameters (stage of development and floe size) from the Canadian Ice Service charts and scatterometer backscatter. The Random Forest Regression is trained on observed CryoSat-2 sea ice thickness. 

seaiceproxy/
|
|- data/
|  |-
|
|- model/
|  |-README.txt 		| Explanation of contents folder
|  |-RFR_C_01.sav 		| Trained Random Forest Regression Model C-band January
|  |-RFR_C_02.sav		
|  |-RFR_C_03.sav		
|  |-RFR_C_04.sav		
|  |-RFR_C_11.sav		
|  |-RFR_C_12.sav		
|  |-RFR_Ku_01.sav		
|  |-RFR_Ku_02.sav		
|  |-RFR_Ku_03.sav		
|  |-RFR_Ku_04.sav		
|  |-RFR_Ku_11.sav		
|  |-RFR_Ku_12.sav		
|
|- src/
|  |- analyze/ 						| Scripts to analyze proxy SIT record
|  |- create_dataset_1992_2020/     | Scripts to create long-term dataset
|  |- create_training_dataset/      | Scripts to create training dataset
|  |- functions/ 					| Define functions
|  |- preprocessing/ 				| Preprocessing of ice charts
|  |- testing/ 						| Testing model
|  |- predict_sit_1992-2020.py 		| Train and apply model
|
|- README.md