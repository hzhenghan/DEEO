# DEEO
A straightforward yet powerful algorithm that leverages uncertainty estimation for open-set chillers fault diagnosis


## :snake:Prepare Data
You need to purchase the ASHRAE RP-1043 datase and place it in the /data_chiller folder.
Firstly, use res_feaure.py to process the data and calculate the residual features. Then run data_process.ipynb to divide the fault categories into known faults and unknown faults according to the proportion you want.


## :rocket: Training

Simply:
```bash
python3 train.py 
```



