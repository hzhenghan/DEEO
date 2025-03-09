# DEEO
A straightforward yet powerful algorithm that leverages uncertainty estimation for open-set chillers fault diagnosis


## :snake:Prepare Data
You need to purchase the ASHRAE RP-1043 datase and place it in the /data_chiller folder.
Firstly, use res_feaure.py to process the data and calculate the residual features. Then run data_process.ipynb to divide the fault categories into known faults and unknown faults according to the proportion you want.


## :rocket: Training

You can find a training code on the sine and stock datasets in the repository. You may also run other datasets by adding them and adjusting the dataloader.
To run the training process, run the script of sine_regular.sh or stock_regular.sh.
Or simply:
```bash
python3 train.py --dataset <data_name> --w_kl <a> --w_pred <b>
```



