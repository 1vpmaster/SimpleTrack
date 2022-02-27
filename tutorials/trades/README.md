# TraDeS

Step1.  git clone https://github.com/JialianW/TraDeS.git


Step2. 

replace https://github.com/JialianW/TraDeS/blob/master/src/lib/utils/tracker.py

replace https://github.com/JialianW/TraDeS/blob/master/src/lib/opts.py


Step3. run
```
python3 test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --inference --load_model ../models/mot_half.pth --gpus 0 --clip_len 3 --trades --track_thresh 0.4 --new_thresh 0.4 --out_thresh 0.2 --pre_thresh 0.5
```

