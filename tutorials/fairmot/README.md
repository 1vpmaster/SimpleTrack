# FairMOT

Step1.  git clone https://github.com/ifzhang/FairMOT.git


Step2. replace https://github.com/ifzhang/FairMOT/blob/master/src/lib/tracker/multitracker.py

Step3. replace https://github.com/ifzhang/FairMOT/blob/master/src/lib/tracker/matching.py

Step4. run EG tracker example: 
```
python3 track_half.py mot --load_model ../exp/mot/mot17_half_dla34/model_last.pth --match_thres 0.8
```

## Notes
If you want to test the only EG_matrix, you can find the EG_matrix in the EG_tracker.py. And you can replace the origin matrix with EG_matrix, and set the linear assignment thresh to 0.8.

