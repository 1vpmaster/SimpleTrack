cd src
python train.py mot --exp_id mix_ft_ch_Decoupledla34_bs_32 --load_model '../models/crowdhuman_simple.pth' --data_cfg '../src/lib/cfg/data.json'
cd ..