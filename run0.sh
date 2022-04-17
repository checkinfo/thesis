export CUDA_VISIBLE_DEVICES=0
#python  main.py @argfiles/base_argfile.txt > ./logs/mlp_$(date +"%Y_%m_%d_%I_%M_%S").log
#python  main.py @argfiles/base_argfile_sideinfo.txt > ./logs/mlp_sideinfo_$(date +"%Y_%m_%d_%I_%M_%S").log
#python  main.py @argfiles/lstm_argfile.txt > ./logs/lstm_$(date +"%Y_%m_%d_%I_%M_%S").log
#python  main.py @argfiles/rgcn_argfile.txt > ./logs/rgcn_$(date +"%Y_%m_%d_%I_%M_%S").log
#python  main.py @argfiles/rgcn_argfile_nomaskadj.txt > ./logs/rgcn_nomaskadj_$(date +"%Y_%m_%d_%I_%M_%S").log
#python  main.py @argfiles/rgcn_argfile_adjseq.txt > ./logs/rgcn_adjseq_$(date +"%Y_%m_%d_%I_%M_%S").log
#python  main.py @argfiles/rgcn_argfile_sideinfo.txt > ./logs/rgcn_sideinfo_$(date +"%Y_%m_%d_%I_%M_%S").log
CUDA_VISIBLE_DEVICES=3 python  main.py @argfiles/biglstm_argfile_adjseq_2layer.txt > ./logs/biglstm_$(date +"%Y_%m_%d_%I_%M_%S").log
