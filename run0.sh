export CUDA_VISIBLE_DEVICES=0
#python main.py @argfiles/ashare/base_argfile.txt > ./logs/ashare/mlp_$(date +"%Y_%m_%d_%I_%M_%S").log
#python main.py @argfiles/ashare/lstm_argfile.txt > ./logs/ashare/lstm_$(date +"%Y_%m_%d_%I_%M_%S").log
#python main.py @argfiles/ashare/rgcn_argfile_1rel.txt > ./logs/ashare/rgcn_1rel_$(date +"%Y_%m_%d_%I_%M_%S").log
#python main.py @argfiles/ashare/rgcn_argfile_2rels.txt > ./logs/ashare/rgcn_2rels_$(date +"%Y_%m_%d_%I_%M_%S").log
python main.py @argfiles/ashare/rsr_argfile_implicit.txt > ./logs/ashare/rsr_implicit_$(date +"%Y_%m_%d_%I_%M_%S").log
python main.py @argfiles/ashare/rsr_argfile_explicit.txt > ./logs/ashare/rsr_explicit_$(date +"%Y_%m_%d_%I_%M_%S").log