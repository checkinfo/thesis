export CUDA_VISIBLE_DEVICES=1
python  main.py @argfiles/rgcn_argfile_icgraph_250_0.6.txt > ./logs/rgcn_icgraph_250_0.6_$(date +"%Y_%m_%d_%I_%M_%S").log
python  main.py @argfiles/rgcn_argfile_icgraph_750_0.6.txt > ./logs/rgcn_icgraph_750_0.6_$(date +"%Y_%m_%d_%I_%M_%S").log
python  main.py @argfiles/rgcn_argfile_icgraph_250_0.8.txt > ./logs/rgcn_icgraph_250_0.8_$(date +"%Y_%m_%d_%I_%M_%S").log
python  main.py @argfiles/rgcn_argfile_icgraph_750_0.8.txt > ./logs/rgcn_icgraph_750_0.8_$(date +"%Y_%m_%d_%I_%M_%S").log
