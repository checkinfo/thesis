export CUDA_VISIBLE_DEVICES=1
#python main.py @argfiles/ic_graph_2rels_gate/biglstm_rgcn_attn_2rels_20_0.6.txt > ./logs/ic_graph_2rels_gate_4e-4/biglstm_rgcn_attn_gate_2rels_20_0.6_$(date +"%Y_%m_%d_%I_%M_%S").log
#python main.py @argfiles/ic_graph_2rels_gate/biglstm_rgcn_attn_2rels_20_0.8.txt > ./logs/ic_graph_2rels_gate_4e-4/biglstm_rgcn_attn_gate_2rels_20_0.8_$(date +"%Y_%m_%d_%I_%M_%S").log
#python main.py @argfiles/ic_graph_2rels_gate/biglstm_rgcn_attn_2rels_60_0.6.txt > ./logs/ic_graph_2rels_gate_4e-4/biglstm_rgcn_attn_gate_2rels_6_0.6_$(date +"%Y_%m_%d_%I_%M_%S").log
#python main.py @argfiles/ic_graph_2rels_gate/biglstm_rgcn_attn_2rels_60_0.8.txt > ./logs/ic_graph_2rels_gate_4e-4/biglstm_rgcn_attn_gate_2rels_6_0.8_$(date +"%Y_%m_%d_%I_%M_%S").log
#python main.py @argfiles/ic_graph_2rels_gate/biglstm_rgcn_attn_2rels_250_0.6.txt > ./logs/ic_graph_2rels_gate_4e-4/biglstm_rgcn_attn_gate_2rels_250_0.6_$(date +"%Y_%m_%d_%I_%M_%S").log
#python main.py @argfiles/ic_graph_2rels_gate/biglstm_rgcn_attn_2rels_250_0.8.txt > ./logs/ic_graph_2rels_gate_4e-4/biglstm_rgcn_attn_gate_2rels_250_0.8_$(date +"%Y_%m_%d_%I_%M_%S").log
# python main.py @argfiles/ic_graph_2rels_gate/biglstm_rgcn_attn_2rels_750_0.6.txt > ./logs/ic_graph_2rels_gate_4e-4/biglstm_rgcn_attn_gate_2rels_750_0.6_$(date +"%Y_%m_%d_%I_%M_%S").log
#python main.py @argfiles/ic_graph_2rels_gate/biglstm_rgcn_attn_2rels_750_0.8.txt > ./logs/ic_graph_2rels_gate_4e-4/biglstm_rgcn_attn_gate_2rels_750_0.8_$(date +"%Y_%m_%d_%I_%M_%S").log

python main.py @argfiles/ablation/biglstm_rgcn_attn_2rels_singlemask_750_0.6.txt > ./logs/ablation/biglstm_rgcn_attn_gate_2rels_singlemask_750_0.6_$(date +"%Y_%m_%d_%I_%M_%S").log
# CUDA_VISIBLE_DEVICES=2 python main.py @argfiles/ablation/biglstm_rgcn_2rels_750_0.6.txt > ./logs/ablation/biglstm_rgcn_gate_2rels_750_0.6_$(date +"%Y_%m_%d_%I_%M_%S").log
python main.py @argfiles/ablation/biglstm_rgcn_attn_2rels_nomask_750_0.6.txt > ./logs/ablation/biglstm_rgcn_attn_gate_2rels_nomask_750_0.6_$(date +"%Y_%m_%d_%I_%M_%S").log