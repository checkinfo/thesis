export CUDA_VISIBLE_DEVICES=0
python main.py @argfiles/ic_graph_2rels_gate/biglstm_rgcn_attn_2rels_5_0.6.txt > ./logs/ic_graph_2rels_gate/biglstm_rgcn_attn_gate_2rels_5_0.6_$(date +"%Y_%m_%d_%I_%M_%S").log
python main.py @argfiles/ic_graph_2rels_gate/biglstm_rgcn_attn_2rels_5_0.8.txt > ./logs/ic_graph_2rels_gate/biglstm_rgcn_attn_gate_2rels_5_0.8_$(date +"%Y_%m_%d_%I_%M_%S").log
python main.py @argfiles/ic_graph_2rels_gate/biglstm_rgcn_attn_2rels_20_0.6.txt > ./logs/ic_graph_2rels_gate/biglstm_rgcn_attn_gate_2rels_20_0.6_$(date +"%Y_%m_%d_%I_%M_%S").log
python main.py @argfiles/ic_graph_2rels_gate/biglstm_rgcn_attn_2rels_20_0.8.txt > ./logs/ic_graph_2rels_gate/biglstm_rgcn_attn_gate_2rels_20_0.8_$(date +"%Y_%m_%d_%I_%M_%S").log