python  main.py @argfiles/biglstm_argfile_adjseq.txt > ./logs/$(date +"%Y_%m_%d_%I_%M_%S")_biglstm_adjseq.log
python  main.py @argfiles/biglstm_argfile.txt > ./logs/$(date +"%Y_%m_%d_%I_%M_%S")_biglstm.log
python  main.py @argfiles/biglstm_argfile_nomaskadj.txt > ./logs/$(date +"%Y_%m_%d_%I_%M_%S")_biglstm_nomaskadj.log
python  main.py @argfiles/gnn_argfile_adjseq.txt > ./logs/$(date +"%Y_%m_%d_%I_%M_%S")_gnn_adjseq.log
python  main.py @argfiles/gnn_argfile.txt > ./logs/$(date +"%Y_%m_%d_%I_%M_%S")_gnn.log
python  main.py @argfiles/gnn_argfile_nomaskadj.txt > ./logs/$(date +"%Y_%m_%d_%I_%M_%S")_gnn_nomaskadj.log