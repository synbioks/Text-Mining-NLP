cd /sbksvol/sudhanshu/sbks-ucsd/relation-extraction/biobert_RE/models/pt;
python main_warmup_wd_fix.py --batch_size $1 --epochs $2 --max_seq_len $3 --upsampling $4 > "logs/logfile_drugprot_bsize_$1_epochs_$2_max_seq_len_$3_upratio_$4"
