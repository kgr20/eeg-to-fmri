#first run -save_metrics
python -u main.py metrics $1 -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics -seed $2
python -u main.py metrics $1 -fourier_features -random_fourier -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics -seed $2
python -u main.py metrics $1 -topographical_attention -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics -seed $2
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics -seed $2
python -u main.py metrics $1 -topographical_attention -conditional_attention_style -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics -seed $2
python -u main.py metrics $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -save_metrics -seed $2

#mean residues
python -u main.py mean_residues $1 -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -seed $2
python -u main.py mean_residues $1 -fourier_features -random_fourier -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -seed $2
python -u main.py mean_residues $1 -topographical_attention -conditional_attention_style -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -seed $2
python -u main.py mean_residues $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -seed $2

#Relevances of EEG and fMRI
python -u main.py lrp_eeg_fmri $1 -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -seed $2
python -u main.py lrp_eeg_fmri $1 -fourier_features -random_fourier -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -seed $2
python -u main.py lrp_eeg_fmri $1 -topographical_attention -conditional_attention_style -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -seed $2
python -u main.py lrp_eeg_fmri $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -seed $2

#channel relevances
python -u main.py lrp_eeg_channels $1 -topographical_attention -conditional_attention_style -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -seed $2
python -u main.py lrp_eeg_channels $1 -fourier_features -random_fourier -topographical_attention -conditional_attention_style -learning_rate 0.0001 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -verbose -gpu_mem 2000 -batch_size 4 -seed $2

#compare models
python compare.py $1 -name1 \(i\) -name2 \(ii\) -topographical_attention2 -conditional_attention_style2 -epochs 10 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -gpu_mem 2000 -verbose -seed $2
python compare.py $1 -name1 \(i\) -name2 \(iii\) -fourier_features2 -random_fourier2 -epochs 10 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -gpu_mem 2000 -verbose -seed $2
python compare.py $1 -name1 \(i\) -name2 \(iv\)  -topographical_attention2 -conditional_attention_style2 -fourier_features2 -random_fourier2 -epochs 10 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -gpu_mem 2000 -verbose -seed $2
python compare.py $1 -name1 \(ii\) -name2 \(iii\)  -topographical_attention1 -conditional_attention_style1 -fourier_features2 -random_fourier2 -epochs 10 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -gpu_mem 2000 -verbose -seed $2
python compare.py $1 -name1 \(ii\) -name2 \(iv\)  -topographical_attention1 -conditional_attention_style1 -topographical_attention2 -conditional_attention_style2 -fourier_features2 -random_fourier2 -epochs 10 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -gpu_mem 2000 -verbose -seed $2
python compare.py $1 -name1 \(iii\) -name2 \(iv\)  -fourier_features1 -random_fourier1 -topographical_attention2 -conditional_attention_style2 -fourier_features2 -random_fourier2 -epochs 10 -na_path_eeg /home/ist_davidcalhas/eeg_to_fmri/na_models_eeg/na_specification_2 -na_path_fmri /home/ist_davidcalhas/eeg_to_fmri/na_models_fmri/na_specification_2 -gpu_mem 2000 -verbose -seed $2