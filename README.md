## Paper Title:
## Towards Automated Meta Review Generation via an NLP/ML Pipeline in Different Stages of the Scholarly Peer Review Process

![alt text](https://github.com/anonymous12-lab/seq-to-seq-decision-aware-mrg/blob/main/fig.png)

Train:
```
python run_seq2seq.py --model_name_or_path experiment_decsion_at_decoder_feat_ext --do_train True --do_eval False <br> --task summarization --train_file peer_data/feat_decoder/mrg_data_with_feat/train.csv --validation_file peer_data/feat_decoder/mrg_data_with_feat/validation.csv <br> --output_dir experiment_decsion_at_decoder_feat_ext --decision_label decision --feat_index feat_index --overwrite_output_dir <br> --per_device_train_batch_size=8 --per_device_eval_batch_size=1 --predict_with_generate --summary_column metareview --do_predict False <br> --save_steps 3606 --num_train_epochs 100
```
Test:
```
python run_seq2seq.py --model_name_or_path experiment_decsion_at_decoder_feat_ext/ --do_train False --do_eval False --task summarization --train_file peer_data/feat_decoder/mrg_data_with_feat/test.csv --validation_file peer_data/feat_decoder/mrg_data_with_feat/validation.csv --output_dir experiment_decsion_at_decoder_feat_ext/checkpoint-674322 --decision_label decision --feat_index feat_index --overwrite_output_dir --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --predict_with_generate --summary_column metareview --do_predict True
```
