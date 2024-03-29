## Paper Title: Towards Automated Meta Review Generation via an NLP/ML Pipeline in Different Stages of the Scholarly Peer Review Process

## Contributing
This repo contains training and prediction code for the recommendation scores and confidence scores for the reviews, using which we then predict the decision on a particular manuscript. Finally, we utilize the decision signals for generating the meta-reviews using a [transformer-based seq2seq architecture](https://arxiv.org/abs/1706.03762).

## Installation Packages

```bash
pip install transformers
pip install pytorch
pip install sklearn
pip install numpy
```

Figure 1. Detailed architecture for recommendation score and confidence score prediction

![Figure 1.](https://github.com/anonymous12-lab/seq-to-seq-decision-aware-mrg/blob/main/recommendation_confidence_pred.jpg)


Figure 2. Below Digram is the detailed architecture for decision prediction. Here R refers to the predicted recommendation scores and C refer to the predicted confidence scores. Three encoders act as feature extractors that map the input vector(for 3 reviews) to a high-level representation. With this representation, the decoder recursively predicts the sequence one at a time auto-regressively.

![Figure 2.](https://github.com/anonymous12-lab/seq-to-seq-decision-aware-mrg/blob/main/fig.png)


Train:
```
python run_seq2seq.py --model_name_or_path experiment_decsion_at_decoder_feat_ext --do_train True --do_eval False --task summarization --train_file peer_data/feat_decoder/mrg_data_with_feat/train.csv --validation_file peer_data/feat_decoder/mrg_data_with_feat/validation.csv --output_dir experiment_decsion_at_decoder_feat_ext --decision_label decision --feat_index feat_index --overwrite_output_dir --per_device_train_batch_size=16 --per_device_eval_batch_size=16 --predict_with_generate --summary_column metareview --do_predict False --save_steps 3606 --num_train_epochs 100
```
Test:
```
python run_seq2seq.py --model_name_or_path experiment_decsion_at_decoder_feat_ext/ --do_train False --do_eval False --task summarization --train_file peer_data/feat_decoder/mrg_data_with_feat/test.csv --validation_file peer_data/feat_decoder/mrg_data_with_feat/validation.csv --output_dir experiment_decsion_at_decoder_feat_ext/checkpoint-674322 --decision_label decision --feat_index feat_index --overwrite_output_dir --per_device_train_batch_size=16 --per_device_eval_batch_size=1 --predict_with_generate --summary_column metareview --do_predict True
```

# Results
![Table 1.](https://github.com/anonymous12-lab/seq-to-seq-decision-aware-mrg/blob/main/evaluate_R_C.png)
Table 1. Are results with respect to F1 score for the both the labels and overall accuracy for decision prediction, where S → sentiment and H → uncertainty score.

![Table 2.](https://github.com/anonymous12-lab/seq-to-seq-decision-aware-mrg/blob/main/evaluate_Seq.png)
Table 2. Are the scores for automatic evaluation metrics for seq-to-seq. The output is the average of all the scores in the test set. R and P refers to recall and precision.

### Next Update/Work:

Adding the Dataset for all four differnet Tasks.

As our immediate next step, we would like to deeply investigate fine-tuning of the specific sub-tasks, use the final-layer representations of the sub-tasks instead of the predictions, and perform a sensitivity analysis of each sub-task on the main task. Additionally, we would like to incorporate more finegrained decisions such as strong/weak accept/reject or minor/major revisions instead of binary decisions.
