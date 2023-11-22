# T-MoENet
This is the repository for our work _Temporal-guided Mixture-of-Experts for Zero-Shot Video Question Answering (submitted to TMM)_.


## Evaluation

The pre-trained models and processed test/val data have been uploaded to [Google Drive](https://drive.google.com/drive/folders/1xCiR8t8RxO1ExMM5HI2skSHEzRZ2_cL7?usp=drive_link).

**To test the zero-shot performance on the open-ended VideoQA** datasets (e.g. MSRVTT-QA, MSVD-QA, TGIF-QA, and iVQA), please run `scripts/eval_oe.sh` and change the relevant file paths in it.
```bash
python inference_oe.py --dataset_path <dataset root>/test.csv \
--feat_path <datset root>/clipvitl14.pth \
--vocab_path <dataset root>/vocab1000.json \
--model_path <pretrained checkpoint>
--batch_size 12
```


**To test the zero-shot performance on the multiple choice VideoQA** datasets(e.g. NExT-QA, STAR), please run `scripts/eval_mc.sh` and change the relevant file paths in it. It should be noted that in STAR, we use the test set ```test.csv```, while in NExT-QA, we use ```val.csv```. We recommend adding ```--save_result``` and specifying ```--save_dir``` in the script when make inference on STAR. Then upload the result file to [eval.ai](https://eval.ai/web/challenges/challenge-page/1325/leaderboard/3328/Mean) for evaluating.

```bash
python inference_mc.py --dataset_path <dataset root>/test.csv \
--feat_path <datset root>/clipvitl14.pth \
--model_path <pretrained checkpoint> \
--save_result --save_dir <the directory to save the resulting file>
--batch_size 12
```

## Train

_coming soon..._
