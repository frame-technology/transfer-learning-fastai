# transfer-learning-fastai

Run on floydhub workspace. Walk through in the 00_start_here.ipynb

[![Run on FloydHub](https://static.floydhub.com/button/button-small.svg)](https://floydhub.com/run?template=https://github.com/frame/transfer-learning-fastai)

If you want to run jobs locally, just change the sample size

```
floyd init frame/fastai_ulmfit_test # if you haven't created job title yet
floyd run "bash setup.sh && python train.py mytest floyd \
--sample-size 100" \
--env pytorch-1.0 \
--gpu \
--follow \
--data frame/datasets/imdb_reviews_wt103/1:imdb_reviews_wt103 \
-m "sample size 100"
```
