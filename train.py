from fastai import *
from fastai.text import *
import fire
from tensorboard_cb import *


def build_directory_from_mounts(src='/floyd/input/imdb_reviews_wt103/', dst='data/', sample_size=1000):
    """
    Copy mounted directory into writable local, then save training language model csv with the specified sample_size
    """
    shutil.copytree(src, dst)
    dftr_lm = read_csv_with_sample_size(
        dst+'train.csv', sample_size=sample_size)
    dftr_lm.to_csv(dst+'train_lm.csv', index=False)


def read_csv_with_sample_size(path, sample_size, chunksize=24000):
    total_rows = get_total_length(path, chunksize=chunksize)
    frac = sample_size/total_rows
    df = pd.DataFrame()
    for chunk in pd.read_csv(path, chunksize=chunksize, header=None):
        df = pd.concat([df, chunk.sample(frac=frac)])
    return df


def train(dir_path, sample_size=1000):
    print(
        f'dir_path {dir_path}  sample_size {sample_size}')

    build_directory_from_mounts(sample_size=sample_size)
    dftr_lm = pd.read_csv(dir_path+'train_lm.csv', header=None)
    dftr_clas = pd.read_csv(dir_path+'train_clas.csv', header=None)

    data_lm = TextLMDataBunch.from_csv(dir_path, train="train_lm")
    data_clas = TextClasDataBunch.from_csv(
        dir_path, train='train_clas', valid='valid_clas', vocab=data_lm.train_ds.vocab)

    # TODO Everything below is an attempt to mimic the hyperparaments set in this v0.7 example
    # https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb
    wd = 1e-7
    lr = 1e-3
    lrs = lr
    learn = RNNLearner.language_model(data_lm, drop_mult=0.7, pretrained_fnames=[
        'lstm_wt103', 'itos_wt103'])
    learn.fit_one_cycle(max_lr=slice(1e-3/2, 1), wd=wd, cyc_len=1)
    learn.save_encoder(f'lm_last_ft')
    learn.load_encoder(f'lm_last_ft')
    learn.unfreeze()
    learn.lr_find(start_lr=lrs/10, end_lr=lrs*10)
    learn.fit(lr=slice(lrs, 1), wd=wd, epochs=15)
    learn.save_encoder('lm1_enc')

    # TODO Unsure if it's necessary to train a classifier for every class, or if v1 handles it
    learn = RNNLearner.classifier(data_clas, drop_mult=0.5, clip=0.25)
    learn.metrics = [accuracy]
    learn.load_encoder('lm1_enc')

    wd = 1e-7
    learn.load_encoder('lm1_enc')
    learn.freeze_to(-1)
    learn.lr_find(lrs/1000)
    learn.fit(lr=slice(1e-4, 1e-2), wd=wd, epochs=1)
    learn.save('clas_0')
    learn.load('clas_0')
    learn.freeze_to(-2)
    learn.fit(lr=slice(1e-4, 1e-2), wd=wd, epochs=1)
    learn.save('clas_1')
    learn.load('clas_1')
    learn.unfreeze()
    learn.fit(lr=slice(1e-4, 1e-2), wd=wd, epochs=14)
    learn.save('clas_2')

    # TODO the tensorboard file should be setup here if it works.
    # print('{"metric": "accuracy", "value": 0.985}')


if __name__ == '__main__':
    fire.Fire(train)
