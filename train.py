import time

from fastai.text import *
import fire
from tensorboard_cb import *
from fastai import *


def timeit(method):
    """
    VIA https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
    :param method:
    :return:
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


@timeit
def sample_for_experiment(sample_size, dst, env, clean_run):
    if env == 'floyd':
        src = '/floyd/input/imdb_reviews_wt103/'
    else:
        src = 'data/csv/'

    if src == dst:
        raise ValueError("Data directory must not be the same as it's source.")

    if clean_run and os.path.isdir(dst):
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    dftr_lm = read_csv_with_sample_size(
        os.path.join(dst, 'train.csv'), sample_size=sample_size)
    dftr_lm.to_csv(os.path.join(dst, 'train_lm.csv'),
                   header=False, index=False)


@timeit
def read_csv_with_sample_size(path, sample_size, chunksize=24000):
    total_rows = get_total_length(path, chunksize=chunksize)
    chunk_length = get_chunk_length(path, chunksize=chunksize)
    # protect against zero
    sample_size = max(sample_size, 1)
    # protect against a fraction too small to sample from a chunk
    frac = min(max(sample_size, chunk_length + 1) / total_rows, 1.0)
    df = pd.DataFrame()
    for chunk in pd.read_csv(path, chunksize=chunksize, header=None):
        df = pd.concat([df, chunk.sample(frac=frac)])
    return df[:sample_size]


@timeit
def train_language_model(data_dir, env, sample_size, global_lm):
    if sample_size < 100:
        return None, None

    data_lm = TextLMDataBunch.from_csv(data_dir,
                                       train="train_lm")
    print(f'Vocabulary size: {data_lm.train_ds.vocab_size}')

    # TODO Everything below is an attempt to mimic the hyperparaments set in this v0.7 example
    # https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb
    wd = 1e-7
    lr = 1e-3
    lrs = lr
    pretrained_fnames = ('lstm_wt103', 'itos_wt103') if global_lm is True else None

    learn = RNNLearner.language_model(data_lm,
                                      # drop_mult=0.7,
                                      pretrained_fnames=pretrained_fnames,
                                      metrics=[accuracy])

    if env == 'floyd':
        cbs = [TensorboardLogger(learn, f"transfer-learning-{sample_size}"),
               FloydHubLogger(learn, f"transfer-learning-{sample_size}")]
    else:
        cbs = list()

    # learn.fit_one_cycle(max_lr=lrs * 10,
    #                     wd=wd, cyc_len=1, callbacks=cbs)
    # learn.save_encoder(f'lm_last_ft')
    # learn.load_encoder(f'lm_last_ft')
    learn.unfreeze()
    # learn.lr_find(start_lr=lrs / 10, end_lr=lrs * 10)
    learn.fit(lr=slice(1e-4, 1e-2),
              # wd=wd,
              epochs=15,
              callbacks=cbs)

    return learn, data_lm


@timeit
def train_classification_model(data_dir, env, lm_data, sample_size, lm_encoder_name):
    vocab = lm_data.train_ds.vocab if lm_data else None
    data_clas = TextClasDataBunch.from_csv(data_dir,
                                           train='train_clas',
                                           valid='valid_clas',
                                           vocab=vocab)

    learn = RNNLearner.classifier(
        data_clas, drop_mult=0.5, clip=0.25, metrics=[accuracy])

    if env == 'floyd':
        cbs = [TensorboardLogger(learn, f"transfer-learning-{sample_size}"),
               FloydHubLogger(learn, f"transfer-learning-{sample_size}")]
    else:
        cbs = list()

    try:
        learn.load_encoder(lm_encoder_name)
    except:
        print(f"Couldn't find encoder {lm_encoder_name}")

    # wd = 1e-7
    # learn.fit_one_cycle(max_lr=1e-3 * 10, wd=wd, cyc_len=1, callbacks=cbs)

    learn.fit(15, 1e-3, callbacks=cbs)

    return learn


@timeit
def train_lm_and_sentiment_classifier(exp_name, sample_size=1000, env='local', clean_run=True,
                                      global_lm=True):
    data_dir = '_'.join([exp_name, str(sample_size)])
    print(
        f'train.py data_directory {data_dir} sample_size={sample_size} global_lm={global_lm} clean_run {clean_run} env {env}')

    sample_for_experiment(sample_size=sample_size,
                          dst=data_dir, env=env, clean_run=clean_run)

    print("Training Language Model...")
    lm_encoder_name = 'lm1_enc'
    lm_learner, lm_data = train_language_model(
        data_dir, env, sample_size, global_lm)
    lm_learner.save_encoder(lm_encoder_name)

    print("Training Sentiment Classifier...")
    sentiment_learner = train_classification_model(
        data_dir, env, lm_data, sample_size, lm_encoder_name)
    acc = f'{sentiment_learner.recorder.metrics[0][0]}'
    print(f'{{"metric": "accuracy", "value": {acc}}}')


if __name__ == '__main__':
    fire.Fire(train_lm_and_sentiment_classifier)
