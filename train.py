from fastai.text import *
import fire
from tensorboard_cb import *
from fastai import *


@dataclass
class FloydHubLogger(Callback):
    learn: Learner
    run_name: str
    histogram_freq: int = 100
    path: str = None

    def _emit_floydhub_log_line(self, key, value):
        try:
            value = float(value)
        except ValueError:
            print("Floydhub reporting only support single dimension float metrics")
        else:
            print(f'{{"metric": "{key}", "value": {value}}}')

    def on_epoch_end(self, **kwargs):
        lm = kwargs["last_metrics"]
        metrics_names = ["valid_loss"] + \
            [o.__name__ for o in self.learn.metrics]

        for val, name in zip(lm, metrics_names):
            self._emit_floydhub_log_line(name, val)


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
    dftr_lm.to_csv(os.path.join(dst, 'train_lm.csv'), index=False)


def read_csv_with_sample_size(path, sample_size, chunksize=24000):
    total_rows = get_total_length(path, chunksize=chunksize)
    frac = sample_size / total_rows if sample_size < total_rows else 1.0
    df = pd.DataFrame()
    for chunk in pd.read_csv(path, chunksize=chunksize, header=None):
        df = pd.concat([df, chunk.sample(frac=frac)])
    return df


def train_language_model(data_dir, env, sample_size):
    # TODO allow sample_size to exceed spreadhseet size. it crashes when I enter 10000
    data_lm = TextLMDataBunch.from_csv(data_dir,
                                       train="train_lm",
                                       chunksize=np.ceil(sample_size/2)+1)
    print(f'Vocabulary size: {data_lm.train_ds.vocab_size}')

    # TODO Everything below is an attempt to mimic the hyperparaments set in this v0.7 example
    # https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb
    wd = 1e-7
    lr = 1e-3
    lrs = lr
    learn = RNNLearner.language_model(data_lm,
                                      drop_mult=0.7,
                                      pretrained_fnames=(
                                          'lstm_wt103', 'itos_wt103'),
                                      metrics=accuracy)

    if env == 'floyd':
        cbs = [TensorboardLogger(learn, f"transfer-learning-{sample_size}"),
               FloydHubLogger(learn, f"transfer-learning-{sample_size}")]
    else:
        cbs = list()

    learn.fit_one_cycle(max_lr=slice(1e-3 / 2, 1),
                        wd=wd, cyc_len=1, callbacks=cbs)
    learn.save_encoder(f'lm_last_ft')
    learn.load_encoder(f'lm_last_ft')
    learn.unfreeze()
    learn.lr_find(start_lr=lrs / 10, end_lr=lrs * 10)
    learn.fit(lr=slice(lrs, 1), wd=wd, epochs=15, callbacks=cbs)

    return learn, data_lm


def train_classification_model(data_dir, env, lm_data, sample_size, lm_encoder_name):

    data_clas = TextClasDataBunch.from_csv(data_dir,
                                           train='train_clas',
                                           valid='valid_clas',
                                           vocab=lm_data.train_ds.vocab,
                                           chunksize=np.ceil(sample_size / 2) + 1)

    learn = RNNLearner.classifier(
        data_clas, drop_mult=0.5, clip=0.25, metrics=accuracy)

    if env == 'floyd':
        cbs = [TensorboardLogger(learn, f"transfer-learning-{sample_size}"),
               FloydHubLogger(learn, f"transfer-learning-{sample_size}")]
    else:
        cbs = list()

    learn.metrics = [accuracy]
    learn.load_encoder(lm_encoder_name)

    wd = 1e-7
    learn.load_encoder('lm1_enc')
    learn.fit_one_cycle(max_lr=slice(1e-3 / 2, 1),
                        wd=wd, cyc_len=1, callbacks=cbs)

    return learn


def train_lm_and_sentiment_classifier(exp_name, sample_size=1000, env='local', clean_run=True):
    data_dir = '_'.join([exp_name, str(sample_size)])
    print(f'train.py data_directory {data_dir}  sample_size {sample_size}')

    sample_for_experiment(sample_size=sample_size,
                          dst=data_dir, env=env, clean_run=clean_run)

    print("Training Language Model...")
    lm_learner, lm_data = train_language_model(data_dir, env, sample_size)
    lm_encoder_name = 'lm1_enc'
    lm_learner.save_encoder(lm_encoder_name)

    print("Training Sentiment Classifier...")
    sentiment_learner = train_classification_model(
        data_dir, env, lm_data, sample_size, lm_encoder_name)
    acc = f'{sentiment_learner.recorder.metrics[0][0]}'
    print(f'{{"metric": "accuracy", "value": {acc}}}')


if __name__ == '__main__':
    fire.Fire(train_lm_and_sentiment_classifier)
