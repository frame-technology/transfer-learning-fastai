from fastai.text import *   # Quick access to NLP functionality
import fire
from tensorboardX import SummaryWriter
from fastai import *


@dataclass
class TensorboardLogger(Callback):
    """
    Via: https://github.com/Pendar2/fastai-tensorboard-callback/blob/master/tensorboard_cb.py
    """
    learn: Learner
    run_name: str
    histogram_freq: int = 100
    path: str = None

    def __post_init__(self):
        self.path = self.path or os.path.join(self.learn.path, "logs")
        self.log_dir = os.path.join(self.path, self.run_name)

    def on_train_begin(self, **kwargs):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_epoch_end(self, **kwargs):
        iteration = kwargs["iteration"]
        metrics = kwargs["last_metrics"]
        metrics_names = ["valid_loss"] + [o.__name__ for o in self.learn.metrics]

        for val, name in zip(metrics, metrics_names):
            self.writer.add_scalar(name, val, iteration)

        for name, emb in self.learn.model.named_children():
            if isinstance(emb, nn.Embedding):
                self.writer.add_embedding(list(emb.parameters())[0], global_step=iteration,
                                          tag=name)

    def on_batch_end(self, **kwargs):
        iteration = kwargs["iteration"]
        loss = kwargs["last_loss"]
        if isinstance(loss, tuple):
            pass
            # self.writer.add_scalar("valid_loss", loss[0], iteration)
            # self.writer.add_scalar("accuracy", loss[1], iteration)
        else:
            self.writer.add_scalar("loss", loss, iteration)

        self.writer.add_scalar("learning_rate", self.learn.opt.lr, iteration)
        self.writer.add_scalar("momentum", self.learn.opt.mom, iteration)

        if iteration % self.histogram_freq == 0:
            for name, param in self.learn.model.named_parameters():
                self.writer.add_histogram(name, param, iteration)

    def on_train_end(self, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # dummy_input = next(iter(self.learn.data.train_dl))[0]
                dummy_input = tuple(next(iter(self.learn.data.train_dl))[:-1])
                # TODO waiting on this bug to be fixed before uncommenting below
                # https://github.com/lanpa/tensorboardX/issues/229
                # https://github.com/pytorch/pytorch/pull/12400
                # self.writer.add_graph(self.learn.model, dummy_input)
        except Exception as e:
            print("Unable to create graph.")
            print(e)
        self.writer.close()

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
    frac = sample_size/total_rows if sample_size < total_rows else 1.0
    df = pd.DataFrame()
    for chunk in pd.read_csv(path, chunksize=chunksize, header=None):
        df = pd.concat([df, chunk.sample(frac=frac)])
    return df


def train(dir_path, sample_size=1000):
    print(
        f'train.py dir_path {dir_path}  sample_size {sample_size}')

    build_directory_from_mounts(sample_size=sample_size)
    dftr_lm = pd.read_csv(dir_path+'train_lm.csv', header=None)
    dftr_clas = pd.read_csv(dir_path+'train_clas.csv', header=None)

    # TODO allow sample_size to exceed spreadhseet size. it crashes when I enter 10000
    data_lm = TextLMDataBunch.from_csv(dir_path, train="train_lm")
    data_clas = TextClasDataBunch.from_csv(
        dir_path, train='train_clas', valid='valid_clas', vocab=data_lm.train_ds.vocab)
    print(f'Vocabulary size: {data_lm.train_ds.vocab_size}')
    # TODO Everything below is an attempt to mimic the hyperparaments set in this v0.7 example
    # https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb
    wd = 1e-7
    lr = 1e-3
    lrs = lr
    learn = RNNLearner.language_model(data_lm, drop_mult=0.7, pretrained_fnames=[
        'lstm_wt103', 'itos_wt103'], metrics=accuracy)
    learn.fit_one_cycle(max_lr=slice(1e-3/2, 1), wd=wd, cyc_len=1)
    learn.save_encoder(f'lm_last_ft')
    learn.load_encoder(f'lm_last_ft')
    learn.unfreeze()
    learn.lr_find(start_lr=lrs/10, end_lr=lrs*10)
    learn.fit(lr=slice(lrs, 1), wd=wd, epochs=15,
              callbacks=[TensorboardLogger(learn, f"transfer-learning-{sample_size}")])
    learn.save_encoder('lm1_enc')

    # TODO Unsure if it's necessary to train a classifier for every class, or if v1 handles it
    learn = RNNLearner.classifier(
        data_clas, drop_mult=0.5, clip=0.25, metrics=accuracy)
    learn.metrics = [accuracy]
    learn.load_encoder('lm1_enc')

    wd = 1e-7
    learn.load_encoder('lm1_enc')
    learn.freeze_to(-1)
    learn.lr_find(lrs/1000)
    learn.fit(lr=slice(1e-4, 1e-2), wd=wd, epochs=1,
              callbacks=[TensorboardLogger(learn, f"transfer-learning-{sample_size}")])
    learn.save('clas_0')
    learn.load('clas_0')
    learn.freeze_to(-2)
    learn.fit(lr=slice(1e-4, 1e-2), wd=wd, epochs=1,
              callbacks=[TensorboardLogger(learn, f"transfer-learning-{sample_size}")])
    learn.save('clas_1')
    learn.load('clas_1')
    learn.unfreeze()
    learn.fit(lr=slice(1e-4, 1e-2), wd=wd, epochs=14,
              callbacks=[TensorboardLogger(learn, f"transfer-learning-{sample_size}")])
    learn.save('clas_2')

    acc = f'{learn.recorder.metrics[0][0]}'
    print(f'{"metric": "accuracy", "value": acc}')


if __name__ == '__main__':
    fire.Fire(train)
