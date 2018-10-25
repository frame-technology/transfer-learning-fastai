from tensorboardX import SummaryWriter
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


@dataclass
class TensorboardLogger(Callback):

    """
    via from https://github.com/Pendar2/fastai-tensorboard-callback
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
        metrics_names = ["valid_loss"] + \
            [o.__name__ for o in self.learn.metrics]

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
