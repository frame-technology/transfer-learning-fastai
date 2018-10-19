from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality
import fire


def build_directory_from_mounts(src='/floyd/input/imdb_reviews_wt103/', dst='data/', sample_size=1000):
    """
    Copy mounted directory into writable local, then save training language model csv with the specified sample_size
    """
    shutil.copytree(src, dst)
    dftr_lm = read_csv_with_sample_size(
        dst+'train.csv', sample_size=sample_size)
    dftr_lm.to_csv(dst+'train_lm.csv', index=False)


def read_csv_with_sample_size(path, sample_size=10, chunksize=24000):
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

    learn = RNNLearner.language_model(data_lm, pretrained_fnames=[
                                      'lstm_wt103', 'itos_wt103'])
    learn.unfreeze()
    learn.fit(2, slice(1e-4, 1e-2))
    # TODO this only needs hashing if it's called from the same workspace
    learn.save_encoder(f'enc{len(df)}')

    learn = RNNLearner.classifier(data_clas)
    learn.load_encoder(f'enc{len(df)}')
    learn.fit(3, 1e-3)
    learn.save_encoder(f'enc_clas{len(df)}')

    print('{"metric": "accuracy", "value": 0.985}')


if __name__ == '__main__':
    fire.Fire(train)
