import tarfile

import fire
from fastai.text import *
from callbacks import *


def get_imdb_data(working_dir):
    """
    Downloads and parses the aclIMDB dataset from Stanford. Will not repeat download if a local
    copy already exists.
    :param working_dir:
    :return: df_trn - training data (labeled and unlabeled)
             df_val - validation data (all labeled)
    """

    def untar_data(url: str, fname: PathOrStr = None, dest: PathOrStr = None):
        "Download `url` if doesn't exist to `fname` and un-tgz to folder `dest`"
        dest = Path(dest)
        fname = Path(fname)
        download_url(url, fname)
        if not dest.exists():
            tarfile.open(fname, 'r:gz').extractall(dest.parent)
        return dest

    untar_data(url='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
               fname='aclImdb_v1.tar.gz',
               dest=working_dir/'aclImdb')

    CLASSES = ['neg', 'pos', 'unsup']

    def get_texts(path):
        texts, labels = [], []
        for idx, label in enumerate(CLASSES):
            for fname in (path/label).glob('*.*'):
                texts.append(fname.open('r', encoding='utf-8').read())
                labels.append(idx)
        return np.array(texts), np.array(labels)

    trn_texts, trn_labels = get_texts(working_dir / 'aclImdb' / 'train')
    val_texts, val_labels = get_texts(working_dir / 'aclImdb' / 'test')

    col_names = ['labels', 'text']

    df_trn = pd.DataFrame({'text': trn_texts, 'labels': trn_labels}, columns=col_names)
    df_val = pd.DataFrame({'text': val_texts, 'labels': val_labels}, columns=col_names)

    return df_trn, df_val


def sample_for_experiment(train_df, test_df, sample_size, dst):
    """
    Given training and test data, sample with respect to sample size and store samples in csv's
    (as fast.ai prefers) in dst. Note we make a number of assumptions here:
        1. we are not stratifying our sample (imdb data is already balanced)
        2. labeled data set aside
    :param train_df:
    :param test_df:
    :param sample_size:
    :param dst:
    :return:
    """

    # count of labeled examples to be used in our classification task
    task_training_size = 500
    task_testing_size = 500

    # sample labeled sentiment
    labeled_train = train_df[train_df['labels'] != 2].sample(n=task_training_size)

    if sample_size > task_training_size:
        additional_rows = sample_size - task_training_size
    else:
        additional_rows = 0

    if additional_rows > 0:
        # pull additional examples from unlabeled set for language model training
        lm_train = train_df[False == train_df.index.isin(labeled_train.index)].sample(n=additional_rows)
        lm_train = pd.concat([labeled_train, lm_train])
    else:
        lm_train = labeled_train

    lm_train.to_csv(dst / 'train_lm.csv', header=False, index=False)

    CLASSES = ['neg', 'pos']
    (dst / 'classes.txt').open('w', encoding='utf-8').writelines(f'{o}\n' for o in CLASSES)
    labeled_train.to_csv(dst / 'train_clas.csv', header=False, index=False)
    test_df.sample(n=task_testing_size).to_csv(dst / 'valid.csv', header=False, index=False)


def train_language_model(data_dir, env, global_lm):

    data_lm = TextLMDataBunch.from_csv(data_dir,
                                       train="train_lm")
    print(f'LM Train Vocabulary size: {data_lm.train_ds.vocab_size}')

    pretrained_fnames = (
        'lstm_wt103', 'itos_wt103') if global_lm is True else None

    learn = RNNLearner.language_model(data_lm,
                                      pretrained_fnames=pretrained_fnames,
                                      metrics=[accuracy])

    print(f'LM Vocabulary size: {learn.model[0].encoder.num_embeddings}')
    print(f'LM Embedding dim: {learn.model[0].encoder.embedding_dim}')

    if env == 'floyd':
        cbs = [TensorboardLogger(learn, f"transfer-learning"),
               FloydHubLogger(learn, f"transfer-learning")]
    else:
        cbs = list()
    learn.unfreeze()
    learn.fit(lr=slice(1e-4, 1e-2),
              epochs=8,
              callbacks=cbs)

    return learn, data_lm


def train_classification_model(data_dir, env, lm_data, lm_encoder_name):

    vocab = lm_data.train_ds.vocab

    data_clas = TextClasDataBunch.from_csv(data_dir,
                                           train='train_clas',
                                           vocab=vocab)

    print(f'Classifier Train Data Vocabulary size: {len(data_clas.vocab.itos)}')
    learn = RNNLearner.classifier(data_clas,
                                  drop_mult=0.5,
                                  clip=0.25,
                                  metrics=[accuracy])

    print(f'Classifier Vocabulary size: {learn.model[0].encoder.num_embeddings}')
    print(f'Classifier Embedding dim: {learn.model[0].encoder.embedding_dim}')

    if env == 'floyd':
        cbs = [TensorboardLogger(learn, f"transfer-learning"),
               FloydHubLogger(learn, f"transfer-learning")]
    else:
        cbs = list()

    learn.load_encoder(lm_encoder_name)

    learn.fit(8, 1e-3, callbacks=cbs)
    return learn


def train_lm_and_sentiment_classifier(exp_name, sample_size=1000, env='local',
                                      global_lm=True):
    data_dir = '_'.join([exp_name, str(sample_size)])
    data_dir = Path(f'./{data_dir}/')

    # grab and parse imdb review sentiment data
    df_trn, df_val = get_imdb_data(data_dir)

    # make sure we have wikitext language model
    model_path = data_dir / 'models'
    model_path.mkdir(exist_ok=True)
    url = 'http://files.fast.ai/models/wt103_v1/'
    download_url(f'{url}lstm_wt103.pth', model_path / 'lstm_wt103.pth')
    download_url(f'{url}itos_wt103.pkl', model_path / 'itos_wt103.pkl')

    sample_for_experiment(train_df=df_trn,
                          test_df=df_val,
                          sample_size=sample_size,
                          dst=data_dir)

    print("Training Language Model...")
    lm_encoder_name = 'lm1_enc'
    lm_learner, lm_data = train_language_model(
        data_dir, env, global_lm)
    lm_learner.save_encoder(lm_encoder_name)

    print("Training Sentiment Classifier...")
    sentiment_learner = train_classification_model(
        data_dir, env, lm_data, lm_encoder_name)
    f'{sentiment_learner.recorder.metrics[0][0]}'


if __name__ == '__main__':
    fire.Fire(train_lm_and_sentiment_classifier)
