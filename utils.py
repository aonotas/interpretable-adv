
import numpy as np

def convert_to_vocab_id(vocab, pos, neg, convert_vocab=True, ignore_unk=False, ign_eos=False):
    # binary class
    # Positive => 1
    # Negative => 0
    dataset_x = []
    dataset_x_length = []
    dataset_y = []

    def conv(words):
        if ignore_unk:
            return [vocab.get(w, 1) for w in words if w in vocab]
        else:
            return [vocab.get(w, 1) for w in words]

    for words in pos:
        if convert_vocab:
            if ign_eos:
                conv_words = conv(words)
            else:
                conv_words = conv(words) + [0]
            word_ids = np.array(conv_words, dtype=np.int32)  # EOS
        else:
            word_ids = ' '.join(words)
        dataset_x.append(word_ids)
        dataset_x_length.append(len(word_ids))
        dataset_y.append(1)

    for words in neg:
        if convert_vocab:
            if ign_eos:
                conv_words = conv(words)
            else:
                conv_words = conv(words) + [0]
            word_ids = np.array(conv_words, dtype=np.int32)  # EOS
        else:
            word_ids = ' '.join(words)
        dataset_x.append(word_ids)
        dataset_x_length.append(len(word_ids))
        dataset_y.append(0)

    dataset_y = np.array(dataset_y, dtype=np.int32)
    return dataset_x, dataset_x_length, dataset_y

def load_file_preprocess(filename, lower=True):
    dataset = []
    def conv(w):
        if lower:
            return w.lower()
        return w
    with open(filename, 'r') as f:
        for l in f:
            words = [conv(w) for w in l.strip().split(' ')]
            dataset.append(words)
    return dataset

def load_dataset_imdb(include_pretrain=False, convert_vocab=True, lower=True,
                      min_count=0, ignore_unk=False, use_semi_data=False,
                      add_labeld_to_unlabel=True):
    lm_dataset = None
    imdb_validation_pos_start_id = 10621  # total size: 12499
    imdb_validation_neg_start_id = 10625

    pos_train = load_file_preprocess('data/imdb/imdb_pos_train.txt', lower=lower)
    pos_dev = load_file_preprocess('data/imdb/imdb_pos_dev.txt', lower=lower)

    neg_train = load_file_preprocess('data/imdb/imdb_neg_train.txt', lower=lower)
    neg_dev = load_file_preprocess('data/imdb/imdb_neg_dev.txt', lower=lower)

    if include_pretrain:
        # Pretrain with LM
        unlabled_lm_train = load_file_preprocess('data/imdb/imdb_unlabled.txt', lower=lower)

    pos_test = load_file_preprocess('data/imdb/imdb_pos_test.txt', lower=lower)
    neg_test = load_file_preprocess('data/imdb/imdb_neg_test.txt', lower=lower)

    train_set = pos_train + neg_train
    if include_pretrain:
        # Pretrain with LM
        train_set += unlabled_lm_train

    word_nums = [float(len(words)) for words in train_set]
    print('train_set:{}'.format(len(train_set)))
    print('avg word number:{}'.format(sum(word_nums) / len(word_nums)))

    vocab = {}
    vocab['<eos>'] = 0  # EOS
    vocab['<unk>'] = 1  # EOS
    word_cnt = {}
    for words in train_set:
        for w in words:
            if lower:
                w = w.lower()
            word_cnt[w] = word_cnt.get(w, 0) + 1
    doc_counts = {}
    for words in train_set:
        doc_seen = set()
        for w in words:
            if w not in doc_seen:
                doc_counts[w] = doc_counts.get(w, 0) + 1
                doc_seen.add(w)

    for words in train_set:
        for w in words:
            if lower:
                w = w.lower()
            if w not in vocab and doc_counts[w] > min_count:
                vocab[w] = len(vocab)
    print('vocab:{}'.format(len(vocab)))

    vocab_limit = {}
    for words in pos_train + neg_train:
        for w in words:
            if lower:
                w = w.lower()
            if w not in vocab_limit and doc_counts[w] > min_count:
                vocab_limit[w] = len(vocab_limit)
    train_vocab_size = len(vocab_limit)

    train_x, train_x_len, train_y = convert_to_vocab_id(vocab, pos_train,
                                                        neg_train, convert_vocab=convert_vocab, ignore_unk=ignore_unk)
    word_nums = [len(x) for x in train_x]
    print('avg word number (train_x): {}'.format(sum(word_nums) / len(word_nums)))
    dev_x, dev_x_len, dev_y = convert_to_vocab_id(
        vocab, pos_dev, neg_dev, convert_vocab=convert_vocab, ignore_unk=ignore_unk)

    word_nums = [len(x) for x in dev_x]
    print('avg word number (dev_x):{}'.format(sum(word_nums) / len(word_nums)))
    test_x, test_x_len, test_y = convert_to_vocab_id(
        vocab, pos_test, neg_test, convert_vocab=convert_vocab, ignore_unk=ignore_unk)

    word_nums = [len(x) for x in test_x]
    print('avg word number (test_x):{}'.format(sum(word_nums) / len(word_nums)))
    dataset = (train_x, train_x_len, train_y,
               dev_x, dev_x_len, dev_y,
               test_x, test_x_len, test_y)
    if include_pretrain:
        lm_train_x, _, _ = convert_to_vocab_id(vocab, unlabled_lm_train, [], ignore_unk=ignore_unk)
        lm_train_all = lm_train_x
        if add_labeld_to_unlabel:
            lm_train_all += train_x

        lm_dev_all = test_x
        lm_train_words_num = sum([len(x) for x in lm_train_all])
        lm_dev_words_num = sum([len(x) for x in lm_dev_all])
        print('lm_words_num:{}'.format(lm_train_words_num))

        lm_train_dataset = np.concatenate(lm_train_all, axis=0).astype(np.int32)
        lm_dev_dataset = np.concatenate(lm_dev_all, axis=0).astype(np.int32)

        lm_dataset = (lm_train_dataset, lm_dev_dataset)
        if use_semi_data:
            lm_train_all_length = [len(x) for x in lm_train_all]
            lm_dataset = (lm_train_all, lm_train_all_length)

    vocab_tuple = (vocab, doc_counts)
    return vocab_tuple, dataset, lm_dataset, train_vocab_size



# FCE

def load_file_preprocess_fce_replace(filename, lower=True, ign_eos=False):
    dataset_correct = []
    dataset_wrong = []
    y_tags = []
    def conv(w, correct_flag=True):
        if correct_flag:
            w = w.split('::')[0]
        else:
            w = w.split('::')[-1]
        if lower:
            return w.lower()
        return w
    add_eos = [1]
    if ign_eos is True:
        add_eos = []

    for l in open(filename):
        words = l.strip().split(' ')
        dataset_correct.append([conv(w, False) for w in words])
        dataset_wrong.append([conv(w, True) for w in words])
        y = [0 if len(w.split('::')) >= 2 else 1 for w in words] + add_eos
        y = np.array(y, dtype=np.int32)
        y_tags.append(y)

    return dataset_correct, dataset_wrong, y_tags


# [WIP]
def load_fce(lower=False, min_count=1, ignore_unk=False, use_all_for_lm=False, use_char=False, use_w2v_flag=0, use_semi_data=False):
    dirpath = './gramatical_error/fce-error-detection/tsv/'
    # TODO: replace `::` => split to two differenct text
    ign_eos = True
    train_x_raw, train_y = load_file_preprocess_fce(dirpath + 'fce-public.train.original.tsv', lower=lower, ign_eos=ign_eos)
    dev_x_raw, dev_y = load_file_preprocess_fce(dirpath + 'fce-public.dev.original.tsv', lower=lower, ign_eos=ign_eos)
    test_x_raw, test_y = load_file_preprocess_fce(dirpath + 'fce-public.test.original.tsv', lower=lower, ign_eos=ign_eos)
    w2v = None
    if use_w2v_flag == -1:
        vocab = {}
        vocab['<eos>'] = 0  # EOS
        vocab['<unk>'] = 1  # EOS

        doc_counts = {}
        embedding = './grammatical-error-detection/embedding.txt' # /GWE [Kaneko et al., 2017]
        f = open(embedding)
        f.readline()
        # import gensim
        # from gensim.models.keyedvectors import KeyedVectors
        # w2v = KeyedVectors.load_word2vec_format(embedding, binary=False)
        w2v = {}
        vecs = []
        for l in f:
            l = l.strip()
            w, vec = l.split(' ')[0], l.split(' ')[1:]
            vec = np.array(vec).astype('f')
            if lower:
                w = w.lower()
            if min_count == -1:
                vocab[w] = len(vocab)
            w2v[w] = vec
            vecs.append(vec)
        train_set = train_x_raw
        # print 'train_set:', len(train_set)
        if min_count == -2:
            train_set = train_x_raw + dev_x_raw + test_x_raw
        # print 'train_set:', len(train_set)
        # print 'min_count:', min_count
        doc_counts = {}
        for words in train_set:
            doc_seen = set()
            for w in words:
                if lower:
                    w = w.lower()
                if w not in doc_seen:
                    doc_counts[w] = doc_counts.get(w, 0) + 1
                    doc_seen.add(w)

        for words in train_set:
            for w in words:
                if lower:
                    w = w.lower()
                if w not in vocab and doc_counts[w] > min_count:
                    if min_count >= 0 or min_count == -2:
                        vocab[w] = len(vocab)
        vecs = np.array(vecs).astype('f')

        for words in train_set:
            for w in words:
                if lower:
                    w = w.lower()
                if w in vocab:
                    doc_counts[w] = doc_counts.get(w, 0) + 1

    elif use_w2v_flag == 1:
        vocab = {}
        vocab['<eos>'] = 0  # EOS
        vocab['<unk>'] = 1  # EOS

        doc_counts = {}
        # w2v vocab:
        from gensim.models.keyedvectors import KeyedVectors
        w2v_model = './GoogleNews-vectors-negative300.bin'
        w2v = KeyedVectors.load_word2vec_format(w2v_model, binary=True)

        if min_count == -1:
            for w in w2v.vocab.keys():
                if lower:
                    w = w.lower()
                if w not in vocab:
                    vocab[w] = len(vocab)
        train_set = train_x_raw
        if min_count == -2:
            train_set = train_x_raw + dev_x_raw + test_x_raw

        for words in train_set:
            for w in words:
                if lower:
                    w = w.lower()
                if w not in vocab:
                    if min_count >= 0 or min_count == -2:
                        vocab[w] = len(vocab)
                doc_counts[w] = doc_counts.get(w, 0) + 1

    else:
        vocab = {}
        vocab['<eos>'] = 0  # EOS
        vocab['<unk>'] = 1  # EOS
        word_cnt = {}
        train_set = train_x_raw
        for words in train_set:
            for w in words:
                if lower:
                    w = w.lower()
                word_cnt[w] = word_cnt.get(w, 0) + 1
        doc_counts = {}
        for words in train_set:
            doc_seen = set()
            for w in words:
                if lower:
                    w = w.lower()
                if w not in doc_seen:
                    doc_counts[w] = doc_counts.get(w, 0) + 1
                    doc_seen.add(w)

        for words in train_set:
            for w in words:
                if lower:
                    w = w.lower()
                if w not in vocab and doc_counts[w] > min_count:
                    vocab[w] = len(vocab)
    print('vocab:{}'.format(len(vocab)))

    train_x, train_x_len, _ = convert_to_vocab_id(vocab, train_x_raw, [], ignore_unk=ignore_unk, ign_eos=ign_eos)
    dev_x, dev_x_len, _ = convert_to_vocab_id(vocab, dev_x_raw, [], ignore_unk=ignore_unk, ign_eos=ign_eos)
    test_x, test_x_len, _ = convert_to_vocab_id(vocab, test_x_raw, [], ignore_unk=ignore_unk, ign_eos=ign_eos)
    dataset = (train_x, train_x_len, train_y,
               dev_x, dev_x_len, dev_y,
               test_x, test_x_len, test_y)

    lm_train_dataset = np.concatenate(train_x, axis=0).astype(np.int32)
    lm_dev_dataset = np.concatenate(dev_x, axis=0).astype(np.int32)
    lm_test_dataset = np.concatenate(test_x, axis=0).astype(np.int32)
    lm_dataset = (lm_train_dataset, lm_dev_dataset, lm_test_dataset)
    if use_semi_data:
        lm_dataset = (train_x, train_x_len)
    return vocab, doc_counts, dataset, lm_dataset, w2v
