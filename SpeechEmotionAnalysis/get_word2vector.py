from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
def textWordvec():
    num_features = 300  # Word vector dimensionality
    min_word_count = 10  # Minimum word count
    num_workers = 0  # Number of threads to run in parallel
    context = 1  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    #pinyinWithoutT
    # sentences = word2vec.LineSentence("./data/pinyinWithoutT.txt")
    sentences = word2vec.LineSentence("RawData/CASIA database/text_jieba.txt")

    model = Word2Vec(sentences, workers=num_workers, vector_size=num_features, window=context)
    # model.init_sims(replace=True)
    # 保存模型，供日後使用
    # model.wv.save_word2vec_format("./data/pinyin20211127.model"+".bin", binary=True)
    model.wv.save_word2vec_format("RawData/CASIA database/text20211223.model"+".bin", binary=True)
    # 可以在加载模型之后使用另外的句子来进一步训练模型
    # model =  KeyedVectors.load_word2vec_format('./data/pinyin20211127.model.bin', binary=True)
    model =  KeyedVectors.load_word2vec_format('RawData/CASIA database/text20211223.model.bin', binary=True)

    print(model["天空"])
    # print(model[3])
    print(len(model))
    # print( model.vectors)
    # print(model['ni'])
    # model.train(more_sentences)

def textpyTWordvec():
    num_features = 300  # Word vector dimensionality
    min_word_count = 10  # Minimum word count
    num_workers = 0  # Number of threads to run in parallel
    context = 1  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    #pinyinWithoutT
    # sentences = word2vec.LineSentence("./data/pinyinWithoutT.txt")
    sentences = word2vec.LineSentence("RawData/CASIA database/pinyinWithT.txt")

    model = Word2Vec(sentences, workers=num_workers, vector_size=num_features, window=context)
    # model.init_sims(replace=True)
    # 保存模型，供日後使用
    # model.wv.save_word2vec_format("./data/pinyin20211127.model"+".bin", binary=True)
    model.wv.save_word2vec_format("RawData/CASIA database/pyT20220108.model"+".bin", binary=True)
    # 可以在加载模型之后使用另外的句子来进一步训练模型
    # model =  KeyedVectors.load_word2vec_format('./data/pinyin20211127.model.bin', binary=True)
    model =  KeyedVectors.load_word2vec_format('RawData/CASIA database/pyT20220108.model.bin', binary=True)

    print(model["jiu4"])
    # print(model[3])
    print(len(model))
    # print( model.vectors)
    # print(model['ni'])
    # model.train(more_sentences)

def toPinyin():
    from pypinyin import pinyin, lazy_pinyin, Style
    with open('RawData/CASIA database/text.txt', encoding='utf8') as f:
        for line in f:
            line=line.strip('\n')
            pinyin=lazy_pinyin(line, style=Style.TONE3, neutral_tone_with_five=True)
            pinyin=" ".join(i for i in pinyin)
            with open('RawData/CASIA database/pinyinWithT.txt','a',encoding='utf8') as fw:
                fw.write(pinyin)
                fw.write('\n')
            # pinyinlazy=lazy_pinyin(line)
            # pinyinlazy = " ".join(i for i in pinyinlazy)
            # with open('RawData/CASIA database/PinyinwithouT.txt','a',encoding='utf8') as fw1:
            #     fw1.write(pinyinlazy)
            #     fw1.write('\n')
            # print(pinyin)

def fenCi():
    # 分词
    import jieba as jieba
    # s="12410枚金币，这些金币去哪了我都还记得"
    # s=jieba.cut(s)
    # s=" ".join(s)
    # print(s)
    with open('RawData/CASIA database/text.txt',encoding='utf8') as f:
        for line in f:
            line = line.strip('\n')
            line=" ".join(jieba.cut(line))
            with open('RawData/CASIA database/text_jieba.txt', 'a', encoding='utf8') as fw1:
                fw1.write(line)
                fw1.write('\n')

    # s="ai you mei shen me bu hao yi si de ni shi yi ge zhi de tuo fu de ren"
    # pinyin=s.split()
    # print(pinyin)

def getTextFeature():
    import numpy as np
    model= KeyedVectors.load_word2vec_format('RawData/CASIA database/text20211223.model.bin', binary=True)
    index2word_set = set(model.index_to_key)
    def avg_feature_vector(sentence, model, num_features, index2word_set):
        words = sentence.split()
        feature_vec = np.zeros((num_features,), dtype='float32')
        n_words = 0
        for word in words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec
    s=avg_feature_vector("你们 称呼 长辈",model,300,index2word_set)
    print(len(s))


def getTextwithPinyinTFeature():
    import numpy as np
    model= KeyedVectors.load_word2vec_format('RawData/CASIA database/pyT20220108.model.bin', binary=True)
    index2word_set = set(model.index_to_key)
    def avg_feature_vector(sentence, model, num_features, index2word_set):
        words = sentence.split()
        feature_vec = np.zeros((num_features,), dtype='float32')
        n_words = 0
        for word in words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec
    s=avg_feature_vector("你们 称呼 长辈",model,300,index2word_set)
    print(len(s))

if __name__ == "__main__":

    # toPinyin()
    # fenCi()
    # textWordvec()
    # getTextFeature()
    textpyTWordvec()