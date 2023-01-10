"""
用于训练词向量
"""
from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import TSNE

def cosine_similarity(u, v):
    """
    u与v的余弦相似度反映了u与v的相似程度

    参数：
        u -- 维度为(n,)的词向量
        v -- 维度为(n,)的词向量

    返回：
        cosine_similarity -- 由上面公式定义的u和v之间的余弦相似度。
    """
    distance = 0

    # 计算u与v的内积（点积）
    dot = np.dot(u, v)  # 得到一个数

    norm_u = np.sqrt(np.sum(np.power(u, 2)))
    norm_v = np.sqrt(np.sum(np.power(v, 2)))

    cosine_similarity = np.divide(dot, norm_u * norm_v)  # dot / (norm_u * norm_v)结果一样

    return cosine_similarity


def main():
    num_features = 300  # Word vector dimensionality
    min_word_count = 10  # Minimum word count
    num_workers = 0  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    #pinyinWithoutT
    # sentences = word2vec.LineSentence("./data/pinyinWithoutT.txt")
    sentences = word2vec.LineSentence("./data/pinyinWithT.txt")

    model = Word2Vec(sentences, workers=num_workers, vector_size=num_features, window=context)
    # model.init_sims(replace=True)
    # 保存模型，供日後使用
    # model.wv.save_word2vec_format("./data/pinyin20211127.model"+".bin", binary=True)
    model.wv.save_word2vec_format("./data/pinyinT20211127.model"+".bin", binary=True)



    # 可以在加载模型之后使用另外的句子来进一步训练模型
    model1 =  KeyedVectors.load_word2vec_format('./data/pinyin20211127.model.bin', binary=True)
    model =  KeyedVectors.load_word2vec_format('./data/pinyinT20211127.model.bin', binary=True)
    xiang=model1["xiang"]
    xiang3=model["xiang3"]
    # print("xiang",model1["xiang"])
    # print("xiang3",model["xiang3"])
    # print("cosine_similarity(father, mother) = ", cosine_similarity(model1["xiang"], model["xiang3"]))
    # print(model[3])
    # print(len(model))
    # print( model.vectors)
    # print(model['ni'])
    # model.train(more_sentences)
    vecs=[]
    label=[]
    vecs.append(xiang)
    vecs.append(xiang3)
    for values in vecs:
        label.append(np.argmax(values))
    fea=TSNE


if __name__ == "__main__":
    main()