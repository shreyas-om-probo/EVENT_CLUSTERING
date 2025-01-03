import gensim.downloader as api
import numpy as np
import pickle

def load_encoder():
    word2vec_model = api.load("word2vec-google-news-300")
    return word2vec_model

def load_knn(path):
    with open(path, 'rb') as handle:
        knn = pickle.load(handle)
    return knn

def sentence_to_vec(sentence,word2vec_model):
    words = sentence.split()
    word_vecs = [word2vec_model[word] for word in words if word in word2vec_model]
    if word_vecs:
        return np.mean(word_vecs, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def classify_knn(knn, sentence,word2vec_model):
    sentence = sentence_to_vec(sentence,word2vec_model)
    category = knn.predict([sentence])
    return category[0]

if __name__ == "__main__":
    sentence = "TRXUSDT to be priced at 0.12344 USDT or more at 01:00 PM?. Tether's USDT that is issued on the TRON network. It is a technical standard token, and It works based on TRON's network\n"
    print(sentence)
    encoder = load_encoder()
    knn = load_knn('KNN_CLASSIFIER.pickle')
    label = classify_knn(knn,sentence,encoder)
    print(label)