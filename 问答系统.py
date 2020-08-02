import json
from collections import defaultdict
from matplotlib import pyplot as plt
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
import numpy as np
import re
from scipy.spatial.distance import cosine as cosdist
import nltk
nltk.download('wordnet')
nltk.download ('stopwords')
qa_corpus_path = ''
glove_path = ''
def read_corpus(corpus_path):
    qlist,alist = [],[]
    with open(corpus_path,'r') as in_file:
        json_corpus = json.load(in_file)['data']
    for article in json_corpus:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                for ans in qa['answers']:
                    qlist.append(qa['question'])
                    alist.append(ans['text'])
    assert len(qlist) == len(alist)
    return qlist,alist

def get_dict(textlist):
    word_dict = defaultdict(lambda:0)
    for text in textlist:
        for token in text.split(""):
            word_dict[token] +=1
    word_dict = sorted(word_dict.items(),key = lambda item:item[1],reverse = True)
    return dict(word_dict)

def get_topk(n,word_dict):
    res = []
    for word , freq in word_dict.items():
        res.append('{}({})').format(word,freq)
        n -= 1
        if n ==0:
            return res
 #统计一下在qlist 共出现了多少个单词？总共出现了多少个不同的单词
 #这里需要做简单的分词，英文用空格
qlist,alist = read_corpus(qa_corpus_path)
q_dict = get_dict(qlist)
word_total_q = sum(q_dict.values())
n_distinctive_words_q = len(q_dict)
print('There are {} words and {} distinctive tokens in question texts'.format(word_total_q,n_distinctive_words_q))
print(word_total_q)


#todo :统计一下qlist中每个单词出现频率，并把这些频率排一下序
#使用matplotlib里的plot函数，y是词频

plt.bar(np.arange(10000),list(q_dict.value())[100:10100])
plt.ylabe('Frequency')
plt.xlabe('Word Order')
plt.title('Word Frequencies of the Question Corpus')
plt.show()


a_dict = get_dict(alist)
print('The 10 frequentist words in question list (qlist) are :\n{}'.format('|'.join(get_topk(10,q_dict))))
class TextNormalizer:
    def __init__(self,stopwords,filter_vocab,re_cleaners):
        self.lemmatizer = WordNetLemmatizer()
        self.filter_vocab = filter_vocab
        self.stopwords = stopwords
        self.re_cleaners = re_cleaners

    def get_wordnet_pos(self,treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''

    def normalize_texts(self,text_list):
        return [self.normalize_text(text) for text in text_list]

    def normalize_text(self,text):
        for re_pattern in self.re_cleaners:
            text = re.sub(re_pattern,'',text)
        pos_tokens = pos_tag(word_tokenize(text))

        lemmatized = []
        for w,pos in pos_tokens:
            if not w or w in self.stopwords or w in self.filter_vocab:
                continue
            if pos == 'CD':
                lemmatized.append('#NUM')
                continue
            wn_pos = self.get_wordnet_pos(pos)
            if wn_pos:
                lemmatized.append(self.lemmatizer.lemmatize(w,pos=wn_pos))
            else:
                lemmatized.append(self.lemmatizer.lemmatize(w))
        return ''.join(lemmatized).lower()

#todo:把qlist中的没一个问题字符串装换成tf-idf向量，转换之后的结果存储在X矩阵里，X的大小是：N*D的矩阵。这里N是问题的个数（样本个数）
#D是字典库的大小

vectorizer = TfidfVectorizer()#定一个tf-idf的vectorizer
X =vectorizer.fit_transform(qlist)# #结果放在X矩阵


#X矩阵有什么特点？计算一下他的稀疏度
sparsity = np.divide(np.prod(X.shape) - len(X.nonzero()),np.prod(X.shape))
print(sparsity)# 用sparse matrix来存储



#todo 对于用户输入的问题，找到相似度最高的top5问题，并把5个潜在的答案做返回


def top5results(input_q,K=5):
    if not input_q or type(input_q) != type(''):
        print('input error! Please input avalid query string!')
        return
    input_q = TextNormalizer.normalize_text(input_q)
    q_vec = vectorizer.transform([input_q]).todense()
    top_k_indices, top_idxs = [],[]#top-idxs存放相似度最高的（存在qlist里的）问题的下表
    for i in range(X.shape[0]):
        similarity = 1 - cosdist(X[i,:].todense(),q_vec)
        similarity = 0 if np.isnan(similarity) else similarity
        if len(top_k_indices) == K:
            heapq.heappushpop(top_k_indices,(similarity,i))
        else:
            heapq.heappushpop(top_k_indices,(similarity,i))

        top_inxs = sorted(top_k_indices,reverse=True)
        _,top_idxs = zip(*top_idxs[::-1])
        return [alist[id] for id in top_idxs] #返回相似度最高的问题的对应的答案，作为top5答案

print(top5results('Who is president of the United Ststes?'))
print(top5results('What is Shanghai famous for?'))




