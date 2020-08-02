# coding=utf-8

import pandas as pd
import fool
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression




stopwords = {}
with open(r'stopword.txt', 'r', encoding='utf-8') as fr:
    for word in fr:
        stopwords[word.strip()] = 0



# 定义类
class clf_model:

    def __init__(self):
        self.model = ""
        self.vectorizer = ""
    def train(self):
        d_train = pd.read_excel("data_train.xlsx")

        d_train.sentence_train = d_train.sentence_train.apply(self.fun_clean)
        print("训练样本 = %d" % len(d_train))

        self.vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b")  # 注意，这里自己指定token_pattern，否则sklearn会自动将一个字长度的单词过滤筛除
        features = self.vectorizer.fit_transform(d_train.sentence_train)
        print("训练样本特征表长度为 " + str(features.shape))

        self.model = LogisticRegression(C=10)
        self.model.fit(features, d_train.label)


    def predict_model(self, sentence):

        if sentence in ["好的", "需要", "是的", "要的", "好", "要", "是"]:
            return 1, 0.8


        sent_features = self.vectorizer.transform([sentence])
        pre_test = self.model.predict_proba(sent_features).tolist()[0]
        clf_result = pre_test.index(max(pre_test))
        score = max(pre_test)
        return clf_result, score


    def predict_rule(self, sentence):

        sentence = sentence.replace(' ', '')
        if re.findall(r'不需要|不要|停止|终止|退出|不买|不定|不订', sentence):
            return 2, 0.8
        elif re.findall(r'订|定|预定|买|购', sentence) or sentence in ["好的","需要","是的","要的","好","要","是"]:
            return 1, 0.8
        else:
            return 0, 0.8


    def fun_clean(self, sentence):


        words, ners = fool.analysis(sentence)

        ners = ners[0].sort(key=lambda x: len(x[-1]), reverse=True)
           if ners:
            for ner in ners:
                sentence = sentence.replace(ner[-1], ' ' + ner[2] + ' ')

        word_lst = [w for w in fool.cut(sentence)[0] if w not in stopwords]
        output_str = ' '.join(word_lst)
        output_str = re.sub(r'\s+', ' ', output_str)
        return output_str.strip()


    def fun_clf(self, sentence):

        sentence = self.fun_clean(sentence)
        clf_result, score = self.predict_model(sentence)
        return clf_result, score



def fun_replace_num(sentence):

    time_num = {"一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","十":"10","十一":"11","十二":"12"}
    for k, v in time_num.items():
        sentence = sentence.replace(k, v)
    return sentence


def slot_fill(sentence, key=None):

    slot = {}
    words, ners = fool.analysis(sentence)
    to_city_flag = 0
    for ner in ners[0]:

        if ner[2]=='time':

            date_content = re.findall(r'后天|明天|今天|大后天|周末|周一|周二|周三|周四|周五|周六|周日|本周一|本周二|本周三|本周四|本周五|本周六|本周日|下周一|下周二|下周三|下周四|下周五|下周六|下周日|这周一|这周二|这周三|这周四|这周五|这周六|这周日|\d{,2}月\d{,2}号|\d{,2}月\d{,2}日', ner[-1])
            slot["date"] = date_content[0] if date_content else ""

            time_content = re.findall(r'\d{,2}点\d{,2}分|\d{,2}点钟|\d{,2}点', ner[-1])

            pmam_content = re.findall(r'上午|下午|早上|晚上|中午|早晨', ner[-1])
            slot["time"] = pmam_content[0] if pmam_content else "" + time_content[0] if time_content else ""

        if ner[2]=='location':

            if key is None:
                if re.findall(r'(到|去|回|回去)%s'%(ner[-1]), sentence):
                    to_city_flag = 1
                    slot["to_city"] = ner[-1]
                    continue
                if re.findall(r'从%s|%s出发'%(ner[-1], ner[-1]), sentence):
                    slot["from_city"] = ner[-1]
                elif to_city_flag==1:
                    slot["from_city"] = ner[-1]

            elif key in ["from_city", "to_city"]:
                slot[key] = ner[-1]

    return slot


def fun_wait(clf_obj):
    print("\n\n\n")
    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")
    print("Starting ...")
    sentence = input("客服：请问需要什么服务？(时间请用12小时制表示）\n")

    clf_result, score = clf_obj.fun_clf(sentence)
    return clf_result, score, sentence


def fun_search(clf_result, sentence):


    name = {"time":"出发时间", "date":"出发日期", "from_city":"出发城市", "to_city":"到达城市"}
    slot = {"time":"", "date":"", "from_city":"", "to_city":""}
    sentence = fun_replace_num(sentence)
    slot_init = slot_fill(sentence)
    for key in slot_init.keys():
        slot[key] = slot_init[key]
    while "" in slot.values():
        for key in slot.keys():
            if slot[key]=="":
                sentence = input("客服：请问%s是？\n"%(name[key]))
                sentence = fun_replace_num(sentence)
                slot_cur = slot_fill(sentence, key)
                for key in slot_cur.keys():
                    if slot[key]=="":
                        slot[key] = slot_cur[key]

    if random.random()>0.5:
        print("客服：%s%s从%s到%s的票充足"%(slot["date"], slot["time"], slot["from_city"], slot["to_city"]))
        return 1
    else:
        print("客服：%s%s从%s到%s无票" % (slot["date"], slot["time"], slot["from_city"], slot["to_city"]))
        print("End !!!")
        print("-------------------------------------------------------------")
        print("-------------------------------------------------------------")

        return 0


def fun_book():

    print("客服：已为您完成订票。\n\n\n")
    print("End !!!")
    print("-------------------------------------------------------------")
    print("-------------------------------------------------------------")



if __name__=="__main__":

    clf_obj = clf_model()
    clf_obj.train()
    threshold = 0.55
    while 1:
        clf_result, score, sentence = fun_wait(clf_obj)

        if score<threshold or clf_result==2:
            continue


        else:
            search_result = fun_search(clf_result, sentence)
            if search_result==0:
                continue
            else:
                sentence = input("客服：需要为您订票吗？\n")

                clf_result, score = clf_obj.fun_clf(sentence)

                if clf_result == 1:
                    fun_book()
                    continue

