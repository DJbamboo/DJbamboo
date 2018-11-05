import re
from konlpy.tag import Twitter
import math
import heapq, random
import numpy as np
import pickle

def read_data():
    global data
    data = {}
    path = './data/'
    file_names = ['fam','lov','sch', 'soc', 'topic1_family', 'topic2_school', 'topic3_love', 'topic4_society', 'word_vec']
    file_name_extension = 'pic'
    for file_name in file_names:
        s = path + file_name + '.' + file_name_extension
        with open(s, 'rb') as f:
            data[file_name] = pickle.load(f)
    return None

pos_tagger = Twitter()
def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(str(doc), norm=True, stem=True)]

def word_count(docs):
    topic = [
        set(["엄마/Noun","아빠/Noun","아버지/Noun","어머니/Noun","할머니/Noun","부모님/Noun","동생/Noun","가족/Noun","아들/Noun","집안/Noun","자식/Noun","결혼/Noun","이혼/Noun","사촌/Noun"]),
        set(["선배/Noun","새내기/Noun","후배/Noun","동기/Noun","동아리/Noun","행사/Noun","인사/Noun","술자리/Noun","학생회/Noun","학교/Noun","학년/Noun","입학/Noun","활동/Noun","술/Noun","개강/Noun","밥약/Noun","꼰대/Noun","존댓말/Noun","학번/Noun","학우/Noun","존댓말/Noun","학번/Noun","신입생/Noun"]),
        set(["사랑/Noun","마음/Noun","행복/Noun","감정/Noun","추억/Noun","상처/Noun","이별/Noun","서로/Noun","연애/Noun","벚꽃/Noun","미안/Noun","후회/Noun","마지막/Noun","소중/Noun","미소/Noun","표현/Noun","따뜻/Noun","첫사랑/Noun","웃음/Noun","곰신/Noun","고백/Noun","성격/Noun","사이/Noun","서운/Noun","남자친구/Noun","여자친구/Noun"]),
        set(["사회/Noun","문제/Noun","여성/Noun","이유/Noun","의견/Noun","동성애/Noun","잘못/Noun","종교/Noun","정치/Noun","집단/Noun","혐오/Noun","행위/Noun","차별/Noun","주장/Noun","가치관/Noun","정치/Noun","소수자/Noun","자유/Noun","발언/Noun"])
    ]
    count = [0,0,0,0]    
    for word in docs:
        for i in range(4):
            if word in topic[i]:
                count[i] += 1
                break
    return count ##max값이 2개면;;?

def prepro(s):
    hangul = re.compile('[^ |가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
    result = hangul.sub('', s) # 한글과 띄어쓰기를 제외한 모든 부분을 제거
    return(result)

def dot_product(v1, v2):
    return sum(v1*v2)

def cosine_measure(v1, v2):
    return dot_product(v1, v2) / (math.sqrt(dot_product(v1, v1)) * math.sqrt(dot_product(v2, v2)))

def Djbamboo(sa):
    #75 set {{docs}}
    docs = tokenize(sa)
    
    #77~108 STEP1 : word count & find max position
    max_index = np.argmax(word_count(docs))
    cate = ['fam', 'sch', 'lov', 'soc'][max_index]
    topic = ['topic1_family', 'topic2_school', 'topic3_love', 'topic4_society'][max_index]

    #111 ~ 117 STEP2 : docs to word2vec position
    rex = [s2 for s2 in [prepro(s) for s in docs] if s2!='']

    #119~122
    ind = []
    for r in rex:
        if r in data['word_vec'].keys():
            ind.append(data['word_vec'][r])

    ind = np.array(ind, dtype = 'float64')
    savec = sum(ind[0:len(ind)-1])/len(ind) ##마지막 값 뺀 이유???????????????

    # after this line, we use {{max_index}} and {{savec}}
    # 133 ~ 158
    tem = []
    for i in range(1,len(data[cate])): ##ERROR가 난다면 try except -> append(0)
        tem.append(cosine_measure(savec, data[cate][i]))
    
    #160 ~ 165
    title = slice(1,2)
    artist = slice(2,3)
    idx = np.argsort(tem)
    topN = 3
    reco = []
    overlap = set() # 이 변수가 필요한 이유는...같은 노래가 다른 songid로 존재할 경우가 있음.(싱글로 내고 정규에서 또 발매하는경우)
    for i in idx[::-1]:
        if topN==len(reco):
            break
        if math.isnan(tem[i]):
            continue
        if (data[topic][i+1][title][0], data[topic][i+1][artist][0]) in overlap:
            continue
        overlap.add((data[topic][i+1][title][0], data[topic][i+1][artist][0]))
        reco.append(i+1) ##+1을 지워야하나 말아야하나
    # reco1 = tem.index(heapq.nlargest(10,tem)[0])+1 #왜 +1?
    rst = {}
    for i in range(topN):
        rst['song' + str(i+1)] = data[topic][reco[i]][title][0]
        rst['name' + str(i+1)] = data[topic][reco[i]][artist][0]
    return(rst)

if __name__ == '__main__':
    read_data()
    #sa = '연대 숲 고민이 있어요. ㅠㅜㅠ 요즘 따라 연락을 하고 지내는 남자가 있어요. 원래 알던 사이이지만 방학을 한 거 나서 얼굴을 못 보게 되면서 카톡을 자주 하고 있거든요 밤에만 연락을 하게 돼요 바쁘긴 하지만 카톡을 전혀 오 가지 않고 저녁 이후부터 잠잘 때 까지만, 연락을 하고 있어요……. 카톡 내용은 뭔가…. 솔직히 남이 보면 호감이라고 느낄 것 같아요. 그렇지만 친구들 말에 의하면 저녁 6시 이후에 연락이 되는 남자는 그냥 외로워서 그런 거라나 맞는 말인 것 같아서 고민이 많네요!! 여러분도 밤에만 연락이 되는 남자는 제가 좋다기보다는 그냥 외로워서라고 생각하시나요!!?'
    sa = '며칠 아니 몇주동안 준비한 과제가 있었는데 몇초차이로 제출을 못했어요 너무 속상해서 계속 눈물이 나네요 죽고싶어요'
    print(Djbamboo(sa))