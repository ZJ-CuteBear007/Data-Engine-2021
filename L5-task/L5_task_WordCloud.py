# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 20:57:52 2021
@author: ZhangJian

"""

import pandas as pd
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import numpy
from PIL import Image
import matplotlib.pyplot as plt

#数据清洗函数
def CleanData(raw_data):
    item_count={}
    transactions=[]
    for i in range(raw_data.shape[0]):#行循环
        record=[]
        for j in range(raw_data.shape[1]):#列循环，一列是一条记录
            item = str(raw_data.values[i, j])
            if item!='nan':
                record.append(item)
                if item not in item_count:
                    item_count[item]=1
                else:
                    item_count[item]+=1
        transactions.append(record)
    return transactions,item_count

def remove_stop_words(f):
    stop_words=["fat","low"]
    for stop_word in stop_words:
        f.replace(stop_word,'')
    return f

def create_word_cloud(f):
    #设置背景图片
    color_mask = numpy.array(Image.open('car.png'))
    f = remove_stop_words(f)
    cut_text = word_tokenize(f)
    #print(cut_text)
    cut_text = " ".join(cut_text)
    wc = WordCloud(
        background_color="white",
        mask=color_mask,#设置背景图片
		max_words=120,
		width=2500,
		height=1500,
    )
    wordcloud = wc.generate(cut_text)
    # 写词云图片
    wordcloud.to_file("wordcloud.jpg")
    plt.imshow(wordcloud )
    plt.axis('off')
    plt.show()
raw_data = pd.read_csv('./Market_Basket_Optimisation.csv', header = None)
transactions,item_count=CleanData(raw_data)
#生成词云
all_word = " ".join('%s' %item for item in transactions)
create_word_cloud(all_word)
print("词云生成完毕！")
#To10的产品

print(sorted(item_count.items(),key=lambda x:x[1],reverse=True)[:10])

