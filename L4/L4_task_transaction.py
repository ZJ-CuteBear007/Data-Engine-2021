# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 20:57:52 2021
@author: ZhangJian

"""

import pandas as pd

def main():
    raw_data = pd.read_csv('./Market_Basket_Optimisation.csv', header = None)
    transactions=CleanData(raw_data)
    #Correlation_Analysis1(transactions)#Correlation_Analysis1：利用efficient_aprior库
    Correlation_Analysis2(transactions)#Correlation_Analysis2：利用mlxtend库进行

#数据清洗函数
def CleanData(raw_data):
    transactions=[]
    for i in range(0,raw_data.shape[0]):#行循环
        record=[]
        for j in range(0,raw_data.shape[1]):#列循环，一列是一条记录
            if str(raw_data.values[i,j])!='nan':
                record.append(str(raw_data.values[i,j]))
        transactions.append(record)
    return transactions

#Aprion算法关联分析函数-挖掘频繁项集和频繁规则

#Correlation_Analysis1：利用efficient_aprior库
def Correlation_Analysis1(transactions):
    from efficient_apriori import apriori
    itemsets, rules = apriori(transactions, min_support=0.02, min_confidence=0.4)
    print("频繁项集：", itemsets)
    print("关联规则：", rules)

#Correlation_Analysis2：利用mlxtend库进行
def Correlation_Analysis2(transactions):
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    from mlxtend.preprocessing import TransactionEncoder
    # one-hot编码
    te = TransactionEncoder()
    hot_encoded = te.fit(transactions).transform(transactions)
    transactions_temp = pd.DataFrame(hot_encoded, columns=te.columns_)
    itemsets = apriori(transactions_temp, min_support=0.04, use_colnames=True)
    itemsets = itemsets.sort_values(by='support', ascending=False)
    rules = association_rules(itemsets, metric='lift', min_threshold=1.1)
    rules = rules.sort_values(by='lift', ascending=False)  # 从大到小排序
    print("频繁项集：", itemsets)
    print("关联规则：", rules)

if __name__ == '__main__':
    main()