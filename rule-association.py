# -*- coding: utf-8 -*-
# http://artedosdados.blogspot.com.br/2015/02/regras-de-associacao-em-python-modulo.html

import argparse
import pandas as pd
import itertools

from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--DB_PATH", required=False, type=str, default="database-07.csv")
parser.add_argument("--ATT_PATH", required=False, type=str, default="attr-07.csv")
parser.add_argument("--SUPP", required=False, type=float, default=0.03)
parser.add_argument("--CONF", required=False, type=float, default=0.5)
parser.add_argument("--LIFT", required=False, type=float, default=1.0)
args = parser.parse_args()

DB_PATH = args.DB_PATH
ATT_PATH = args.ATT_PATH
SUPP = args.SUPP
CONF = args.CONF
LIFT = args.LIFT

# Faz a leitura dos atributos do dataset
def readAttributes(filePath):
    file = open(filePath, 'r')
    return file.read().split(',')

# Faz a leitura do dataset
def readDataset(filePath, attributes):
    return pd.read_csv(filePath, names=attributes)

# Retorna a partir do dataset as transações realizadas
def parseTransactions(dataSet, attributes):
    transactionList = {}
    transKey = 0
    
    for row in dataSet.values:
        transaction = []
        
        for i in range(0, len(row)):
            # Verifica se o item foi comprado na transação
            if row[i] == 1:
                # Guarda o tipo do item
                transaction.append(attributes[i])
                
        # Se a transação teve itens comprados
        if len(transaction) > 0:
            # Guarda os itens da transação
            transactionList.update({ transKey:set(transaction) })
            transKey += 1
        
    return transactionList

# Retorna os itens contidos nas transações
def parseItems(transactions):
    itemsList = []
    
    for items in transactions.values():
        for item in items:
            itemsList.append(item)
    
    return set(itemsList)

# Obtém a frequência de um item
def getItemFrequency(item, trans):
    itemCount = []
    
    # Para todas as transações
    for key, items in trans.items():
        # Verifica se a transação possui a combinação dos itens
        if set(items).intersection(set(item)) == set(item):
            # Adiciona na lista de itens
            itemCount.append(item)
            
    return len(itemCount)

# Calcula o suporte da regra
def getRuleSupport(rule, trans):
    # A=>B para { A, B }
    ruleAB = set(rule[0]) | set(rule[1])
    # Obtem a frequência de { A, B }
    itemFreq = getItemFrequency(ruleAB, trans)
    # Suporte = freq({A,B})/N
    return itemFreq / len(trans)

# Calcula a confiança da regra
def getRuleConfidence(rule, trans):
    # A=>B para { A, B }
    ruleAB = set(rule[0]) | set(rule[1])
    # Obtém a frequência de { A, B }
    itemABFreq = getItemFrequency(ruleAB, trans)
    # Obtém a frequência de { A }
    itemAFreq = getItemFrequency(rule[0], trans)
    # Confiança = freq({A,B})/freq({A})
    return itemABFreq / itemAFreq

# Calcula a correlação Lift da regra
def getRuleLift(ruleB, ruleConf, trans):
    # Obtém a frequência de {B}
    itemBFreq = getItemFrequency(ruleB, trans)
    # Obtém o suporte de {B}=freq({B})/N
    suppB = itemBFreq / len(trans)
    # Calcula a correlação Lift=conf({A,B})/supp({B})
    return ruleConf / suppB

# Retorna os itens frequentes de tamanho N
def getFrequentItems(items, trans, tam, supp):
    # Faz a combinação dos itens com tamanho N
    itemCombination = set(itertools.combinations(items, tam))
    itemTransactions = []
    
    for comb in itemCombination:
        # Para todas as transações
        for key, items in trans.items():
            # Verifica se a transação possui a combinação dos itens
            if set(items).intersection(set(comb)) == set(comb):
                # Adiciona na lista de itens
                itemTransactions.append(comb)

    freqItems = []
    
    # Verifica a frequência dos itens
    for item, freq in sorted(Counter(itemTransactions).items()):
        # Verifica se a frequência do item é maior do que o suporte
        if freq >= (supp * len(trans)):
            freqItems.append([item, freq])
            
    return dict(freqItems)

# Algoritmo Apriori para mineração de regras de associação
def apriori(items, trans):
    print("\nObtendo itens frequentes...")
    
    allFreqItemList = {}
    cand_items = items
    
    for tam in range(1, len(items)):
        print("Tamanho " + str(tam) + "...")
        freqItem = getFrequentItems(cand_items, trans, tam, SUPP)
        #print(freqItem)
        print("Encontrados=", len(freqItem))
        
        if(len(freqItem) > 0):
            allFreqItemList.update(freqItem)
        else:
            break
        
        new_item = []
        
        for itemKey, freq in freqItem.items():
            for item in itemKey:
                new_item.append(item)
            
        cand_items = set(new_item)
        
    tmpAllFreqItems = []
    
    for item in allFreqItemList:
        tmpAllFreqItems.append(item)
        
    print("Total de itens frequentes: ", len(tmpAllFreqItems))

    print("\nCriando regras de associação...")
    
    tmpRules = set(itertools.combinations(tmpAllFreqItems, 2))
    rules = []
    
    for rule in tmpRules:
        # Verifica se não há intersecção entre A => B
        if len(set(rule[0]).intersection(rule[1])) == 0:
            ruleSupp = getRuleSupport(rule, trans)
            # Verifica se a regra atende ao parâmetro de suporte
            if ruleSupp >= SUPP:
                ruleConf = getRuleConfidence(rule, trans)
                # Verifica se a regra atende ao parâmetro de confiança
                if ruleConf >= CONF:
                    ruleLift = getRuleLift(rule[1], ruleConf, trans)
                    # Verifica se a regra atende ao parâmetro de lift
                    if ruleLift >= LIFT:
                        rules.append([rule, ruleSupp, ruleConf, ruleLift])

    print("Total de regras=",len(rules))

    # Imprime as regras geradas
    for rule in rules:
        print(set(rule[0][0]), "->", set(rule[0][1]), "[ SUPP=" + str(rule[1]) + ", CONF="+ str(rule[2]) + ", LIFT="+ str(rule[3]) + " ]")

# Carrega os atributos da base de dados
attributes = readAttributes(ATT_PATH)
print("Total de atributos: " + str(len(attributes)))

# Carrega a base de dados
dataSet = readDataset(DB_PATH, attributes)
print("Total de dados: " + str(len(dataSet)))

# Faz a leitura da base de dados para o formato de transações
transactions = parseTransactions(dataSet, attributes)
#print(transactions)
print("Total de transações: " + str(len(transactions)))

# Obtém os itens presentes nas transações
items = parseItems(transactions)
#print(items)
print("Total de itens das transações: " + str(len(attributes)))

# Executa o algoritmo Apriori
apriori(items, transactions)