# -*- coding: utf-8 -*-
# http://artedosdados.blogspot.com.br/2015/02/regras-de-associacao-em-python-modulo.html

import argparse
import pandas as pd
import itertools

from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--DB_PATH", required=False, type=str, default="database-test.csv")
parser.add_argument("--ATT_PATH", required=False, type=str, default="attributes.csv")
parser.add_argument("--SUPP", required=False, type=float, default=0.01)
parser.add_argument("--CONF", required=False, type=float, default=0.5)
args = parser.parse_args()

DB_PATH = args.DB_PATH
ATT_PATH = args.ATT_PATH
SUPP = args.SUPP
CONF = args.CONF

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

def getItemFrequency(item, trans):
    itemCount = []
    
    # Para todas as transações
    for key, items in trans.items():
        # Verifica se a transação possui a combinação dos itens
        if set(items).intersection(set(item)) == set(item):
            # Adiciona na lista de itens
            itemCount.append(item)
            
    return len(itemCount)

def getRuleSupport(rule, trans):
    ruleAB = set(rule[0]) | set(rule[1])
    itemFreq = getItemFrequency(ruleAB, trans)
    return itemFreq / len(trans)

def getRuleConfidence(rule, trans):
    ruleAB = set(rule[0]) | set(rule[1])
    
    itemABFreq = getItemFrequency(ruleAB, trans)
    itemAFreq = getItemFrequency(rule[0], trans)
    
    return itemABFreq / itemAFreq

def getRuleLift(ruleB, ruleConf, trans):
    itemBFreq = getItemFrequency(ruleB, trans)
    suppB = itemBFreq / len(trans)
    
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
        
def apriori(items, trans):
    allFreqItemList = {}
    cand_items = items
    
    print("\nObtendo itens frequentes...")
    
    for tam in range(1, len(items)):
        print("Tamanho " + str(tam) + "...")
        freqItem = getFrequentItems(cand_items, trans, tam, SUPP)
        print("Encontrados=",len(freqItem))
        
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
            
            if ruleSupp >= SUPP:
                ruleConf = getRuleConfidence(rule, trans)
                
                if ruleConf >= CONF:
                    ruleLift = getRuleLift(rule[1], ruleConf, trans)
                    rules.append([rule, ruleSupp, ruleConf, ruleLift])
                    
    print("Total de regras=",len(rules))
            
    for rule in rules:
        print(set(rule[0][0]), "->", set(rule[0][1]), "[ SUPP=" + str(rule[1]) + ", CONF="+ str(rule[2]) + ", LIFT="+ str(rule[3]) + " ]")

attributes = readAttributes(ATT_PATH)
print("Total de atributos: " + str(len(attributes)))

dataSet = readDataset(DB_PATH, attributes)
print("Total de dados: " + str(len(dataSet)))

transactions = parseTransactions(dataSet, attributes)
print("Total de transações: " + str(len(transactions)))

items = parseItems(transactions)
print("Total de itens das transações: " + str(len(attributes)))

apriori(items, transactions)