#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install mlxtend


# In[4]:


import time
import pandas as pd
from math import comb
from itertools import combinations
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth, association_rules


# In[5]:


def load_and_display_dataset(choice):
    dataset_paths = {
        1: 'C:/Users/kanna/Desktop/Project/amazon.csv',
        2: 'C:/Users/kanna/Desktop/Project/bestbuy.csv',
        3: 'C:/Users/kanna/Desktop/Project/kmart.csv',
        4: 'C:/Users/kanna/Desktop/Project/nike.csv',
        5: 'C:/Users/kanna/Desktop/Project/generic.csv'
    }

    try:
        if choice in dataset_paths:
            df = pd.read_csv(dataset_paths[choice])
            return df
        else:
            print("Invalid choice. Please select a number between 1 and 5.")
            return None
    except FileNotFoundError:
        print(f"File not found for choice {choice}. Please check the file path and try again.")
        return None

try:
    choice = int(input("Please, Select your Dataset for \n 1 Amazon.\n 2 BestBuy.\n 3 K-Mart.\n 4 Nike.\n 5 Generic. \n"))
    df = load_and_display_dataset(choice)
    if df is not None:
        print(df)
except ValueError:
    print("Please enter a valid integer.")


# In[10]:


min_sup = input("Please, input your Min. Support \n")
min_sup = float(min_sup)
min_con = input("Please, input your Min. confidence \n")
min_con = float(min_con)


# In[12]:


unique_transactions = df['Transaction ID'].unique()
transaction_items = df['Transaction'].tolist()

transactions = transaction_items

def frequent_items(new_patterns, current_items):
    items_in_patterns = set(item for pattern in new_patterns for item in pattern)
    return [item for item in current_items if item in items_in_patterns]

def find_frequent_patterns(transactions, min_support):
    unique_items = set(item for sublist in transactions for item in sublist)
    pattern_size = 1
    frequent_patterns = []
    frequent_patterns_count = []
    current_frequent_items = list(unique_items)
    while current_frequent_items:
        potential_patterns = combinations(current_frequent_items, pattern_size)
        new_frequent_patterns = []
        for pattern in list(potential_patterns):
            count = sum(1 for transaction in transactions if set(pattern).issubset(set(transaction)))
            if count >= min_support * len(transactions):
                new_frequent_patterns.append(pattern)
                frequent_patterns_count.append(count)
        frequent_patterns.extend(new_frequent_patterns)
        pattern_size += 1
        current_frequent_items = frequent_items(new_frequent_patterns, current_frequent_items)
    return frequent_patterns, frequent_patterns_count

def generate_association_rules(frequent_patterns, frequent_patterns_count, transactions, min_confidence):
    rules_with_confidence = []
    for pattern, pattern_count in zip(frequent_patterns, frequent_patterns_count):
        if len(pattern) > 1:
            sub_patterns = [sub_pattern for i in range(1, len(pattern))
                            for sub_pattern in combinations(pattern, i)]
            for sub_pattern in sub_patterns:
                sub_pattern_count = sum(1 for transaction in transactions if set(sub_pattern).issubset(set(transaction)))
                if sub_pattern_count > 0:  # Avoid division by zero
                    confidence = pattern_count / sub_pattern_count
                    if confidence >= min_confidence:
                        consequence = set(pattern) - set(sub_pattern)
                        rules_with_confidence.append(((tuple(sub_pattern), tuple(consequence)), confidence))
    return rules_with_confidence

def format_rules_for_printing(rules_with_confidence):
    formatted_rules = []
    for (antecedent, consequent), confidence in rules_with_confidence:
        rule_string = f"{antecedent} ---> {consequent} with confidence = {confidence:.2f}"
        formatted_rules.append(rule_string)
    return formatted_rules
start_time = time.time()
frequent_patterns, frequent_patterns_count = find_frequent_patterns(transactions, min_sup)
rules_with_confidence = generate_association_rules(frequent_patterns, frequent_patterns_count, transactions, min_con)
end_time = time.time()
bruteapriori_runtime = end_time - start_time

formatted_rules = format_rules_for_printing(rules_with_confidence)

def print_frequent_patterns_and_rules(frequent_patterns, frequent_patterns_count, transactions, min_confidence,formatted_rules):
    print("Frequent patterns:\n")
    for pattern, count in zip(frequent_patterns, frequent_patterns_count):
        print(f"{pattern}, support: {count/len(transactions):.2f}")
    print('\nAssociation rules:')
    for rule in formatted_rules:
        print(rule)

print_frequent_patterns_and_rules(frequent_patterns, frequent_patterns_count, transactions, min_con,formatted_rules)

print(f"Brute-forced Apriori runtime: {bruteapriori_runtime} seconds")


# In[13]:


te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

start_time = time.time()

frequent_itemsets = apriori(df_encoded, min_support=min_sup, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_con)

end_time = time.time()
apriori_runtime = end_time - start_time

def display_output_like_brute_force(frequent_itemsets, rules):
    print("Frequent patterns:\n")
    for index, row in frequent_itemsets.iterrows():
        print(f"{list(row['itemsets'])}, support: {row['support']}")

    print("\nAssociation rules:")
    for index, row in rules.iterrows():
        print(f"{list(row['antecedents'])} ---> {list(row['consequents'])} with confidence = {row['confidence']:.2f}")

display_output_like_brute_force(frequent_itemsets, rules)

print(f"Apriori runtime: {apriori_runtime} seconds")


# In[14]:


start_time = time.time()

frequent_itemsets_fp = fpgrowth(df_encoded, min_support=0.1, use_colnames=True)

rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.1)

end_time = time.time()
fpgrowth_runtime = end_time - start_time

def display_output_like_brute_force(frequent_itemsets, rules):
    print("Frequent patterns:\n")
    for index, row in frequent_itemsets.iterrows():
        print(f"{list(row['itemsets'])}, support: {row['support']}")

    print("\nAssociation rules:")
    for index, row in rules.iterrows():
        print(f"{list(row['antecedents'])} ---> {list(row['consequents'])} with confidence = {row['confidence']:.2f}")

display_output_like_brute_force(frequent_itemsets_fp, rules_fp)

print(f"FP-Growth runtime: {fpgrowth_runtime} seconds")


# In[15]:


data = {
    "Algorithm": ["BruteApriori", "Apriori", "FPGrowth"],
    "Runtime": [bruteapriori_runtime, apriori_runtime, fpgrowth_runtime]
}

df = pd.DataFrame(data)
df_sorted = df.sort_values(by="Runtime", ascending=True)
print(df_sorted)


# In[ ]:




