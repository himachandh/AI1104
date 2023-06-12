import pandas as pd #importing pandas module

#1st sub question

#loading the shopping.csv file into a data frame
df = pd.read_csv('/home/chandu/Downloads/Shopping.csv') 

print(df) #printing the dataframe

#dropping the no of items column as it is including in furthur questions


#2nd sub question
count = 0
dictionary = {} #taking empty dictionary initially
for i, j in df.iterrows(): #taking all the rows in dataframe to obtain list of items 
    transaction_ID = 'T' + str(i+1) #format of transaction id
    items = j.dropna().tolist()[1:] #converting the items in a transaction into a list
    dictionary[transaction_ID] = items #adding the key value pairs to dictionary taking transaction_ID as key and items as value
    count += 1
    print(f"{transaction_ID}: {items}")  # Print each key-value pair
    if count >= 5:
        break
df = df.drop(columns=['#Item(s) purchased'])
#3rd sub question

import re #importing re module

#taking function to convert the word to a single format
def convert_item(item): 
    # Checking if the item is a string or bytes-like object
    if isinstance(item, str): 
     item = re.sub(r"\s+", '_', item) #replacing white space with _
    return item

df = df.applymap(convert_item) #the funtion is applied on each element of dataframe
dictionary = {} #taking empty dictionary initially
for i, j in df.iterrows(): #taking all the rows in dataframe to obtain list of items 
    transaction_ID = 'T' + str(i+1)  #format of transaction id
    items = j.tolist() #converting the items in a transaction into a list
    dictionary[transaction_ID] = items #adding the key value pairs to dictionary taking transaction_ID as key and items as value

#4th sub question

#function to print the items in a particular transaction_ID
def print_items(transaction_ID):
    if transaction_ID in dictionary: #taking the ID which is in dictionary
        items_purchased = [item for item in dictionary[transaction_ID] if pd.notnull(item)]  # Exclude NaN values
       #taking the items 
        print(f"Items purchased in transaction {transaction_ID}:")
        for item in items_purchased:
            print(item) #printing the list of items
    else:
        print(f"Transaction {transaction_ID} does not exist.")

# Call the function for the specified transaction IDs
transaction_ids = ['T32', 'T68', 'T78']
for transaction_ID in transaction_ids:
    print_items(transaction_ID)

#5th sub question
# Count the occurrences of each item
item_counts = df.stack().value_counts()

# Get the total number of transactions
N_total = len(df)

# Define the support thresholds
support_thresholds = [0.5, 1, 2, 3, 5, 10]

# Initialize the table
table = pd.DataFrame(index=support_thresholds, columns=['Support', 'No. of items'])

# Generate the table for different support thresholds
for support in support_thresholds:
    # Filter the items based on the support threshold
    frequent_items = item_counts[item_counts / N_total > support / 100]
    no_of_items = len(frequent_items)
    table.loc[support] = [f'support@{support}', no_of_items]

# Print the table
print(table)

# List the frequent one-itemsets for support@3
support_3_items = item_counts[item_counts / N_total > 3 / 100]
print(f"Frequent one-itemsets for support@3:\n{support_3_items}")

#6th sub question
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# Load the shopping.csv file into a DataFrame
df = pd.read_csv('/home/chandu/Downloads/Shopping.csv')

# Replace NaN values with empty strings
df = df.fillna('')

# Create a list of transactions
transactions = df.iloc[:, 1:].values.tolist()

# Encode the transactions
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
encoded_df = pd.DataFrame(te_array, columns=te.columns_)

# Generate frequent itemsets for different support thresholds
support_thresholds = [0.5, 1, 2, 3, 5, 10]
no_of_items = []

# Generate the table of frequent two-itemsets
table_data = []
for threshold in support_thresholds:
    frequent_itemsets = apriori(encoded_df, min_support=threshold/100, use_colnames=True, max_len=2)
    no_of_items.append(len(frequent_itemsets))
    table_data.append([f'support@{threshold}', len(frequent_itemsets)])

# Create the table of frequent two-itemsets
table_columns = ['Support', 'No. of items']
table = pd.DataFrame(table_data, columns=table_columns)

# Print the table of frequent two-itemsets
print(table)

# List all the frequent one-itemsets with support@3
frequent_itemsets_support_3 = apriori(encoded_df, min_support=3/100, use_colnames=True)
frequent_itemsets_support_3 = frequent_itemsets_support_3[frequent_itemsets_support_3['itemsets'].apply(lambda x: len(x) == 1)]
frequent_itemsets_support_3['Support'] = frequent_itemsets_support_3['support'] * len(transactions)
frequent_itemsets_support_3 = frequent_itemsets_support_3[['Support', 'itemsets']]
print(frequent_itemsets_support_3)
