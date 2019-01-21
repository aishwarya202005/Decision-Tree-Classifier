#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pprint
import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log


# In[35]:


a=pd.read_csv('train_data.csv')
a.describe()

df = pd.DataFrame(a,columns=['Work_accident','promotion_last_5years','sales','salary','left'])


# In[36]:


# train, validate = np.split(df, [int(.8*len(df))]) #for sequential data
train, validate = np.split(df.sample(frac=1), [int(.8*len(df))]) # for random 


# In[37]:


entropy_node = 0 
values = train.left.unique()
x= len(train.left)
for val in values:
    fraction = train.left.value_counts()[val]/float(x)
    entropy_node += -fraction*np.log2(fraction+eps)


# In[58]:
pprint.pprint("main entropy=")
pprint.pprint(entropy_node)


# In[38]:


def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/float(len(df[Class]))
        entropy += -fraction*np.log2(fraction+eps)
    return entropy

def find_entropy_attribute(df,attribute):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
    variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
            den = len(df[attribute][df[attribute]==variable])
            fraction = num/(den+eps)
            entropy += -fraction*log(fraction+eps)
        fraction2 = den/float(len(df))
        entropy2 += -fraction2*entropy
    return abs(entropy2)


# In[39]:


def max_IG(dataf):    
    IG = []
    i=0
    max_=0
    max_in=''
    for key in dataf.keys()[:-1]:
        ig_=find_entropy(dataf)-find_entropy_attribute(dataf,key)       
        if ig_>max_:
            max_=ig_
            max_in=key
        IG.append(find_entropy(dataf)-find_entropy_attribute(dataf,key))
        i=i+1
    return max_in,max_


# In[40]:



class DTree:
    def __init__(self,pos=0,neg=0, child={},val = None):
        self.val = val
        self.child = child
        self.pos=pos
        self.neg=neg       
        

def buildTree(df1,label_node,tree=None): 
    max_ig_node,gn = max_IG(df1)
    
#     print("MAX IG=",max_ig_node)
#     pprint.pprint(max_ig_node)
#     pprint.pprint(gn)
    c_pos=0
    c_len=0
    if gn==0 or len(df1.columns)==1:
        ########switched 0 and 1
        c_pos=len(df1[df1[label_node]==1])
        c_len=len(df1[df1[label_node]==0])
        if c_pos>=c_len:
            max_ig_node="YES"
        else:
            max_ig_node="NO"
        child={}
        leaf_n=DTree(c_pos,c_len,child,max_ig_node)
        return leaf_n
        
    if max_ig_node== None:                    
        return None
    
    c_pos=len(df1[max_ig_node][df1[label_node]==1])
    c_len=len(df1[max_ig_node][df1[label_node]==0])
    child={}
    newnode=DTree(c_pos,c_len,child,max_ig_node)
#     pprint.pprint(df1)

    attr = df1[max_ig_node].unique()
#     pprint.pprint(attr)
    for labels in attr:
        nd1=df1[df1[max_ig_node]==labels]
        nd1=nd1.drop(columns=[max_ig_node])
        newnode.child[labels] = buildTree(nd1,label_node) #Calling the function recursively 
    return newnode

def traverse(tree):
    pprint.pprint(tree.val)
    if tree.val==1:
        return
    for k,vals in tree.child.items():
        pprint.pprint(k)
        traverse(vals)
    
    
tree = buildTree(train,'left')
pprint.pprint("tree")
traverse(tree)
# pprint.pprint(tree.val)
# pprint.pprint(tree.child['rain'].val)


# In[41]:


def predict(row,tree):
#     global tp,tn,fp,fn
    if tree.val=="YES" or tree.val=="NO":
        return tree.val
    
    attr=row[tree.val]
    if attr in tree.child:
        return predict(row,tree.child[attr])
    return "NO"

def predict1(df,tree):
    tp=0
    tn=0
    fp=0
    fn=0
    i=0
    while i< len(df):
        pred_val=predict(df.iloc[i],tree)
        if df.iloc[i]['left']==1:
            if pred_val=="YES":
                tp=tp+1
            else:
                fn=fn+1 #### fn here 
        elif df.iloc[i]['left']==0:
            if pred_val=="NO":
                tn=tn+1
            else:
                fp=fp+1 #### fp here
        i=i+1
#         print"i="
    return tp,fp,tn,fn
    
tp,fp,tn,fn=predict1(validate,tree)
print "True Positive=\t",tp
print "True Negative=\t",tn
print "false positive=\t",fp
print "False negative=\t",fn
total=tp+tn+fp+fn
prec=tp/float(tp+fp)
rec=tp/float(tp+fn)
den=float((1/rec)+(1/prec))
f1=2/float(den)

print "\nPrecision: ", prec
print "Recall: ",rec
print "F1: ",f1
print "accuracy: ",(tp+tn)/float(total)


# In[ ]:




