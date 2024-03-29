#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pprint
import sys
import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log


# In[71]:


a=pd.read_csv('train_data.csv')
a.describe()

df = pd.DataFrame(a)#,columns=['Work_accident','promotion_last_5years','sales','salary','left'])


# In[72]:


# train, validate = np.split(df, [int(.8*len(df))])  #for sequential data
train, validate = np.split(df.sample(frac=1), [int(.8*len(df))]) # for random 
# df= train


# In[73]:


def find_entropy(df):
    Class = 'left'   #To make the code generic, changing target variable class name
    entropy = 0
    values = df['left'].unique()
    for value in values:
        fraction = df['left'].value_counts()[value]/float(len(df[Class]))
        entropy += -fraction*np.log2(fraction+eps)
    return entropy




def find_entropy_attribute(df,attribute):
    Class = 'left'   #To make the code generic, changing target variable class name
    target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
    variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    if len(target_variables) == 1 or len(variables) == 1:
        return 0;
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
            den = len(df[attribute][df[attribute]==variable])
            fraction = num/(den+eps)
            entropy += -fraction*np.log2(fraction+eps)
        fraction2 = den/float(len(df))
        entropy2 += fraction2*entropy
    return entropy2



def min_num_entropy(df,attribute):
#     temp_d=df    
#     x=temp_d.sort_values(by=attribute)
#     print "min_continuous entropy"
#     variables = x[attribute].unique()
  
    variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    entropy2 = 0
    entropy1 = 0
    min_entr=sys.maxint
    min_split=sys.maxint
    entropy_fin=0
    for variable in variables:
        entropy = 0
        df1 = df[df[attribute]<=variable]
        df2 = df[df[attribute]>variable]
        entropy1 = find_entropy(df1)
        entropy2 = find_entropy(df2)

        
        fraction2 = len(df2)/float(len(df))
        fraction1 = len(df1)/float(len(df))
        entropy_fin = fraction1*entropy1+ fraction2*entropy2
        
        if min_entr>entropy_fin:
            min_entr=entropy_fin
            min_split=variable

    return min_entr,min_split
 
    
# print min_num_entropy(df,'satisfaction_level')    


# In[74]:


numerical_attributes=[]
for i in df.keys()[0:5]:
    numerical_attributes.append(i)
#     print(i)
# print numerical_attributes


# In[75]:


def max_IG(dataf):    
    IG = []
    i=0
    split=-1
    max_=0
    max_in=None
    #categ
    for key in dataf.keys()[5:-1]:
        if key!= 'left':
            temp_en = find_entropy_attribute(dataf,key) 
#             print temp_en
            ig1= find_entropy(dataf)-temp_en      
#             print("IG 1: ",ig1,key)        
            if ig1>max_:
                max_=ig1
                max_in=key
            IG.append(ig1)

    #continuous
    for key in dataf.keys()[0:5]:
        temp_en = min_num_entropy(dataf,key)[0]
        ig1=find_entropy(dataf)-temp_en    
#         print("IG 2conti: ",ig1,key)        
        if ig1>max_:
            max_=ig1
            max_in=key
            split=min_num_entropy(dataf,key)[1]      

        IG.append(ig1)
#     print 'ig=',IG
    return max_in,max_,split,IG

# max_IG(df)


# In[76]:


class DTree:
    def __init__(self,pos=0,neg=0, child={},val = None,spl_pt=-1):
        self.val = val
        self.child = child
        self.pos=pos
        self.neg=neg
        self.spl_pt=spl_pt
        

def buildTree(df1,label_node,tree=None):
    max_ig_node,gn,split,Info_gain_array = max_IG(df1)
#     print max_ig_node,gn,split
#     dd=find_entropy_attribute(df1,label_node)
#     print 'MAX IG= ',gn,'split=',split
#     pprint.pprint(max_ig_node)

    c_pos=0
    c_len=0
    
    
    if gn <= 0.000001 or len(df1.columns)==1:
#         print 'IG-array',Info_gain_array
           ########switched 0 and 1
        c_pos=len(df1[df1[label_node]==1])
        c_len=len(df1[df1[label_node]==0])
        if c_pos>c_len:
            max_ig_node="YES"
        else:
            max_ig_node="NO"
        child={}
#         print "leaf nnode hai"
        leaf_n=DTree(c_pos,c_len,child,max_ig_node)
        return leaf_n

    print 'node is--- ',max_ig_node
    c_pos=len(df1[max_ig_node][df1[label_node]==1])
    c_len=len(df1[max_ig_node][df1[label_node]==0])
    child={}
    
    
    
    if split==-1:
        newnode=DTree(c_pos,c_len,child,max_ig_node,-1)
        attr = df1[max_ig_node].unique()
      
        for labels in attr:
            nd1=df1[df1[max_ig_node]==labels]
            nd1=nd1.drop(columns=[max_ig_node])
            newnode.spl_pt=-1
            newnode.child[labels] = buildTree(nd1,label_node) #Calling the function recursively 

    else:
        newnode=DTree(c_pos,c_len,child,max_ig_node,split)
        label1="<="+str(split)
        label2=">"+str(split)   
#         print "numericl ka bno"
        nd1=df1[df1[max_ig_node]<=split]
        nd2=df1[df1[max_ig_node]>split]
#         print len(nd1)
#         print len(nd2)
        if len(nd1)==0 or len(nd2)==0:
            c_pos=len(df1[df1[label_node]==1])
            c_len=len(df1[df1[label_node]==0])
            if c_pos>c_len:
                max_ig_node="YES"
            else:
                max_ig_node="NO"
            child={}
#             print "leaf nnode hai"
            leaf_n=DTree(c_pos,c_len,child,max_ig_node)
            return leaf_n

        newnode.child[label1] = buildTree(nd1,label_node) #Calling the function recursively 
        newnode.child[label2] = buildTree(nd2,label_node) #Calling the function recursively 
    return newnode



# In[77]:


def traverse(tree):
    print(tree.val) #,'split=',tree.spl_pt)
    if tree.val==1:
        return
    for k,vals in tree.child.items():
        pprint.pprint(k)
        traverse(vals)
    
tree = buildTree(train,'left')
print "tree:-"

traverse(tree)
# pprint.pprint(tree.val)


# In[79]:


def predict(row,tree):
#     global tp,tn,fp,fn
    if tree.val=="YES" or tree.val=="NO":
        return tree.val
    
    attr=row[tree.val]
    
    if tree.val in numerical_attributes:
        if attr<=tree.spl_pt:
            x="<="+str(tree.spl_pt)
            return predict(row,tree.child[x])
        else:
            x=">"+str(tree.spl_pt)
            return predict(row,tree.child[x])
    else:
        if attr in tree.child:
            return predict(row,tree.child[attr])
    return "NO"

def calc_measures(df,tree):
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
    
        

tp,fp,tn,fn=calc_measures(validate,tree)
print "tp=",tp
print "tn=",tn
print "false positive=",fp
print "False negative=",fn
total=tp+tn+fp+fn
prec=tp/float(tp+fp)
rec=tp/float(tp+fn)
den=float((1/rec)+(1/prec))
f1=2/float(den)

print "Precision: ", prec
print "Recall: ",rec
print "F1: ",f1
print "accuracy: ",(tp+tn)/float(total)


# In[ ]:




