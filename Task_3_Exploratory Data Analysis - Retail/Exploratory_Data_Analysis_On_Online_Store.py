#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis (EDA)
# 
# 
# The purpose of this EDA is to find insights for online store.
# Steps involved are -:
# 
# - Data cleaning -> Find missing values
# - Data preparation -> look for correlation
# - Data transformation -> Removing duplicates and unneccesary columns
# - Visualization -> Through matplotlib and seaborn
# - Last step -> To conclude the insights
# 
# This is the very first data analysis. Please take the informations on this notebook with a grain of salt. I'm open to all improvements (even rewording), don't hesitate to leave me a comment or upvote if you found it useful. If I'm completely wrong somewhere or if my findings makes no sense don't hesitate to leave me a comment.
# This work was influenced by some kernels and some articles from medium/towards data science.
# 
#  <img src="b.png" width="600" align="left">

# 
# - Online shopping has grown in popularity over the years, mainly because people find it convenient and easy to bargain shop from the comfort of their home or office.
# 
# - One of the most enticing factor about online shopping, particularly during a holiday season, is it alleviates the need to wait in long lines or search from store to store for a particular item.
# 
# <img src="a.jpg" width="500" align="left">
# 

# # Preparations
# 
# 
# For the preparations lets first import the necessary libraries and load the files needed for our EDA

# ## Import All Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Load The Dataset

# In[2]:


dataFrame = pd.read_csv('SampleSuperstore.csv')


# **Let's read the dataset and analyse rows and columns.**
# 
# 

# In[3]:


# First Print any  random 9 rows
dataFrame.sample(9)


# In[4]:


# Now Let's Check the head of the dataset means check first 5 rows in dataset.
dataFrame.head()


# In[5]:


# Now Check last five rows
dataFrame.tail()


# In[6]:


# Check shape of the dataset  
dataFrame.shape


# **So in this dataset there are 9994 rows and 13 columns**

# In[7]:


# Now check The information about dataset like data types, names number of the columns present in dataset 
dataFrame.info()


# In[8]:


#Check all the dtypes or data-types of columns in the dataset
dataFrame.dtypes


# In[9]:


#Find all the columns names in our dataset
dataFrame.columns


# In[10]:


# Now check the Missing/NAN values .
dataFrame.isnull().sum()


# In[11]:


# Finding total number of null values in a dataset
print("Total number of null values: ",dataFrame.isnull().sum().sum())


# **Here we can see that there are no missing values present. Hence we can directly go for exploratory analysis part .**

# In[12]:


# Now Let's see the Statistical details of our dataset
dataFrame.describe()


# In[13]:


# Check for unnecessary columns,duplicates and drop them if not required
dataFrame.duplicated().sum()


# **'Row ID' column is nothing but the serial number so we will drop this column.**

# In[14]:


dataFrame.drop_duplicates()
# drop_duplicates returns only the dataframe's unique values
# The first occurrence of the value in the list is kept, but other identical values are deleted


# In[15]:


# nunique() function return number of unique elements in the object. 
# It returns a scalar value which is the count of all the unique values in the Index. 
# By default the NaN values are not included in the count.
dataFrame.nunique()


# In[16]:


# Now Find the correlation between columns name in our dataset
dataFrame.corr()


# In[17]:


# Now find the covariance in our dataset
dataFrame.cov()


# In[18]:


# value_counts() function returns object containing counts of unique values. 
# The resulting object will be in descending order so that the first element is the most frequently-occurring element.
# Excludes NAN values by default.
dataFrame.value_counts()


# In[19]:


# Now Let's delete the variable 
# Drop postol code column and assigning it to dataFrame1

col = ['Postal Code']
dataFrame1=dataFrame.drop(columns=col,axis=1)


# In[20]:


dataFrame['Country'].nunique()


# In[21]:


dataFrame['Country'].value_counts()


##dropping Country column
#df=df.drop('Country',axis=1)
#df.head()


# **By using above valu_counts() and nunique() we can Clearly say that the data is for US country only.**
# 
# 
# **so we can drop the 'Country' column as we dont need any analysis to be done based on it.** 
# 
# 

# **Now We can analyse the data from our dataset further in 3 different ways -:**
# 
# - PRODUCT LEVEL ANALYSIS
# - CUSTOMER LEVEL ANALYSIS
# - ORDER LEVEL ANALYSIS
# 
# 

# **Now Lets look at the product categories  and sub-categories available to shop for  a customers.**
# 
# 

# In[22]:


# Types of Category Available
print(dataFrame['Category'].unique())


#Number/Count of products in each category 
dataFrame['Category'].value_counts()



# In[23]:


#Types of Sub-Category Available
print(dataFrame['Sub-Category'].nunique())


#Number of products in each sub-category
dataFrame['Sub-Category'].value_counts()


# # Visualization of the dataset

# In[24]:


#First Let's see how sub-categories are distributed with respect to category.
plt.figure(figsize=(16,8))
plt.bar('Sub-Category','Category',data=dataFrame,color='g')
plt.show()


# - From the above graph, one can easily makeout which Category & Sub-Category to choose when they are looking to purchase a product

# In[25]:


# Now Let's take a look at how the sales  is distributed
print(dataFrame['Sales'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(dataFrame['Sales'], color='g', bins=100, hist_kws={'alpha': 0.4});


# **Note: By using the log function which is present in numpy could also do the job**
# 
# 

# **Now Draw a pie chart to visualise the sub-category**

# In[26]:


plt.figure(figsize=(12,10))
dataFrame['Sub-Category'].value_counts().plot.pie(autopct="%1.1f%%")
plt.show()


# - **So from above pie chart we can say that the store has wide variety of Office Supplies especially in Binders and Paper department.**

# **Now analyse the profit and sales for sub-category using bar chart**

# In[27]:


dataFrame.groupby('Sub-Category')['Profit','Sales'].agg(['sum']).plot.bar()
plt.title('Total Profit and Sales per Sub-Category')
# plt.legend('Profit')
# plt.legend('Sales')
plt.show()


# - **From the above chart we can say that the Highest profit is earned in Copiers while Selling price for Chairs and Phones is extremely high compared to other products.**
# 
# - **Another interesting fact is that people don't prefer to buy Fasteners and Tables from Superstore. Hence these departments are in loss.**

# - **Now Lets Draw a count plot for products in sub-category by region-wise**

# In[28]:


plt.figure(figsize=(15,8))
sns.countplot(x="Sub-Category", hue="Region", data=dataFrame)
plt.show()


# - **From above plot we can say that people who are residing in Western part of US tend to order more from superstore.**

# # Feature Creation

# **For better understanding of data. I will create some new columns like cost of the product and profit %**

# In[29]:


dataFrame['Cost']=dataFrame['Sales']-dataFrame['Profit']
print(dataFrame['Cost'].head())

dataFrame['Profit %']=(dataFrame['Profit']/dataFrame['Cost'])*100


# In[30]:


dataFrame.head()


# In[31]:


# Profit Percentage of first 5 product names
print(dataFrame['Sub-Category'].head() , dataFrame['Profit %'].head())


# In[32]:


#Products with high Profit Percentage 
dataFrame.sort_values(['Profit %','Sub-Category'],ascending=False).groupby('Profit %').head(5)



# - **from above we can say that Retailers selling Phone,Binders,Papers have got 100% Profit in their Business.**
# 

# **Now lets look at the data with respect to segment level**

# In[33]:


dataFrame['Segment'].nunique()
fig=plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
s=sns.countplot('Segment', data = dataFrame)
for s in ax.patches:
    ax.annotate('{:.0f}'.format(s.get_height()), (s.get_x()+0.15, s.get_height()+1))
plt.show()


# - **From the above distribution plot we can say that the distribution is highest in Consumer Segment.**

# In[34]:


# Top states from which store gets the maximum sales and profit
sortedTop = dataFrame.sort_values(['Profit'], ascending=False).head(20)
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
p = sns.barplot(x='Sales', y='Profit',hue='State',palette='Set1', data=sortedTop, ax=ax)
ax.set_title("Top  profitable States")
ax.set_xticklabels(p.get_xticklabels(), rotation=75)
plt.tight_layout()
plt.show()


# -**From above We can see that majority of the profit and sales are from Indiana and Washington State.**

# **Now lets calculate profit gained in each category**

# In[35]:


#Calculating Profit gained in each Category
fig=plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
sns.barplot('Sub-Category','Profit %',hue='Category',palette='Paired',data=dataFrame)
for o in ax.patches:
    ax.annotate('{:.0f}'.format(o.get_height()), (o.get_x()+0.15, o.get_height()+1))
plt.show()


# - **From above we can say that highest profit comes from Office supplies category**

# **State Wise Count-plot**

# In[36]:


print(dataFrame1['State'].value_counts())
plt.figure(figsize=(15,8))
sns.countplot(x=dataFrame1['State'])
plt.xticks(rotation=90)
plt.show()


# **Now Lets see how discount will effect on profits**

# In[37]:


plt.figure(figsize = (10,4))
sns.lineplot('Discount', 'Profit', data = dataFrame, color = 'r', label= 'Discount')
plt.legend()


# - **From above we can say that either we should give discount between 40 to 60 %**

# In[38]:


# From this plot we can say that Binders are the most selling product
print(dataFrame1['Sub-Category'].value_counts())
plt.figure(figsize=(12,6))
sns.countplot(x=dataFrame1['Sub-Category'])
plt.xticks(rotation=90)
plt.show()


# In[39]:


# Now Let's plot the heatmap for analysing the correlation 
fig,axes = plt.subplots(1,1,figsize=(9,6))
sns.heatmap(dataFrame.corr(),annot=True)
plt.show()


# In[40]:


# Now Let's plot the heatmap for analysing the covariance 
fig,axes = plt.subplots(1,1,figsize=(9,6))
sns.heatmap(dataFrame.cov(), annot= True)
plt.show()


# In[41]:


figsize=(15,10)
sns.pairplot(dataFrame1,hue='Sub-Category')


# ### So Now we Grouped or sum the sales ,profit,discount,quantity according to every state of region and also according to sub-categories sales

# In[42]:


grouped=pd.DataFrame(dataFrame.groupby(['Ship Mode','Segment','Category','Sub-Category','State','Region'])['Quantity','Discount','Sales','Profit'].sum().reset_index())
grouped


# ### Now lets see the  sum,mean,min,max,count median,standard deviation,Variance of each states of Profit

# In[43]:


dataFrame.groupby("State").Profit.agg(["sum","mean","min","max","count","median","std","var"])


# ### Now lets check number of quantity of products sold w.r.t sales

# In[44]:


x = dataFrame.iloc[:, [9, 10, 11, 12]].values
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0).fit(x)
    wcss.append(kmeans.inertia_)

sns.set_style("whitegrid") 
sns.FacetGrid(dataFrame, hue ="Sub-Category",height = 6).map(plt.scatter,'Sales','Quantity')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[45]:


Q1 = dataFrame.quantile(q = 0.25, axis = 0, numeric_only = True, interpolation = 'linear')

Q3 = dataFrame.quantile(q = 0.75, axis = 0, numeric_only = True, interpolation = 'linear')
IQR = Q3 - Q1

print(IQR)


# # Conclusion
# 
# Dataset there are 9994 rows and 13 columns
# 
# 
# I have analysed the data from our dataset further in 3 different ways -:
# 
# - PRODUCT LEVEL ANALYSIS
# - CUSTOMER LEVEL ANALYSIS
# - ORDER LEVEL ANALYSIS
# 
# 
#  The store has wide variety of Office Supplies especially in Binders and Paper department.
# 
# The Highest profit is earned in Copiers while Selling price for Chairs and Phones is extremely high compared to other products
# 
# Another interesting fact is that people don't prefer to buy Fasteners and Tables from Superstore. Hence these departments are in loss.
# 
# 
# People who are residing in Western part of US tend to order more from superstore.
# 
# 
# Retailers selling Phone,Binders,Papers have got 100% Profit in their Business.
# 
# The distribution is highest in Consumer Segment.**
# 
# 
# 
# Majority of the profit and sales are from Indiana and Washington State.
# 
# 
# The  highest profit comes from Office supplies category
# 
# 
# And Binder is the most selling product
# 
# 
# 
# 

# In[ ]:




