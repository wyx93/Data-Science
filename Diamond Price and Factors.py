#!/usr/bin/env python
# coding: utf-8

# # Team Members

# Stella Wan, Jerry Tucay, Nikitha

# # Project Description

# Diamonds never fade in the market.Women love it and men have to buy it to get a wife.Therefore it is related to many people.But how to pick a cost-effective diamond? Which factor matters more regarding its price? What kind of diamond can you get with your budget? We would like to use visualization to figure it out,thus to provide some tips and guidance for round diamond shopping. The detailed questions are listed below.
# 
# 
# 1.The price distribution for caret,cut,color,clarity,depth and table.   
# 2.Verify "buying shy saves money" with visuals. (Buying shy suggests shopping for diamonds that weigh just under half-carat and full-carat weights to save money. For example, instead of a 1-carat (100-point) diamond you’d buy a .90-carat diamond. Instead of a half-carat, you’d buy a .49-carat diamond.)   https://diamondcuttersintl.com/buying-shy/     

# # Data Dictionary

# This classic dataset contains the prices and other attributes of almost 54,000 round diamonds. 
# 
# price price in US dollars (\$326--\$18,823)
# 
# carat weight of the diamond (0.2--5.01)
# 
# cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)
# 
# color diamond colour, from J (worst) to D (best)
# 
# clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
# 
# x length in mm (0--10.74)
# 
# y width in mm (0--58.9)
# 
# z depth in mm (0--31.8)
# 
# depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
# 
# table width of top of diamond relative to widest point (43--95)

# # Set up

# In[1]:


# Load Packages
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
import sklearn.metrics as metrics
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.formula.api import ols
import seaborn as sns
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row,gridplot
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
import matplotlib.pyplot as plt


# In[2]:


# Load Data
diamond=pd.read_csv(("C:/Users/Stella/OneDrive/文档/cal poly life/python/project1/diamonds.csv"))
print(diamond.head())


# # Data Preparation

# In[3]:


#convert the dataset to dataframe
diamond = pd.DataFrame(diamond)     

# print datatypes of columns
#print(diamond.dtypes)


# In[4]:


#check duplicates
a=diamond.drop_duplicates()
print(a.shape[0])     # no duplicates

#check nulls
#print(diamond.isnull().sum())  #no nulls    


# In[5]:


#aggregation (j)
diamond[['carat','depth','table','price','x','y','z']].aggregate(['max', 'min']) 


# In[6]:


# check rows with x=0 or y=0 or z=0
#print(diamond[diamond["x"]==0])
#print(diamond[diamond["y"]==0])
#print(diamond[diamond["z"]==0])


# In[7]:


#Categorization for table
diamond["table_level"]=pd.Series()           

for i in range(len(diamond["table"])):
    if diamond["table"][i]>=53 and diamond["table"][i]<=58:         
        diamond["table_level"][i]="Excellent"
    elif diamond["table"][i]>=52 and diamond["table"][i]<53 or diamond["table"][i]>58 and diamond["table"][i]<=60:
        diamond["table_level"][i]="Very Good"
    elif diamond["table"][i]>=51 and diamond["table"][i]<52 or diamond["table"][i]>60 and diamond["table"][i]<=64:
        diamond["table_level"][i]="Good"
    elif diamond["table"][i]>=50 and diamond["table"][i]<51 or diamond["table"][i]>64 and diamond["table"][i]<=69:
        diamond["table_level"][i]="Fair"
    else:
        diamond["table_level"][i]="Poor" 
#print(diamond)


# In[8]:


#Categorization and concat for depth                
depth_level=[]

for i in range(len(diamond["depth"])):
    if (diamond["depth"][i]>=59) and (diamond["depth"][i]<=62.3):
        depth_level.append("Excellent")
    elif diamond["depth"][i]>=58 and diamond["depth"][i]<59 or diamond["depth"][i]>62.3 and diamond["depth"][i]<=63.5:
        depth_level.append("Very Good")
    elif diamond["depth"][i]>=57.5 and diamond["depth"][i]<58 or diamond["depth"][i]>63.5 and diamond["depth"][i]<=64.1:
        depth_level.append("Good")
    elif diamond["depth"][i]>=56.5 and diamond["depth"][i]<57.5 or diamond["depth"][i]>64.1 and diamond["depth"][i]<=65:
        depth_level.append("Fair")
    else:
        depth_level.append("Poor") 

diamond=pd.concat([diamond,pd.DataFrame(depth_level)],axis=1)       
diamond.columns=[*diamond.columns[:-1], 'depth_level']           
#print(diamond)


# In[9]:


#Categorization for color
diamond["color_level"]=pd.Series()

for i in range(len(diamond["color"])):              
    if diamond["color"][i]=="E" or diamond["color"][i]=="D" or diamond["color"][i]=="F":
        diamond["color_level"][i]="Excellent"
    elif diamond["color"][i]=="G" or diamond["color"][i]== "H":
        diamond["color_level"][i]="Very Good"
    elif diamond["color"][i]=="I" or diamond["color"][i]== "J":
        diamond["color_level"][i]="Good"
    elif diamond["color"][i]=="K" or diamond["color"][i]== "L":
        diamond["color_level"][i]="Fair"
    else:
        diamond["color_level"][i]="Poor" 
#print(diamond)


# In[10]:


#Categorization for clarity
diamond["clarity_level"]=pd.Series()

for i in range(len(diamond["clarity"])):
    if diamond["clarity"][i]=="IF" or diamond["clarity"][i]=="VVS1" or diamond["clarity"][i]=="VVS2":  
        diamond["clarity_level"][i]="Excellent"
    elif diamond["clarity"][i]=="VS1" or diamond["clarity"][i]=="VS2":
        diamond["clarity_level"][i]="Very Good"
    elif diamond["clarity"][i]=="SI1" or diamond["clarity"][i]=="SI2":
        diamond["clarity_level"][i]="Good"
    elif diamond["clarity"][i]=="I1":
        diamond["clarity_level"][i]="Fair"
    else:
        diamond["clarity_level"][i]="Poor" 
#print(diamond)


# In[11]:


# drop useless columns
diamond=diamond.drop(['x','y','z'], axis=1)
print(diamond.columns)


# # Descriptive Analysis

# In[41]:


fig, ax =plt.subplots(1,2,figsize = (10,5))
sns.distplot(diamond['price'],ax=ax[0])
sns.distplot(diamond['carat'],color='salmon',ax = ax[1])
#fig.show()
#print(pdist,pdist1)
print("Price and Carat distribution")


# In[13]:


fig, ax =plt.subplots(1,2,figsize=(10,5))
sns.barplot(x= 'clarity_level' , y = 'price',data = diamond, palette= 'Blues_d',ax = ax[0])
sns.barplot(x= 'clarity_level' , y = 'carat',data = diamond, palette= 'Reds_r',ax = ax[1])
print("Mean price and carat distribution among different clarity levels")


# In[14]:


fig, ax =plt.subplots(1,2,figsize=(10,5))
sns.barplot(x= 'color' , y = 'price',data = diamond, palette= 'Blues_d',ax = ax[0])
sns.barplot(x= 'color' , y = 'carat',data = diamond, palette= 'Reds_r',ax = ax[1])
print("Mean price and carat distribution among different colors")


# ## color vs price

# In[15]:


sns.catplot(x="color", kind="count", palette="husl", data=diamond, order = diamond['color'].value_counts().index)


# Diamonds of G,E,F color are the majority in this dataset.

# In[16]:


sns.boxplot(x="color",y="price",order=["D", "E","F","G","H","I","J"],data=diamond)  #seaborn categorical plots


# Diamonds of color G,I and J have bigger variability in price.

# In[18]:


sns.boxplot(x="color_level",y="price",order=["Excellent","Very Good","Good"],data=diamond)


# The best color level Excellent has the smallest variability in price and the lowest median price in this dataset.

# In[19]:


# output to notebook 
output_notebook() 


# In[20]:


#subset colors
excellent=diamond[diamond.color_level=="Excellent"]
very_good=diamond[diamond.color_level=="Very Good"]
good=diamond[diamond.color_level=="Good"]

p2 = figure(plot_width=300, plot_height=225,title = "carat vs price for Good color")
p2.scatter('carat','price',source=good,fill_alpha=1, color='darkorange')
p2.xaxis.axis_label = 'carat'
p2.yaxis.axis_label = 'price'
p2.xgrid.visible = False
p2.ygrid.visible = False

p1 = figure(x_range = p2.x_range ,plot_width=300, plot_height=225,title = "carat vs price for Very Good color")
p1.scatter('carat','price',source=very_good,fill_alpha=1, color='springgreen')
p1.xaxis.axis_label = 'carat'
p1.yaxis.axis_label = 'price'
p1.xgrid.visible = False
p1.ygrid.visible = False

p0 = figure(x_range = p2.x_range,plot_width=300, plot_height=225,title = "carat vs price for Excellent color")
p0.scatter('carat','price',source=excellent,fill_alpha=1, color='lightseagreen')
p0.xaxis.axis_label = 'carat'
p0.yaxis.axis_label = 'price'
p0.xgrid.visible = False
p0.ygrid.visible = False


p = gridplot([[p0,p1,p2]])
show(p)


# For different colors, you can find the price range for the your ideal carat.For example, if you want your diamond to be as big as possible but you don't really care about color,and your budget is \\$5000, then you can choose between a pretty good 1-caret diamond and a cheapest 2-carat diamond.

# ## carat vs price

# In[21]:


#diamond.plot(kind="scatter",x="carat",y="price",pal)
sns.scatterplot(x = 'carat', y = 'price',color= 'g',data=diamond)


# As carat increases, price doesn't increase linearly.This means if a diamond of 0.5 carat costs \\$2500, a diamond of 1 carat of the same other factors will cost more than $5000.

# ## clarity vs price

# In[22]:


# by clarity and clarity_level
fig, ax =plt.subplots(1,2,figsize=(10,5))
sns.scatterplot(x = 'carat',y = 'price',data=diamond,hue='clarity',palette="gist_rainbow",ax = ax[0])
sns.scatterplot(x = 'carat',y = 'price',data=diamond,hue='clarity_level',palette="husl",style='clarity_level',ax = ax[1])


# We can see that the worst clarity I1 has the widest range of carat distribution. And diamonds of the other clarities are mostly <3 carats.
# 
# Fair clarity covers the widest range of carat. Excellent calrity covers the smallest variability in carat and most of these diamonds are <2 carats. 

# In[24]:


#subset clarity_levels
excellent=diamond[diamond.clarity_level=="Excellent"]
very_good=diamond[diamond.clarity_level=="Very Good"]
good=diamond[diamond.clarity_level=="Good"]
fair=diamond[diamond.clarity_level=="Fair"]
# for each clarity_level

p4 = figure(plot_width=300, plot_height=225,  title = "carat vs price by fair clarity")
p4.scatter('carat','price',source= fair ,fill_alpha=1, color='pink',size=5)

p3 = figure(x_range = p4.x_range,plot_width=300, plot_height=225, title = "carat vs price by good clarity")
p3.scatter('carat','price',source= good ,fill_alpha=1, color='palegreen',size=5)

p2 = figure(x_range = p4.x_range,plot_width=300, plot_height=225,title = "carat vs price by Very good clarity level")
p2.scatter('carat','price',source= very_good , color='crimson',size=5)

p1 = figure(x_range = p4.x_range,plot_width=300, plot_height=225,  title = "carat vs price by Excellent clarity level")
p1.scatter('carat','price',source= excellent ,fill_alpha=1, fill_color='steelblue',size=5)

p = gridplot([[p1,p2]])
p0 = gridplot([[p3,p4]])
show(p)

show(p0)


# For excellent clarity_level, we can see the price range for different carats. For example, if you want to buy a 1.5 carat diamond with excellent clarity, it will cost you \\$7500-$18500 depending on other factors.And if your budget is less than \\$3000 and you want to buy a 1 caret diamond, you probably should consider lower clarity level.
# Similar insights can be drawn from reading above graphs from other clarities. 

# ## cut vs price

# In[28]:


sns.catplot(x="cut", kind="count", palette="husl", data=diamond, order = diamond['cut'].value_counts().index)


# The majority of the diamonds in this dataset has pretty good cut.The number of Ideal cut ranks top and Premium ranks the second.

# In[31]:


sns.boxplot(x="cut",y="price",order=["Ideal","Premium","Very Good","Good","Fair"],data=diamond)  #seaborn categorical plots


# Premium cut has the largest variablity in price and also the biggest median price.

# In[29]:


# carat vs price by cut
index_cmap = factor_cmap('cut', palette=["teal","yellow","cornflowerblue","palegreen","darkorange"], 
                         factors=sorted(diamond.cut.unique()))
TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]
p = figure(plot_width=600, plot_height=450,  tooltips=TOOLTIPS,title = "carat vs price by cut")
p.scatter('carat','price',source=diamond,fill_alpha=1, color=index_cmap,size=5,legend='cut')
p.xaxis.axis_label = 'carat'
p.yaxis.axis_label = 'price'
p.legend.location = "top_right"
p.legend.title = 'Cut'

show(p)


# The best cut Ideal and Premium mainly appear for diamonds <3 carats in this dataset.

# In[33]:


#subset cut
ideal=diamond[diamond.cut=="Ideal"]
premium=diamond[diamond.cut=="Premium"]
very_good=diamond[diamond.cut=="Very Good"]
good=diamond[diamond.cut=="Good"]
fair=diamond[diamond.cut=="Fair"]
# for each cut

p5 = figure(plot_width=300, plot_height=225,  title = "carat vs price by fair cut")
p5.scatter('carat','price',source= fair ,fill_alpha=1, color='pink',size=5)

p4 = figure(x_range = p5.x_range,plot_width=300, plot_height=225,  tooltips=TOOLTIPS,title = "carat vs price by good cut")
p4.scatter('carat','price',source= good ,fill_alpha=1, color='green',size=5)

p3 = figure(x_range = p5.x_range,plot_width=300, plot_height=225,  tooltips=TOOLTIPS,title = "carat vs price by very good cut")
p3.scatter('carat','price',source= very_good ,fill_alpha=1, color='orange',size=5)

p1 = figure(x_range = p5.x_range,plot_width=300, plot_height=225,title = "carat vs price by premium cut")
p1.scatter('carat','price',source= premium , color='maroon',size=5)

p2 = figure(x_range = p5.x_range,plot_width=300, plot_height=225,  title = "carat vs price by ideal cut")
p2.scatter('carat','price',source= ideal ,fill_alpha=1, fill_color='steelblue',size=5)



p = gridplot([[p1,p2,p3]])
p10 = gridplot([[p4,p5]])
show(p)
show(p10)


# For each cut level, you can know the price range of diamond based on your desired carat.

# In[37]:


diamond.plot(kind="scatter",x="depth",y="price",color="green", title="Price vs depth for excellent color")


# In[38]:


diamond.plot(kind="scatter",x="table",y="price",color="green", title="Price vs table for excellent color")

