#!/usr/bin/env python
# coding: utf-8

# # Data Exploration and Preparation Group Project
# Group Members: E.G ; Kenny Thomas; A.R; L.B.
# 
# - Alessia 
# - Lyndon 
# - Kenny 
# - Ewelina

# #### Features Explanation from Kaggle 
# 
# Refrences: 
# - https://www.kaggle.com/jessemostipak/hotel-booking-demand
# 
# 
# The dataframe contains the following columns:
# 
# - **hotel**:  Hotel type
# - **is_canceled**: Binary values if a booking that was cancelled (1) or not (0)
# - **lead_time**: Measure of days from when the booking was made in to the Property management system and the arrival date.
# - **arrival_date_year**: Arrival year of the customers.
# - **arrival_date_month**: Arrival Month of the customers.
# - **arrival_date_week_number**: Week number of year for arrival date.
# - **arrival_date_day_of_month**: Day of arrival date.
# - **stays_in_weekend_nights**: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel.
# - **stays_in_week_nights**: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel.
# - **adults**:	Number of adults
# - **children**: Number of children.
# - **babies**: Number of babies.
# - **meal**: Type of meal booked. Categories are presented in standard hospitality meal packages: Undefined/SC – no meal package; BB – Bed & Breakfast; HB – Half board (breakfast and one other meal – usually dinner); FB – Full board (breakfast, lunch and dinner).
# - **country**: Country of origin. Categories are represented in the ISO 3155–3:2013 format.
# - **market_segment**: Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”.
# - **distribution_channel**: Booking distribution channel. The term “TA” means “Travel Agents” and “TO” means “Tour Operators”.
# - **is_repeated_guest**: Value indicating if the booking name was from a repeated guest (1) or not (0).
# - **previous_cancellations**:	Number of previous bookings that were cancelled by the customer prior to the current booking.
# - **previous_bookings_not_canceled**:	Number of previous bookings not cancelled by the customer prior to the current booking.
# - **reserved_room_type**:	Code of room type reserved. Code is presented instead of designation for anonymity reasons.
# - **assigned_room_type**:	Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due to hotel operation reasons (e.g. overbooking) or by customer request. Code is presented instead of designation for anonymity reasons.
# - **booking_changes**: Number of changes/amendments made to the booking from the moment the booking was entered on the PMS until the moment of check-in or cancellation.
# - **deposit_type**: Indication on if the customer made a deposit to guarantee the booking. This variable can assume three categories: *No Deposit – no deposit was made* ,  *Non Refund – a deposit was made in the value of the total stay cost*, *Refundable – a deposit was made with a value under the total cost of stay*.
# - **agent**: ID of the travel agency that made the booking.
# - **company**: ID of the company/entity that made the booking or responsible for paying the booking. ID is presented instead of designation for anonymity reasons.
# - **days_in_waiting_list**: Number of days the booking was in the waiting list before it was confirmed to the customer.
# - **customer_type**: Type of booking, assuming one of four categories: *'Contract', when the booking has an allotment or other type of contract associated to it*, *'Group', when the booking is associated to a group*, *'Transient', when the booking is not part of a group or contract, and is not associated to other transient booking*, *'Transient-party', when the booking is transient, but is associated to at least other transient booking*.
# - **adr**: Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights.
# - **required_car_parking_spaces**: Number of car parking spaces required by the customer.
# - **total_of_special_requests**: Number of special requests made by the customer (e.g. twin bed or high floor).
# - **reservation_status**: Reservation last status, assuming one of three categories: *'Canceled', booking was canceled by the customer*, *'Check-Out', customer has checked in but already departed*, *'No-Show', customer did not check-in and did inform the hotel of the reason why?*
# - **reservation_status_date**: Date at which the last status was set. This variable can be used in conjunction with the Reservation Status to understand when was the booking cancelled or when did the customer checked-out of the hotel.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
fig = plt.figure()
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# code to hide the pink warnings
import warnings 
warnings.filterwarnings('ignore')


# ***
# 
# 
# 
# ## Step 1: Explore features and data types
# Alessia R.
# 
# ##### Requirements
# We are required to classify the data in the dataset by datatypes 'categorical' or 'numerical' (descrete, continuos), therefore we will divide all the information in different groups to hightlight this recorgnition.
# 
# While this step is useful to get familiar with the dataset, it is actually not mandatory for our future analysis, that will see the use of mixed datatypes to answer our business question. 
# 
# ##### Code references:
# - https://stackoverflow.com/questions/22470690/get-list-of-pandas-dataframe-columns-based-on-data-type
# - https://www.geeksforgeeks.org/python-pandas-dataframe-select_dtypes/
# - https://chrisalbon.com/python/data_wrangling/pandas_list_unique_values_in_column/
# - https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
# - https://www.tutorialspoint.com/python_pandas/python_pandas_dataframe.htm
# - https://www.codespeedy.com/exclude-particular-column-in-dataframe-in-python/
# - https://cmdlinetips.com/2018/04/how-to-drop-one-or-more-columns-in-pandas-dataframe/
# - https://www.datacamp.com/community/tutorials/categorical-data
# - https://pbpython.com/categorical-encoding.html

# In[2]:


df = pd.read_csv('hotel_bookings.csv')


# In[3]:


# Code for showing  all features of the table
pd.set_option('max_columns', None)
df.head()


# ### 1.1 Check datatypes of every feature

# In[4]:


# Checking datatypes information 
df.info()


# ### 1.2 Clean data, identify and replace null values

# In[5]:


# Checking which columns have null values/ there are 4 of them
df.isnull().sum()


# ##### Fixing children
# Firstly, columns 'babies' and 'children' should consist the same type of data which is int64 as they provide similar category of data. You cannot say there were 2.5 children so no point for decimal places. Here dtype should be changed to int64. Besides, in the original .csv file this data is without comma.
# 
# Secondly, before changing the data type we have to replace 'NA/null' records in column 'children' as in this situation NA will equal 0.
# 
# ##### Note
# This is not a NaN or Null value. This data was inputed but doesn't make sense - if children were not applicable then there was no children so '0'. Besides, replacing this value will not change anything as there are only 4 of them in the whole column.
# 
# ##### Details
# Only 4 observations are null, against 119386. We can proceed with the data transformation.

# In[6]:


# Checking for null values
df['children'].isnull().value_counts()


# In[7]:


# Checking the mean beforehand, so we can check later if we are affected negatively with the overall result
df['children'].mean()


# In[8]:


# code inspiration https://www.geeksforgeeks.org/python-pandas-dataframe-fillna-to-replace-null-values-in-dataframe/
#replacing null data with '0' as if it is 'not applicable' means there are no children
df['children'].fillna(0, inplace = True) 


# In[9]:


# The change doesn't seem to harm the overall result
df['children'].mean()


# In[10]:


df['children'].isnull().value_counts()


# In[11]:


df['children'] = df['children'].astype('int64')


# In[12]:


df.children.dtype


# In[13]:


df.info()


# In[14]:


df.isnull().sum()


# ##### Fixing country
# 488 observations are null, against 118902. This equals to 0.4% of the tot amount of observations. We can proceed with the data transformation

# In[15]:


df['country'].isnull().value_counts()


# In[16]:


df.country.unique()


# In[17]:


# Replace countries that show null as N/A
df["country"].fillna("N/A", inplace = True)


# In[18]:


df['country'].isnull().value_counts()


# In[19]:


df.isnull().sum()


# ##### Fixing agent
# 16340 observations are null, against 103050. This equals to 15.8% of the total amount of observations. We can proceed with the data transformation.
# 
# ##### Note
# Even if 'agent' variables are numbers, as explained before on the features list, they actually represent ID's of comapny. Therefore I will proceed by using the mode values from the column.

# In[20]:


df['agent'].isnull().value_counts()


# In[21]:


df['agent'].unique()


# In[22]:


print(df['agent'].value_counts())


# In[23]:


df['agent'].fillna(df['agent'].value_counts().index[0], inplace = True)


# In[24]:


df['agent'].isnull().value_counts()


# In[25]:


df.isnull().sum()


# ##### Fixing company
# 112593 observations are null, against 6797 that are not. This number exceed the benchmark of 20% max null values for each feature. We can drop the column as it would not bring any value to further research.

# In[26]:


df['company'].isnull().value_counts()


# In[27]:


df = df.drop(['company'], axis=1)


# In[28]:


df.isnull().sum()


# In[29]:


df.shape


# ### 1.3 Group features by datatype to easily view categories of data and exceptions

# In[30]:


# Code inspiration from https://stackoverflow.com/questions/22470690/get-list-of-pandas-dataframe-columns-based-on-data-type
groups = df.columns.to_series().groupby(df.dtypes).groups
groups


# ### 1.4 Create new dataframes based on data type

# In[31]:


# Code inspiration from https://www.geeksforgeeks.org/python-pandas-dataframe-select_dtypes/
strings = df.select_dtypes(include ='O') 
strings


# In[32]:


num = df.select_dtypes(exclude ='O')
num


# ##### Handling categorical data

# In[33]:


# Code inspiration from https://chrisalbon.com/python/data_wrangling/pandas_list_unique_values_in_column/

# Categorical data - 'is_cancel' is binary
df.is_canceled.unique()


# In[34]:


# Categorical data - 'is_repeated' is binary
df.is_repeated_guest.unique()


# In[35]:


# Categorical data - agent observations are ID's - dtype should be changed to object
df.agent.unique()


# In[36]:


# Changing dtype for categorical columns'company', 'agent', 'is_canceled','is_repeated_guest','arrival_date_year' into object will ease grouping the features by data_type
cols = ['agent','is_canceled','is_repeated_guest', 'arrival_date_year']
df[cols] = df[cols].astype('object')
df.info()


# In[37]:


# Code inspiration from https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/

categorical = pd.DataFrame(df).select_dtypes(include ='O')
categorical = categorical.drop(['arrival_date_month','reservation_status_date','arrival_date_year'], axis=1)
categorical


# ##### Handling dates
# Dates can be manipulated as both categorical or numerical data, depending on the analysis we need to drive. In particular, step 3 requires to find the mean of numerical data, but a mean of a week number or day of the month would not representative of the real world scenario. Therefore, we will save all the dates in a single dataframe to reuse during step 6, 7 and 9 of analysis.

# In[38]:


dates_cols = ['arrival_date_month', 'reservation_status_date','arrival_date_year']
dates = df[dates_cols]

dates


# ##### Handling numerical data

# In[39]:


# Create new dataframe for numerical features only, to refine by grouping in discrete and continous
num = pd.DataFrame(df).select_dtypes(exclude ='O')
num.info()


# In[40]:


df.adr.unique()


# In[41]:


# Code inspiration https://www.tutorialspoint.com/python_pandas/python_pandas_dataframe.htm

# Create new dataframe for countinous features only
continuous =  pd.DataFrame(num, columns=['adr'])
continuous


# In[42]:


# Code inspiration https://www.codespeedy.com/exclude-particular-column-in-dataframe-in-python/

# Excluding unnecessary features
discrete = num.loc[:, ~num.columns.isin(['adr'])]
discrete


# In[43]:


categorical.info()


# In[44]:


continuous.info()


# In[45]:


discrete.info()


# In[46]:


dates.info()


# ***
# 
# 
# ## Step 2: Bar charts construction for categorical variables
# Alessia R.

# #### Domain Issue: Find the busiest months of the year
# Based on the chosen dataset, we proposed our domain problem to focus on knowing exactly which months are the busiest for the hotels. This analysis will be conducted on booking historical data collected from 2015 to 2017.
# 
# For this analysis we will reintegrate the variable *'arrival_date_month'* in the dataset, as this is our target variable for the business question, then we will explore relations and differences between the months and our categorical variables, by plotting with bar charts.
# 
# We chose to visualize specifically the variables that speak of hotels characteristics, therefore we are not interested on the country of booking and the agent customers booked with. We will specifically focus on:
# - **is_canceled**, to inspect what type of hotel gets more cancellations
# - **meal**, to inspect what type of hotel gets more meal reservations
# - **market_segment**, to explore what booking channel the different type of hotels should rely on 
# - **distribution_channel**, to explore what type of vendor is most popular based on hotel type  
# - **is_repeated_guest**, to explore what hotel gets more loyal customer
# - **reserved_room_type**, to explore what room type is most in demand based on hotel type
# - **deposit_type**, what type of hotel applies a non refundable policy, and how much this practice is popular between hotel types
# - **customer_type**, what type of customer should hotels expect based on their type
# - **reservation_status**, what type of hotel gets the most people who don't show for their booking

# ### 2.1 Prepare the dataset

# In[47]:


month = dates[['arrival_date_month']]
month


# In[48]:


categorical_analysis = pd.concat([categorical, month], axis=1)
categorical_analysis


# In[49]:


city_hotel = categorical_analysis[categorical_analysis['hotel'] == 'City Hotel']
city_hotel


# In[50]:


resort_hotel = categorical_analysis[categorical_analysis['hotel'] == 'Resort Hotel']
resort_hotel


# ### 2.2 Visualize Months vs is_canceled

# In[51]:


city_hotel['is_canceled'].value_counts()


# In[52]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
cancel_rh = sns.countplot(x="arrival_date_month", hue="is_canceled", data=resort_hotel, order=Months_order)


# In[53]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
cancel_ch = sns.countplot(x="arrival_date_month", hue="is_canceled", data=city_hotel, order=Months_order)


# ### 2.3 Visualize Months vs meal

# In[54]:


city_hotel['meal'].value_counts()


# In[55]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
meal_rh = sns.countplot(x="arrival_date_month", hue="meal", data=resort_hotel, order=Months_order )


# In[56]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
meal_ch = sns.countplot(x="arrival_date_month", hue="meal", data=city_hotel, order=Months_order)


# ### 2.4 Visualize Months vs market segment

# In[57]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
mark_seg_rh = sns.countplot(x="arrival_date_month", hue="market_segment", data=resort_hotel, order=Months_order )


# In[58]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
mark_seg_ch = sns.countplot(x="arrival_date_month", hue="market_segment", data=city_hotel, order=Months_order )


# ### 2.5 Visualize Months vs distribution channel

# In[59]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
distr_channel_rh = sns.countplot(x="arrival_date_month", hue="distribution_channel", data=resort_hotel, order=Months_order )


# In[60]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
distr_channel_ch = sns.countplot(x="arrival_date_month", hue="distribution_channel", data=city_hotel, order=Months_order )


# ### 2.6 Visualize Months vs repeated guests

# In[61]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
rep_guests_rh = sns.countplot(x="arrival_date_month", hue="is_repeated_guest", data=resort_hotel, order=Months_order)


# In[62]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
rep_guests_ch = sns.countplot(x="arrival_date_month", hue="is_repeated_guest", data=city_hotel, order=Months_order)


# ### 2.7 Visualize Months vs room type

# In[63]:


# Seaborn Countplot

Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
room_type_rh = sns.countplot(x="arrival_date_month", hue="reserved_room_type", data=resort_hotel, order=Months_order)
room_type_rh.legend(ncol=2, loc='upper right', )


# In[64]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
room_type_ch = sns.countplot(x="arrival_date_month", hue="reserved_room_type", data=city_hotel, order=Months_order)


# ### 2.8 Visualize Months vs type of deposit

# In[65]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
deposit_type_rh = sns.countplot(x="arrival_date_month", hue="deposit_type", data=resort_hotel, order=Months_order)


# In[66]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
deposit_type_ch = sns.countplot(x="arrival_date_month", hue="deposit_type", data=city_hotel, order=Months_order)


# ### 2.9 Visualize Months vs type of customer

# In[67]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
customer_type_rh = sns.countplot(x="arrival_date_month", hue="customer_type", data=resort_hotel, order=Months_order)


# In[68]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
customer_type_ch = sns.countplot(x="arrival_date_month", hue="customer_type", data=city_hotel, order=Months_order)


# ### 2.10 Visualize Months vs reservations

# In[69]:


city_hotel['reservation_status'].unique()


# In[70]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
reservations_rh = sns.countplot(x="arrival_date_month", hue="reservation_status", data=resort_hotel, order=Months_order)


# In[71]:


# Seaborn Countplot
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.figure(figsize=(11,8));
reservations_ch = sns.countplot(x="arrival_date_month", hue="reservation_status", data=city_hotel, order=Months_order)


# In[72]:


city_hotel.shape


# In[73]:


resort_hotel.shape


# ***
# 
# 
# ## Step 3: Mean, Median, Minimum, Maximum, Standard deviation
# Lyndon B.
# 
# ##### Code references:
# - https://datacarpentry.org/python-ecology-lesson/05-merging-data/index.html
# - https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475

# ### 3.1 Pull the clean numerical dataframes together
# First, we need to pull together the cleaned datasets (discrete and continuous) we obtained from our step 1 of data preparation, so we can restart the analysis with all the features available in a single dataset.

# In[74]:


numerical = num.copy()
numerical.info()


# ### 3.2 Separate dataframe based on hotel type
# The two hotels are different and therefore may exhibit different values and relationships from
# which differing conclusions can be drawn later on in the analysis, therefore we need to reintegrate the hotel information back on the numerical dataframe, so we can then separate the dataframes meaningfully.

# In[75]:


hotel_type = categorical[['hotel']]
hotel_type


# In[76]:


numerical_analysis = pd.concat([numerical, hotel_type], axis=1)
numerical_analysis


# In[77]:


numerical1 = numerical_analysis[numerical_analysis['hotel'] == 'Resort Hotel']    

numerical2 = numerical_analysis[numerical_analysis['hotel'] == 'City Hotel']


# In[78]:


numerical1


# In[79]:


numerical2


# ### 3.3 Summary statistics of the new dataframes

# In[80]:


numerical1.describe()


# In[81]:


numerical2.describe()


# ***
# 
# 
# ## Step 4: Min-Max Normalization, Z-score
# Lyndon B.
# 
# ### Refrences
# 
# ##### Resources:
# - <https://www.codecademy.com/articles/normalization>
# - <https://www.researchgate.net/post/When_and_why_do_we_need_data_normalization>
# 
# 
# ##### Code resources:
# 
# - https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475
# - https://stackoverflow.com/questions/43131274/how-do-i-plot-two-countplot-graphs-side-by-side-in-seaborn
# - https://stackoverflow.com/questions/62348532/adding-a-title-to-two-side-by-side-plots
# - https://stackoverflow.com/questions/16419670/increase-distance-between-title-and-plot-in-matplolib/56738085

# ### 4.1 Min-Max Normalization
# MinMax normalisation is a data preprocessing procedure that prepares data for later use by machine learning tools. If there are large differences in the scale of some features they may
# inappropriately influence or dominate the results of the later machine learning stages. Sometimes normalisation can make relationships more visible in visualisations though this 
# depends on the data itself

# #### Apply min-max normalization to Resort hotel dataframe

# In[82]:


numerical1 = numerical1.drop(['hotel'], axis=1)
numerical1.info()


# In[83]:


#https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475

# create a scaler object
scaler = MinMaxScaler()
# fit and transform the data
df_num1 = pd.DataFrame(scaler.fit_transform(numerical1), columns=numerical1.columns)

df_num1


# #### Apply min-max normalization to City hotel dataframe

# In[84]:


numerical2 = numerical2.drop(['hotel'], axis=1)
numerical2.info()


# In[85]:


# References: https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475

# Create a scaler object
scaler = MinMaxScaler()

# Fit and transform the data
df_num2 = pd.DataFrame(scaler.fit_transform(numerical2), columns=numerical2.columns)

df_num2


# ##### Example of the effect of Min-max normalisation on a feature
# 
# Notice in the side by side comparison below, how the feature' scale changes, demonstrated by the shrinking of the scale evident in the values comparison between the X axis.

# In[86]:


##https://stackoverflow.com/questions/43131274/how-do-i-plot-two-countplot-graphs-side-by-side-in-seaborn
##https://stackoverflow.com/questions/62348532/adding-a-title-to-two-side-by-side-plots
##https://stackoverflow.com/questions/16419670/increase-distance-between-title-and-plot-in-matplolib/56738085

fig, ax =plt.subplots(1,2)
sns.distplot(numerical1['lead_time'], ax=ax[0])
sns.distplot(df_num1['lead_time'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle(' Scaling: X axis',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# ### 4.2 Before and after normalisation
# 
# Let's look at some plots of interest before and after normalisation

# ##### Resort hotels

# In[87]:


# Seaborn histplot
##sns.histplot(data=numerical1, x="lead_time")
fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['lead_time'], ax=ax[0])
sns.histplot(df_num1['lead_time'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('lead_time: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[88]:


# Seaborn histplot

##sns.histplot(data=numerical1, x="lead_time")
fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['arrival_date_week_number'], ax=ax[0])
sns.histplot(df_num1['arrival_date_week_number'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('arr date wk no: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[89]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['arrival_date_day_of_month'], ax=ax[0])
sns.histplot(df_num1['arrival_date_day_of_month'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('arr date dom: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[90]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['stays_in_weekend_nights'], ax=ax[0])
sns.histplot(df_num1['stays_in_weekend_nights'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('stays in wen: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[91]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['stays_in_week_nights'], ax=ax[0])
sns.histplot(df_num1['stays_in_week_nights'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('stays in weeek n: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[92]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)

sns.histplot(numerical1['adults'],discrete=True, ax=ax[0])
sns.histplot(df_num1['adults'],discrete=True, ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('adults normalised: resort hotel',fontsize=15,size=16, y=1.12)
fig.tight_layout()
plt.xlim([0,1])


# In[93]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['children'], ax=ax[0])

sns.histplot(df_num1['children'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('children: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()
plt.xlim([0,1])


# In[94]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['babies'], ax=ax[0])

sns.histplot(df_num1['babies'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('babies: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[95]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['previous_cancellations'], ax=ax[0])

sns.histplot(df_num1['previous_cancellations'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('prev canc: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[96]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['previous_bookings_not_canceled'], ax=ax[0])

sns.histplot(df_num1['previous_bookings_not_canceled'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('prev bnc: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[97]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['booking_changes'], ax=ax[0])

sns.histplot(df_num1['booking_changes'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('booking_changes: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[98]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['days_in_waiting_list'], ax=ax[0])

sns.histplot(df_num1['days_in_waiting_list'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('DIWL: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[99]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['adr'], ax=ax[0])

sns.histplot(df_num1['adr'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('adr: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[100]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['required_car_parking_spaces'], ax=ax[0])

sns.histplot(df_num1['required_car_parking_spaces'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('RCPS: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[101]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['total_of_special_requests'], ax=ax[0])

sns.histplot(df_num1['total_of_special_requests'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('TOSR: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# ##### City hotels

# In[102]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['lead_time'], ax=ax[0])
sns.histplot(df_num2['lead_time'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('lead_time: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[103]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['arrival_date_week_number'], ax=ax[0])
sns.histplot(df_num2['arrival_date_week_number'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('arr date wk no: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[104]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['arrival_date_day_of_month'], ax=ax[0])
sns.histplot(df_num2['arrival_date_day_of_month'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('arr date dom: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[105]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['stays_in_weekend_nights'], ax=ax[0])
sns.histplot(df_num2['stays_in_weekend_nights'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('stays in wen: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[106]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['stays_in_week_nights'], ax=ax[0])
sns.histplot(df_num2['stays_in_week_nights'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('stays in weeek n: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[107]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)

sns.histplot(numerical2['adults'],discrete=True, ax=ax[0])
sns.histplot(df_num2['adults'],discrete=True, ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('adults normalised: city hotel',fontsize=15,size=16, y=1.12)
fig.tight_layout()
plt.xlim([0,1])


# In[108]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)

sns.histplot(numerical1['children'], ax=ax[0])
sns.histplot(df_num1['children'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('children: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()
plt.xlim([0,1])


# In[109]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)

sns.histplot(numerical1['babies'], ax=ax[0])
sns.histplot(df_num1['babies'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('babies: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()
plt.xlim([0,1])


# In[110]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['previous_cancellations'], ax=ax[0])

sns.histplot(df_num2['previous_cancellations'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('prev canc: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[111]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['previous_bookings_not_canceled'], ax=ax[0])

sns.histplot(df_num2['previous_bookings_not_canceled'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('prev bnc: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[112]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['booking_changes'], ax=ax[0])

sns.histplot(df_num2['booking_changes'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('booking_changes: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[113]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['days_in_waiting_list'], ax=ax[0])

sns.histplot(df_num2['days_in_waiting_list'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('DIWL: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[114]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['adr'], ax=ax[0])

sns.histplot(df_num2['adr'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('adr: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[115]:


# Seaborn histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['required_car_parking_spaces'], ax=ax[0])

sns.histplot(df_num2['required_car_parking_spaces'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('RCPS: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[116]:


fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['total_of_special_requests'], ax=ax[0])

sns.histplot(df_num2['total_of_special_requests'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('TOSR: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# **Adr vs Children before**

# In[117]:


# Seaborn scatterplot

ax = numerical1


x = ax['adr']
y = ax['children']


fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=ax )
scatter.set_title('Adr vs Children', fontsize = 23, y =1.05);


# **Adr vs Children before**

# In[118]:


# Seaborn scatterplot

ax = df_num1

# define the axes content
x = ax['adr']
y = ax['children']

#sns.scatterplot from DataFrame
fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=ax )
scatter.set_title('Adr vs Children Minmax', fontsize = 23, y =1.05);


# ### 4.3 Further exploration
# 
# Min-max Normalisation can sometimes expand our view into clustered or overplotted data but
# in this case, the same overplotted horizontal lines show meaning despite the heatmap data we 
# cannot easily identify linear correlational patterns between these features after normalisation.
# 
# Please pay attention however to the X and Y Axis, the scale has been recallibrated to lie between 0 and 1 for the relevant features.
# 
# Essentially the same result is achieved regarding visibility in the visualisation via utilising Z-score standardisation making this non-usable approach for these features in this case.
# 
# Let's try the same approach with some other variables which appear to exhibit some correlation
# and test if minmax or Z-score standardisation make a difference in visualising the data

# **Before normalization**

# In[119]:


# Seaborn scatterplot

ax = numerical2


x = ax['lead_time']
y = ax['stays_in_week_nights']


fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=ax )
scatter.set_title('stays_in_week_nights vs lead_time', fontsize = 23, y =1.05);


# 
# **After normalization**

# In[120]:


# Seaborn scatterplot

ax = df_num2


x = ax['lead_time']
y = ax['stays_in_week_nights']


fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=ax )
scatter.set_title('stays_in_week_nights vs lead_time', fontsize = 23, y =1.05);


# Again unfortunately, despite the heatmap suggesting some correlation between these two features,
# the normalisation has not helped us to unpack this visually.Normalisation, however, has uses to 
# be discussed later

# In[121]:


#https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475


# apply the z-score method in Pandas using the .mean() and .std() methods
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std
    
# call the z_score function
numerical_standardized1 = z_score(numerical1)

numerical_standardized1


# In[122]:


numerical_standardized1.mean()


# In[123]:


#https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475


# apply the z-score method in Pandas using the .mean() and .std() methods
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std
    
# call the z_score function
numerical_standardized2 = z_score(numerical2)

numerical_standardized2


# In[124]:


numerical_standardized1.describe()


# In[125]:


numerical_standardized2.describe()


# ### 4.4 Z-score standardisation
# 
# Normalisation and standardisation helps to prevent overfitting and helps with training a model
# later in the process. Z-score in particular when used on appropriate data helps to reduce the
# influence of outliers which would otherwise lead to bad results. Normalisation and standardisation can also speed up this process.
# 
# 
# ##### Resources:
# - https://www.codecademy.com/articles/normalization
# - https://www.researchgate.net/post/When_and_why_do_we_need_data_normalization
# 
# ##### Code Resources:
# - https://stackoverflow.com/questions/43131274/how-do-i-plot-two-countplot-graphs-side-by-side-in-seaborn
# - https://stackoverflow.com/questions/62348532/adding-a-title-to-two-side-by-side-plots
# 

# ##### Rescaling in Z-Score Standardisation
# Here you can see comparison of a feature from the non-standardised dataset Resort hotels and
# the same feature afterm Z-score standardisation. Pay attention to the x axis.

# In[126]:


# Seaborn Distplot

fig, ax =plt.subplots(1,2)
sns.distplot(numerical1['adr'], ax=ax[0])
sns.distplot(numerical_standardized1['adr'], ax=ax[1])
ax[0].set_title('Before Z-score standard')
ax[1].set_title('After Z-score standard')
fig.suptitle(' Scaling: X axis',fontsize=15,size=16, y=1.12)


# In[127]:


# Seaborn Histplot


fig, ax =plt.subplots(1,2)
sns.histplot(numerical1['arrival_date_week_number'], ax=ax[0])
sns.histplot(numerical_standardized1['arrival_date_week_number'], ax=ax[1])
ax[0].set_title('Before Standardisation')
ax[1].set_title('After Standardisation')
fig.suptitle('arr date wk no: Z-score standardisation - resort hotel',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[128]:


# Seaborn Histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['arrival_date_week_number'], ax=ax[0])
sns.histplot(numerical_standardized2['arrival_date_week_number'], ax=ax[1])
ax[0].set_title('Before Standardisation')
ax[1].set_title('After Standardisation')
fig.suptitle('arr date wk no: Z-score standardisation - city hotel',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[129]:


# Seaborn Histplot

fig, ax =plt.subplots(1,2)

sns.histplot(numerical2['adults'], ax=ax[0])
sns.histplot(numerical_standardized2['adults'], ax=ax[1])
ax[0].set_title('Before Standardisation')
ax[1].set_title('After Standardisation')
fig.suptitle('adults: Z-score Standardised',fontsize=15,size=16, y=1.12)
fig.tight_layout()
plt.xlim([0,1])


# In[130]:


# Seaborn Histplot

fig, ax =plt.subplots(1,2)
sns.histplot(numerical2['total_of_special_requests'], ax=ax[0])

sns.histplot(df_num2['total_of_special_requests'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('TOSR: normalised',fontsize=15,size=16, y=1.12)
fig.tight_layout()


# In[131]:


# Seaborn Histplot

fig, ax =plt.subplots(1,2)

sns.histplot(numerical1['children'], ax=ax[0])
sns.histplot(numerical_standardized1['children'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('children normalised: resort hotel',fontsize=15,size=16, y=1.12)
fig.tight_layout()
plt.xlim([0,1])


# In[132]:


# Seaborn Histplot

fig, ax =plt.subplots(1,2)

sns.histplot(numerical2['children'], ax=ax[0])
sns.histplot(numerical_standardized2['children'], ax=ax[1])
ax[0].set_title('Before Normalisation')
ax[1].set_title('After Normalisation')
fig.suptitle('children normalised: city hotel',fontsize=15,size=16, y=1.12)
fig.tight_layout()
plt.xlim([0,1])


# ***
# 
# 
# ## Step 5: Scatterplots for numerical variables
# Lyndon B.
# 
# ##### Code references:
# - https://stackoverflow.com/questions/38913965/make-the-size-of-a-heatmap-bigger-with-seaborn

# ### 5.1 Explore correlations for later anaysis with Scatterplot

# ##### Explore with Heatmap (Resort hotel)
# Generating a heatmap which we can cross-reference with pairplots and other data visualisations
# to understand the data better in powerful ways
# 
# This heatmap is displaying values for the resort hotels dataframe

# In[133]:


# References: https://stackoverflow.com/questions/38913965/make-the-size-of-a-heatmap-bigger-with-seaborn

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20,10))         # Sample figsize in inches
sns.heatmap(df_num1.corr(), annot=True, linewidths=.5, ax=ax)
ax.set_title("Resort hotel correlational heatmap",fontsize=35)


# In[134]:


#References: https://stackoverflow.com/questions/38913965/make-the-size-of-a-heatmap-bigger-with-seaborn

fig, ax = plt.subplots(figsize=(20,10))         # Sample figsize in inches
sns.heatmap(df_num2.corr(), annot=True, linewidths=.5, ax=ax)
ax.set_title("City hotel correlational heatmap",fontsize=35)


# ##### Explore with pairplot (Resort hotel)
# Generating a pairplot for the resort hotels dataframe, this can help us to see at a glance
# interesting data relations, potential correlations, patterns, outliers and irrelevancies
# in the data by comparing all columns.

# In[135]:


sns.pairplot(numerical1)


# ##### Explore with pairplot (City hotel)
# Generating a pairplot for the city hotels dataframe, this can help us to see at a glance
# interesting data relations, potential correlations, patterns, outliers and irrelevancies
# in the data by comparing all columns.

# In[136]:


sns.pairplot(numerical2)


# ### 5.2 Scatterplot visualizations

# #### Relationships which show at least some correlation in Resort Hotel
# 
# From the pairplot an the heatmaps as a cross-reference, we can look at some relationships which might be promising.

# In[137]:


# Seaborn Scatterplot

ax = numerical1


x = ax['stays_in_week_nights']
y = ax['stays_in_weekend_nights']


fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=ax )
scatter.set_title('stays_in_week_nights vs stays_in_weekend_nights', fontsize = 23, y =1.05);

sns.lmplot(x='stays_in_week_nights',y='stays_in_weekend_nights',data=df_num1)
ax = plt.gca()
ax.set_title("stays_in_week_nights vs stays_in_weekend_nights")


# #####  Observations 'stays_in_weekend_nights vs stays_in_week_nights' plot
# 
# stays_in_weekend_nights vs stays_in_week_nights shows extreme promise as a candidate for strong correlation, visually easily identifiable in pairplot and in the heatmap for resort hotel. we find a positive correlation of 0.72 and a moderate correlation in the other hotel. Unfortunately this is not helpful as it looks. The data visualisations should show us useful information which we cannot know beforehand. In this case a little interpretation shows that the correlation is just the effect of people who stay in the hotel for more than a week-day and inclusive of a weekend days which we would naturally expect to correlate to some extent.

# In[138]:


# Seaborn Scatterplot

ax = df_num1


x = ax['lead_time']
y = ax['stays_in_week_nights']


fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=ax )
scatter.set_title('stays_in_week_nights vs lead_time', fontsize = 23, y =1.05);


# ##### Observations 'stays_in_week_nights vs lead_time' plot
# 
# In this comparison we can fit a line slightly better though we still have some familiar 
# prolems, outliers, vertical clustered data, overplotting. The correlation is slightly stronger
# than the previous comparison at 0.39 but still relatively weak.

# In[139]:


# Seaborn Scatterplot

ax = df_num1


x = ax['lead_time']
y = ax['stays_in_weekend_nights']


fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=ax )
scatter.set_title('stays_in_weekend_nights vs lead_time', fontsize = 23, y =1.05);


# ##### Observation 'stays_in_weekend_nights vs lead_time'  plot
# 
# These features show weak to moderate correlation in overview.The linear graphic shows but since the vertical bands make the data look more categorical, this might not be an ideal candidate feature comparison for further analysis,this problem is to be discussed more in the section of rejected scatterplots

# In[140]:


# Seaborn Scatterplot
# Seaborn Lmplot

x = numerical2['children']
y = numerical2['adr']


fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=numerical2 )
scatter.set_title('ADR vs children', fontsize = 23, y =1.05);

sns.lmplot(x='children',y='adr',data=df_num2)


# ##### Observations 'adr vs children'  plot
# 
# The vertical bands could be confusing however we expect this to be the case with children which, afterall, are a constant and cannot be a continuous variable.This may make this not an ideal candidate for further analysis. A relationship between adr and children is plausible however, in terms of positive linearity as a greater number of children would presumably require larger rooms. 

# #### Features in City hotels which exhibit at least some correlation
# Overall there are weaker correlations found in this dataframe although the correlations in Resort hotel were generally either not strong or not good candidates for further analysis, or both

# In[141]:


# Seaborn Scatterplot
# Seaborn Lmplot


x = df_num2['previous_bookings_not_canceled']
y = df_num2['previous_cancellations']


fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=df_num2 )
scatter.set_title('previous_bookings_not_canceled vs previous_cancellations', fontsize = 23, y =1.05);

sns.lmplot(x='previous_bookings_not_canceled',y='previous_cancellations',data=df_num2)


# ##### Observations 'previous_bookings_not_canceled vs previous_cancellations' plot
# 
# Although we can still see horizontal bands of data and outliers, we can make out a positive
# line in this graph comparing features from city hotel, which showed a correlation of 0.39.
# Perhaps customers with more stays overall are more likely to have both cancellations and
# bookings not canceled, and if this were a business question, this relationship could be investigated in more depth though the correlation is still not strong.

# In[142]:


# Seaborn Scatterplot
# Seaborn Lmplot

x = ax['children']
y = ax['adr']


fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=df_num2 )
scatter.set_title('ADR vs children', fontsize = 23, y =1.05);

sns.lmplot(x='children',y='adr',data=df_num2)


# In[143]:


# Seaborn Scatterplot
# Seaborn Lmplot

x = df_num2['adults']
y = df_num2['adr']


fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=df_num2 )
scatter.set_title('ADR vs adults', fontsize = 23, y =1.05);

sns.lmplot(x='adults',y='adr',data=df_num2)
ax = plt.gca()
ax.set_title('adr vs adults: city hotel')


# ##### Observations 'adr vs adults' plot
# At 0.26 this is an even weaker correlation found in the city hotel and resembles the comparison between adr and children. Again, there is some plausibility to a positive linear relationship since a greater number of adults presumably require more expensive lodgings

# ***
# 
# 
# 
# ## Step 6: Exploratory Data Analysis
# Kenny T.
# 
# ### Domain Issue: Find the busiest months from the given years 
# 
# Based on the chosen dataset, we proposed our domain problem to focus on knowing exactly which months are the busiest for the hotels. This analysis will be conducted on bookings of historical data collected from 2015, 2016 and 2017. It will be helpful to any business that wishes to plan their resources budgets allocations in advance.
#  
# 
# #### Visualizations included
# - Barplot: differences between city and resort hotel
# - Barplot: the busiest months for Hotel bookings
# - Count of people vs Months of year = June ---*
# - Booking vs Year
# - Country vs Count (hue by year)
# - June ---*  hotel 1 vs hotel2
# - Weeks or Weekends is most preferred and busiest?
# 
# ##### Code references:
# - https://datacarpentry.org/python-ecology-lesson/05-merging-data/index.html
# - https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
# - https://stackoverflow.com/questions/31594549/how-do-i-change-the-figure-size-for-a-seaborn-plot
# - https://www.dataforeverybody.com/seaborn-legend-change-location-size/
# - http://seaborn.pydata.org/generated/seaborn.lmplot.html

# ### 6.1 Dropping columns we do not need for the analysis
# For the purpose of making the Exploratory Data Analysis easier and more granular, I will drop the columns that are not necessary to answer our domain problem. But first, we need to pull together the cleaned and transformed datasets (categorical, discrete, continuous, dates) we obtained from our step 1 of data preparation, so we can build the analysis with all the features available to a single dataset. 

# In[144]:


# code inspiration https://datacarpentry.org/python-ecology-lesson/05-merging-data/index.html
# Place the DataFrames side by side
clean_df = pd.concat([categorical, discrete, continuous, dates], axis=1)
clean_df


# In[145]:


domain_df = clean_df.drop(['lead_time','meal','market_segment','distribution_channel','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','reserved_room_type','assigned_room_type','booking_changes','deposit_type','previous_bookings_not_canceled','reservation_status','deposit_type','agent','days_in_waiting_list','reservation_status_date','customer_type','required_car_parking_spaces','total_of_special_requests','reservation_status','reservation_status_date'], axis=1)

domain_df.info()


# ### 6.2  Sort out the overall cancellation rate for Hotels
# In order to work with sensible data, we need to consider and work out all the bookings that haven't actually been checked in.
# 
# ##### Result:
# There are 44224 cancellations. From here we will operate removing canceled bookings for granular analysis, and we will work only with customers who checked in to the hotels.

# In[146]:


# Return the series containing the unique rows in the columns, 'is_canceled' DataFrame.
domain_df['is_canceled'].value_counts()


# In[147]:


confirmed_df = domain_df[(domain_df.is_canceled == 0)] 


# In[148]:


confirmed_df.shape


# ### 6.3  Adjust children and adults features to fit next analysis 

# In[149]:


confirmed_df.isnull().sum()


# In[150]:


confirmed_df.children.value_counts()


# In[151]:


confirmed_df.adults.value_counts()


# In[152]:


# Replace adult 0 values to higest values counts based on mode values

confirmed_df["adults"].replace(0, 2,inplace=True)


# In[153]:


confirmed_df.adults.value_counts()


# In[154]:


confirmed_df.info()


# In[155]:


confirmed_df.shape


# ### 6.4 Visualize differences in data between hotel types
# Checking for the values of the records of the dataframe features City Hotels and Resort Hotels, to check if it we can distinguish between city and resort hotels results.
# 
# ##### Result:
# We see that the is a major difference between City hotels and Resort hotels for check-in days, therefore, for the purposes of exploring the dataset accurately and for a more refined analysis, we proceed by splitting the dataset in two, based on the feature hotel type: "City Hotel" and "Resort"

# In[156]:


# Seaborn Barplot 

kx = confirmed_df.groupby('hotel').arrival_date_day_of_month.count().reset_index()
sns.barplot(x='hotel', y='arrival_date_day_of_month',data= kx)
plt.xlabel ('Hotel')
plt.ylabel ('Arrival day of the month counts')
plt.show()


# ### 6.5 Generate new dataframes based on hotel type
# We will obtain two datsets to analyse and compare together for a more indepth analysis

# In[157]:


df_resort = confirmed_df[confirmed_df['hotel'] == 'Resort Hotel']
df_resort


# In[158]:


df_city = confirmed_df[confirmed_df['hotel'] == 'City Hotel']
df_city


# ### 6.6 Explore numerical variables association and any significance in findings

# #### Extract numerical variables for both Resort and City Hotels

# In[159]:


city_num_df = df_city.select_dtypes(exclude ='O')
city_num_df


# In[160]:


resort_num_df = df_resort.select_dtypes(exclude ='O')
resort_num_df


# #### Check for any correlation between lead times and other numerical variables from the DataFrame 
# Result: Showing very low correlation in numerical variables from initial touch point

# In[161]:


# Reference: https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
# Checking to see correlation with one variable form the dataset 
# Seaborn correlation plot

plt.figure(figsize=(8, 4))
heatmap = sns.heatmap(resort_num_df.corr()[['arrival_date_day_of_month']].sort_values(by='arrival_date_day_of_month', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Arrival day of the month', fontdict={'fontsize':15}, pad=16);


# In[162]:


# Reference: https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
# Checking to see correlation with one variable form the dataset 
# Seaborn correlation plot

plt.figure(figsize=(8, 4))
heatmap = sns.heatmap(city_num_df.corr()[['arrival_date_day_of_month']].sort_values(by='arrival_date_day_of_month', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Arrival day of the month', fontdict={'fontsize':15}, pad=16);


# #### Check for overall correlations of numerical data from the Dataframe (City hotel) 

# In[163]:


# Seaborn correlation plot

plt.figure(figsize=(8,8))
sns.heatmap(city_num_df.corr(),annot=True,cmap='rocket_r')


# ##### Check for overall correlations of numerical data from the Dataframe (Resort hotel) 

# In[164]:


# Seaborn correlation plot

plt.figure(figsize=(8,8))
sns.heatmap(resort_num_df.corr(),annot=True,cmap='rocket_r')


# There is some correlation for week days and week nights stay in hotels, but due to the week being seven days its an expected relationship we foresee. Other than the two closely related variables, the rest of the variables do not show any coherence in any shape or form we can interpret. we may find more clarity later in Principle component analysis. 

# #### Relationship in plots

# In[165]:


# Seaborn Barplot

# References: https://stackoverflow.com/questions/31594549/how-do-i-change-the-figure-size-for-a-seaborn-plot

sns.set(rc={'figure.figsize':(11.7,8.27)})
ax=sns.barplot(x='arrival_date_week_number',y='adr',data=city_num_df,palette=('flare')) 

ax.set_xlabel('arrival_date_week_number', fontdict={'fontsize':15});
ax.set_ylabel('Average daily room rate count', fontdict={'fontsize':15});


# We see the average room rates for city hotels stay mostly with in the range on 80 - 120 for booking counts. 

# In[166]:


# Seaborn Barplot

# References: https://stackoverflow.com/questions/31594549/how-do-i-change-the-figure-size-for-a-seaborn-plot

sns.set(rc={'figure.figsize':(11.7,8.27)});
fx=sns.barplot(x='arrival_date_week_number',y='adr',data=resort_num_df,palette=("spring")) 

fx.set_xlabel('arrival_date_week_number', fontdict={'fontsize':15});
fx.set_ylabel('Average daily room rate count', fontdict={'fontsize':15});


# With resort hotels we see a pattern of moslty bookings that show 'low' activity to 'high' depending on year seasons perhaps, this could also mean the marginal gross average daily rate is variable depending on seasons peak activity.

# #### Explore adr for Resort Hotels 
# 
# The figure shows most weeknights stays per booking between 0-10 Nights, there are a few outliers, and these could come from group bookings or long stay types. Average daily room rate also is at its peak for day of 10 or less, showing a decrease in price after 10 days and long term bookings may be sold at a lower per night room price point.

# In[167]:


# Seaborn Scatter plot 

# References: https://www.dataforeverybody.com/seaborn-legend-change-location-size/

sns.set_style('whitegrid')

# acquire the data
ax = resort_num_df

# define the axes content
x = ax['stays_in_week_nights']
y = ax['adr']

#sns.scatterplot from DataFrame
fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=ax )
scatter.set_title('Arrival day vs Average daily Rate for Resort Hotels', fontsize = 23, y =1.05);


# #### Explore adr for City Hotels 
# The figure shows most weeknights stays per booking between 0-10 Nights, there are a few outliers as we can see in graph and these may come from group or long stay booking types. Average daily room rate also is at its peak for day of 6 or less, showing a decrease in price after 6 days of Hotel night stays.

# In[168]:


# Seaborn Scatter plot 

# References: https://www.dataforeverybody.com/seaborn-legend-change-location-size/

sns.set_style('whitegrid')

# acquire the data
mx = city_num_df

# define the axes content
x = mx['stays_in_week_nights']
y = mx['adr']

#sns.scatterplot from DataFrame
fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=mx )
scatter.set_title('Arrival day vs Average daily Rate for City Hotels', fontsize = 23, y =1.05);


# ####  Numerical associations and findings with Implots (Resort hotel)
# We see there is peak activity surounding the busy months in summer and once again peaking towards the end of the year. Average Room rate scale upwards with mostly 2 or 3 adults per booking check-in's. 

# In[169]:


# Seaborn Lmplot 

# Refrences: http://seaborn.pydata.org/generated/seaborn.lmplot.html

g = sns.lmplot(x="arrival_date_week_number", y="adr", hue="adults",col="adults",
               data=resort_num_df, height=6, aspect=.4, x_jitter=.1)


# ####  Numerical associations and findings with Implots (City hotel)
# We see there is higher spread with 2 adults in City Hotel bookings. The associations we see above are between the numerical variables are mostly in line with the average room rate, number of adults and time periods like months, weeks, or days of stay.

# In[170]:


# Seaborn Lmplot 

# Refrences: http://seaborn.pydata.org/generated/seaborn.lmplot.html

h = sns.lmplot(x="arrival_date_week_number", y="adr", hue="adults",col="adults",
               data=city_num_df, height=6, aspect=.4, x_jitter=.1)


# #### Final Observations
# 
# **(1)**
# 
# We tried to drop columns and separate the data frame by Resorts and City Hotels, and in doing so even then the numerical correlations where rather poor or showed no relationship. One of the challenges with dealing with this type of dataset are that most variables may come to us Categorical, Nominal, or continuous variables types.
# 
# **(2)**
# 
# In this analysis we took dependent variables to show relationships from the data set. The variables being categorical could mean in most cases a Bar plot could be used to analyse the count of Y axis variables, to draw robust and useful insights (as we did for Step 2)
# 
# **(3)**
# 
# Regarding the accuracy of relationships for each pair of the numerical variables from the dataset, the associations are showing either poor or no correlation among most variables except where average daily rates reveal the busiest month. As our main objective for this analysis is to analyse the busiest and most profitable  months, our variable defers to share the commonalities we expect to show from a numerical relationship. 

# ### 6.7 Domain question: Visualize the busiest months for Hotel bookings
# The visualization will compare the busiest months based on hotel type, therefore I need to generate a new dataframe containing the data of both hotel types separated in different features. (transactions count)

# ##### Which month is most booked for the City Hotels?

# In[171]:


df_city['arrival_date_month'].value_counts()


# ##### Which month is most booked for the Resort Hotels?

# In[172]:


# Return the series containing the unique rows in the columns, 'arrival_date_month' DataFrame.

df_resort['arrival_date_month'].value_counts()


# ##### Calculating the booking rate for August from the given years in the dataframe

# In[173]:


August_booking_perc = df_city['arrival_date_month'].value_counts(normalize=True)['August']
August_booking_perc


# ##### Plot
# We see that the busiest month is showing August from the given years for both Resort and City Hotels Dataset 
# 
# #### Plot Refrences: 
# 
# - Refrences: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html
# - Refrences: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure

# In[174]:


# Seaborn Countplot 

# Column 'arrival_date_month' plot based on categories

#Rearranging months
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Custom size plot

fig.set_size_inches(20, 15)

domain_plot = sns.set(style="darkgrid")
domain_plot = sns.countplot(x="arrival_date_month", data=confirmed_df, palette="mako",hue='hotel',order=Months_order)

domain_plot.annotate("", xy=(7.2, 5520), xytext=(7.2, 5370), arrowprops= dict(facecolor='gray'),annotation_clip=False);
plt.text(6.2, 5545, "Most booked month for city hotel", horizontalalignment='left', size='small', color='red', weight='bold');

domain_plot.annotate("", xy=(6.8, 5160), xytext=(6.8, 3250), arrowprops= dict(facecolor='gray'),annotation_clip=False);
plt.text(3.6, 5175, "Most booked month for Resort Hotels", horizontalalignment='left', size='small', color='blue', weight='bold');


plt.title('Number of monthly bookings per Hotel', weight='bold', fontsize=24,pad=10);

plt.xlabel('Months',labelpad=10, weight='bold',fontsize=15);
plt.ylabel('Count of Bookings', weight='bold',fontsize=15);

domain_plot.set_xticklabels(domain_plot.get_xticklabels(), fontsize=13, rotation=15);


# ### 6.8 Domain sub-question: Visualize the most profitable time of the year

# #### Plot Refrences: 
# 
# - Refrences: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html
# - Refrences: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure

# In[175]:


# Seaborn Barplot 

#Rearranging months
Months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Bar plot
sns.barplot(data=confirmed_df, x='arrival_date_month', y='adr', hue='hotel',palette=("viridis"),order=Months_order)

# Custom size plot
fig.set_size_inches(20, 15)

# Title for plot 
plt.title('Average daily rate comparison between Hotels and Resort', weight='bold', fontsize=18);

# label parameters for the plot
plt.xlabel('Months',labelpad=10, weight='bold',fontsize=18);
plt.ylabel('Average daily rate for rooms',labelpad=10, weight='bold',fontsize=18);


# #### The average daily room rate variations in years visually for City Hotels
# City hotels show a steady increase in room pricing and profitability year on year, also we see from the previous graphs that the city hotels may see marginal growth year on year. Whereas resort hotels we see in the plot shows a decline in year 2016 following a sizable increase the following year 2017, this level of fluctuation might suggest a percentage of uncertainty in projecting future average daily room rates increase.

# In[176]:


# Seaborn Bar Plot for years as x-axis to adr in y-axis to determine the yearly growth patterns in hotel industry. 

import seaborn as sns 

plt.figure(figsize=(16,10))

gx = sns.barplot(x='arrival_date_year',y='adr',data=confirmed_df,palette=("cividis"), hue = 'hotel')

plt.title('Average daily room rate Vs years', weight='bold', fontsize=18,pad=10);

plt.xlabel('Years',labelpad=10, weight='bold',fontsize=18);
plt.ylabel('Average daily room rate', weight='bold',fontsize=18);

gx.set_xticklabels(gx.get_xticklabels(), fontsize=18);


# ### 6.9 Answer to Domain question and sub-question
# 
# ### 1. Busiest Month for hotel bookings (Domain question)
# 
# - From the exploratory data analysis, we discovered that the busiest booking months to be first and the highest to be August and second busiest month to be July for both Resort and City hotels. 
# 
# - This gives us the ability to have conversation with prospects on how best to marginalise on less busy months and increase preparations for the busy months mainly July and August. 
# 
# 
# #### 2. The most profitable months for hotel bookings (Domain sub-question)
# 
# - The second domain issue we had was which month is the most profitable for hotels?
# 
# - From analysis we discovered the busiest months to be the following: 
# 
# 
# #### I. Resort hotels: 
# 
# The highest daily rates were found around August. Whereas the other months fluctuated marginally with smaller daily room rates or these being quieter periods perhaps.
# 
# #### II. City hotels: 
# 
# We found the City Hotels to be the most profitable and stable group among the observed hotel types. City hotels showed the most profitable month being 'May' month. There is also steady scale in pricing between April - September months.

# ***
# 
# 
# ## Step 7: Identifying sub-groups
# Kenny T.

# Based on the current findings there is much to explore with in the dataset with some other variables that could show patterns or insights further.

# ### 7.1 Top 20 City Hotel bookings for 2017 by booking origin
# - A sub group consisting of top 20 customer booking country origin with the higest 'adr' (average daily rate).
# - It will be useful to conduct an analysis on common trends shared among  high performing City hotels, for the purposes of utility preparation ahead for the hotels busiest periods.

# In[177]:


# Creating a dataset for Top 20 countries with the higest adr (average daily rate)

columns = df_city.filter(items=['hotel','country','arrival_date_month','arrival_date_year','adults','adr'])

year_2017 = columns[(columns.arrival_date_year == 2017)]

Top20 = year_2017.sort_values('adr', ascending=False)

Top20.head(20)


# ### 7.2 Top 20 Resort Hotel bookings for 2017 by booking origin

# In[178]:


# Creating a dataset for Top 20 country boookings for resorts with the higest adr (average daily rate) in 2017

columns = df_resort.filter(items=['hotel','country','arrival_date_month','arrival_date_year','adults','adr'])

year_2017 = columns[(columns.arrival_date_year == 2017)]

Top20 = year_2017.sort_values('adr', ascending=False)

Top20.head(20)


# #### From the above analysis we have identified a few subgroups based on:
# 
# 1. The most hotel booked country and the months, specifically to explore and find patterns in the data that could later be utilised for industry operations in marketing key stake holders and how to allocate budgets according to seasons. 
# 
# 2. We took the second data frame to find the average daily rate pricing for people with babies, to analyse and find trends in the dataset.

# ***
# 
# 
# 
# ## Step 8: One-hot encoding
# 
# Alessia R.

# One-hot encoding is used to transform huge amounts of categorical data in numerical variables 0 and 1 representing 'False' and 'True', much easier to plot and to work within machine learning. Also, with one-hot encoding we create new variables that still refer to the original meaning of the dataframe, maintaining therefore the categories almost intact.
# 
# 
# ##### Details
# Because of our business case, we have reduced the original dataset to contain only 3 categorical variables:
# - **'hotel'** is composed by only 2 unique values, therefore the encoding wouldn't bring any value
# - **'is_canceled'** is a categorical data inasmuch binary, but because it already includes only numbers as 0 and 1 (equal to the result of encoding), it is not a good candidate to undego this process
# - **'country'** instead presents numerous string unique values, therefore it is much suitable for encoding

# ### 8.1 Encode the dataset
# Once again, we will use the 'confirmed_df' to generate from the previous Exploratory Data Analysis. This will ensure the encoding can reflect our business domain.

# In[179]:


confirmed_df.info()


# In[180]:


confirmed_df['country'].unique()


# In[181]:


# One-Hot encoding on the 'country' column

confirmed_df_onehot = confirmed_df.copy()
confirmed_df_onehot = pd.get_dummies(confirmed_df_onehot, columns=['country'], prefix=['country'])

confirmed_df_onehot.head()


# ***
# 
# 
# 
# 
# ## Step 9: PCA
# 

# #### What is PCA? (Prabhakaran, 2019)
# 
# PCA is a quite simple dimensionality reduction technique that transforms the columns of a dataset into a new set features called Principal Components (PCs).
# 
# The information contained in each column is the amount of variance it contains. The primary objective of Principal Components is to represent the information in the dataset with minimum columns possible.
# 
# #### Why do we use PCA? (Prabhakaran, 2019)
# 
# In practice PCA is used for two reasons:
# 
# ##### (1) Dimensionality Reduction:
# The information distributed across a large number of columns is transformed into principal components (PC) such that the first few PCs can explain a sizeable chunk of the total information (variance). These PCs can be used as explanatory variables in Machine Learning models.
# ##### (2) Visualize Classes: 
# Visualising the separation of classes (or clusters) is hard for data with more than 3 dimensions (features). With the first two PCs itself, it’s usually possible to see a clear separation.

# ### Use PCA to visualize Classes within our dataset
# For the purpose of this assignment we are going to conduct **PCA with a concrete business problem in our mind**. Namely, we want to see which months are usually the most most busy for each of our two hotels. Our dataset contains data from three consecutive years: *2015, 2016, and 2017*.
# 
# ##### Details
# Considering the above, our **target variable** will be **month - 'arrival_date_month'** column. 
# 
# For our PCA we will choose only these variables that can help us to answer our business question. Therefore, the final dataframe for our PCA will contain the following variables:
# 
#     (1) is_canceled (int64)
#     (2) arrival_date_month (object)
#     (3) arrival_date_week_number (int64)
#     (4) arrival_date_day_of_month (int64)
#     (5) stays_in_weekend_nights (int64)
#     (6) stays_in_week_nights (int64)
#     (7) adults (int64)
#     (8) Children (int64)
#     (9) Babies (int64)
#     (10) adr (float64)
# 
# The above dataset of variables contains both, numerical and categorical variables, and NOTE that we have only one continous variable which is adr (average daily rate).
# 
# Our categorical variable is actually our target veriable so before conducting PCA we are going to separate this variable from the numerical features. 
# 
# ##### Datesets preparation
# 
# Before we proceed with PCA, we have to extract the 10 variables listed above from the dataframe. 
# 
# As per business domain, we are in need of separating the 2 hotel types our original dataframe is built upon, in fact during the Exploratory Data Analysis we discovered there is much difference between the distribution of data of both Resort and City hotels. 
# 
# We will recycle the same datasets used before, and will drop the columns not needed.

# ### 9.1 Data preparation (Resort hotel)

# #### Drop unnecessary information
# We will drop the variables that we do not need by using .drop() function from Pandas, and using the remainig variables we will create a new dataframe called new_df_H1.

# In[182]:


df_resort.info()


# In[183]:


df_H1 = domain_df[domain_df['hotel'] == 'Resort Hotel']
df_H1


# In[184]:


new_df_H1 = df_H1.drop(['hotel','arrival_date_year', 'country'], axis=1)
new_df_H1.info()


# In[185]:


new_df_H1.shape


# So, our new_df_H1 has 10 columns and 40060 records.

# #### Clean dataset  (Resort hotel)
# Before we move any further, we are going just to make sure that there are no NULL values in our dataset.

# In[186]:


new_df_H1.isnull().sum()


# #### Adjust dataset for PCA  (Resort hotel)
# As arrival_date_month is our target variable we have to separate it from the numerical features before conducting PCA. To do so we will use .drop() function.

# In[187]:


pca_new_H1 = new_df_H1.drop(['arrival_date_month'], axis=1)
pca_new_H1.head()


# In[188]:


pca_new_H1.shape


# ### 9.2  PCA exploration  (Resort hotel)
# Before we proceed with PCA we want to see if there is correlation between any variables in the above dataframe. One way of doing this is to create a heatmap.

# In[189]:


# Seaborn Correlation plot 

fig, ax = plt.subplots(figsize=(20,20))  
sns.heatmap(pca_new_H1.corr(), annot=True, fmt='.2f', linewidths=1, ax=ax)


# On the above heatmap we can see that there is very little or no correlation between the majority of our features. We can see only two features that have moderate correlation (represented by the orange squares). This means that for sure we will need more than 2 components to explain our target variable.
# But at this stage this is only our assumption. we will see more after we conduct PCA.

# So, there are 9 variables in our pca_new_H1 dataset. Only 1 out of these 9 is continous. The other 8 variables is descrete. PCA gives the best results on continous variables. With the majority of descrete type of variables our results will not be as impresive as in the books. Nevertheless, let's proceed with our PCA.
# 
# As PCA is affected by the scale, we need to scale our features. 
# 
# We will use StandardScaler() to help us standardize the dataset’s features onto unit scale (mean = 0 and variance = 1) which is a requirement for the optimal performance of many machine learning algorithms. 

# In[190]:


pca_H1 = StandardScaler().fit_transform(pca_new_H1)
pca_H1


# In[191]:


pca_H1.shape


# We have to check whether the normalized data has a mean of zero and a standard deviation of one.

# In[192]:


np.mean(pca_H1), np.std(pca_H1)


# Both results are good enough.

# ### 9.3 PCA Projection to 2D  (Resort hotel)
# 
# In this section, we will try to project this 9 dimensional dataset into 2 dimensions.
# 
# Before we move any further - a bit of theory.
# 
# Principal components are new variables that are constructed as linear combinations or mixtures of the initial variables. These combinations are done in such a way that the new variables, called principal components, are uncorrelated and most of the information within the initial variables is squeezed or compressed into the first components (Jaadi, 2020). However, in our case this task might be difficult as almost all of our variables have no correlation with each other from the very start so all of them might be essential to explain the target variable.
# 
# So theory is as follows: our 9-dimensional data gives us 9 principal components, but PCA tries to put maximum possible information in the first component, then maximum remaining information in the second and so on. If our dataset was perfect, organizing information in principal components, would allow us to reduce dimensionality without losing much information, and this by discarding the components with low information and considering the remaining components as our new variables. 
# 
# Important to NOTE is that the principal components (PCs) are less interpretable and don’t have any real meaning since they are constructed as linear combinations of the initial variables.
# So, after dimensionality reduction, there isn’t a particular meaning assigned to each principal component. The new components are just the two main dimensions of variation.
# 
# So, let's try to reduce our 9-dimensions to 2 dimensions. We will do this using sklearn.decomposition.PCA from scikit-learn library.

# In[193]:


pca_H1 = PCA(n_components=2)
principalComponents = pca_H1.fit_transform(pca_new_H1)
pcDf_H1 = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])


# In[194]:


pcDf_H1.head()


# In[195]:


pcDf_H1.shape


# ##### Plot PCA graph
# So now our 9-dimensions is reduced into 2-dimensions. Next, we will try to plot this 2 dimensional data.
# 
# First, we will have to concatenate our target variable with the 2-dimensional dataframe above.

# In[196]:


finalDf_H1 = pd.concat([pcDf_H1, new_df_H1['arrival_date_month']], axis = 1)
finalDf_H1.head()


# In[197]:


finalDf_H1.shape


# In[198]:


# First, we want to determine the size of our 2D figure
fig = plt.figure(figsize = (20,20))

# Then, we design our plot
sns.set_style('whitegrid')
pca_h1 = fig.add_subplot(1,1,1)

# Naming the lables on axis
pca_h1.set_xlabel('Principal Component 1', fontsize = 15)
pca_h1.set_ylabel('Principal Component 2', fontsize = 15)
pca_h1.set_title('2 Components PCA - Resort Hotel', fontsize = 20)

# defining our targets and their colors
targets = ['January', 'February', 'March', 'April', 'May', 'June',
           'July', 'August', 'September', 'October', 'November', 'December']

colors = ['darkred','salmon','rosybrown','tomato','sandybrown','gold','olive','yellow',
          'darkolivegreen','darkseagreen','darkcyan','dodgerblue']

for target, color in zip(targets,colors):
    indicesToKeep = finalDf_H1['arrival_date_month'] == target
    pca_h1.scatter(finalDf_H1.loc[indicesToKeep, 'PC1'], finalDf_H1.loc[indicesToKeep, 'PC2'], 
               c=color, s = 100, edgecolors='w')
    
pca_h1.legend(targets,fontsize='xx-large', loc='best')
pca_h1.grid()


# On the above 2D projection we can see a few things. First, data from the Resort Hotel are grouped in 12 clusters that seem to overlap but only with the cluster directly next to them. Second, it seems that the largest amont of data is concentrated around the month August (yellow). The other 2 months that seem busy but not as much as August are July (olive) and June (gold). This is actually in line with or findings from EDA.
# 
# This might be much better visible in 3D projection. But before we move there, we want to know explained variance ratio.
# 
# ##### Explained Variance 
# 
# (Data Camp, 2020)
# Once we have the principal components, we can find the explained variance ratio. It will give us the amount of information or variance that each PC has when projecting the data to a lower dimensional subspace. By definition, when using PCA, we want to reduce the number of variables to a minimum, but in the same time we don't want to lose too much information. 
# 
# explained_variance_ratio_ help us to see how much variance each of our PCs holds and if the cumulative percentage is too small it means that we have to increase the number of the principal components to reach at least the total of 90-95%.

# In[199]:


print('Explained variation per principal component: {}'.format(pca_H1.explained_variance_ratio_))


# From the above output we can see that PC1 holds only 93.2% of information while PC2 holds only 4.7%.
# Also, the other point to note is that while projecting 9-dimensional data to a 2-dimensional data, only 2.1% of information was lost.
# 
# We could stop here because our score met the main purpose for conducting PCA, but we want to see how our dataset's prodejection would look like when described by 3 dimensions.
# 
# ### 9.4 PCA Projection to 3D  (Resort hotel)

# In[200]:


pca_H1 = PCA(n_components=3)
principalComponents = pca_H1.fit_transform(pca_new_H1)
pcDf_H1 = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3'])


# In[201]:


pcDf_H1.head()


# In[202]:


finalDf_H1 = pd.concat([pcDf_H1, new_df_H1['arrival_date_month']], axis = 1)
finalDf_H1.head()


# In[203]:


# Define the parameters for 3D plot

fig = plt.figure(figsize=(20, 20))
pca_3d_h1 = fig.add_subplot(111, projection='3d')

# Naming the lables on axis
pca_3d_h1.set_xlabel('Principal Component 1', fontsize = 15)
pca_3d_h1.set_ylabel('Principal Component 2', fontsize = 15)
pca_3d_h1.set_zlabel('Principal Component 3', fontsize = 15)
pca_3d_h1.set_title('3 Components PCA - Resort Hotel', fontsize = 20)

# defining our targets and their colors
targets = ['January', 'February', 'March', 'April', 'May', 'June',
           'July', 'August', 'September', 'October', 'November', 'December']

colors = ['darkred','salmon','rosybrown','tomato','sandybrown','gold','olive','yellow',
          'darkolivegreen','darkseagreen','darkcyan','dodgerblue']

for target, color in zip(targets,colors):
    indicesToKeep = finalDf_H1['arrival_date_month'] == target
    pca_3d_h1.scatter(finalDf_H1.loc[indicesToKeep, 'PC1'], finalDf_H1.loc[indicesToKeep, 'PC2'],
               finalDf_H1.loc[indicesToKeep, 'PC3'], c=color, s = 170, edgecolors='w')
    
pca_3d_h1.legend(targets, fontsize='xx-large', loc='best', bbox_to_anchor=(0.7, 0.7))
pca_3d_h1.grid()


# Three dimensional scatter plot clearly show that August (yellow) and July (olive) are the most busiest months. We also can see that data are separated in clusters. Now, let's see how much more information our 3 PCs hold.

# In[204]:


print('Explained variation per principal component: {}'.format(pca_H1.explained_variance_ratio_))


# From the above output we can see that PC1 holds 93.2% of information, PC2 holds 4.7% of information, and PC3 holds only 1.9%.
# 
# And like earlier, the other point to note is that while projecting 9-dimensional data to a 3-dimensional data, we further reduced the amount of lost information to 0.2%. 

# ### 9.5 Some quick ways to check the number of Principal Components

# An essential part of using PCA in practice is the ability to estimate the number of components needed to describe the data. The usual goal of PCA is to reduce the number of dimensions to a number that contains at least 90-95% of the information contained in the original dataset.
# #### (1)
# We can determine the number of principal components by looking at the cumulative explained_variance_ratio_ as a function of the number of components (VanderPlas, 2020):

# In[205]:


pca_H1_95 = PCA().fit(pca_new_H1)

plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()

xi = np.arange(1, 10, step=1)
y = np.cumsum(pca_H1_95.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 10, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance - Resort Hotel')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(2.0, 0.90, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()


# **(2)**
# 
# Another quick way to check the number of principal component that can give us 95% of explained variance is to run PCA that will reduce the original dataframe only to the number of compnents that can explain 95% of the variance (see below).

# In[206]:


pca_H1_95 = PCA(n_components=0.95)
principalComponents = pca_H1_95.fit_transform(pca_new_H1)


# In[207]:


pcDf_H1 = pd.DataFrame(data = principalComponents)
pcDf_H1.head()


# In[208]:


pcDf_H1.shape


# So, after running only 4 lines of code above we learned that 2 PCs would be enough to explain our target variable.

# **(3) Explained_Variance Table (Gouda, 2019)**
# 
# Below we have included explained_variance table that can be another way of determining how many components is needed.

# In[209]:


pca=PCA()  
pca.n_components=9  
pca_data=pca.fit_transform(pca_new_H1)

# look at explainded variance of PCA components 
exp_var_cumsum=pd.Series(np.round(pca.explained_variance_ratio_.cumsum(),4)*100)

for index,var in enumerate(exp_var_cumsum):  
    print('if n_components= %d,   variance=%f' %(index,np.round(var,3)))


# **(4) Scree Plot**
# 
# A scree plot is like a bar chart and it shows the size of each of the principal components. It helps us visualize the percentage of variation captured by each of the principal components (Kindson The Genius, 2019). 

# In[210]:


percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9']
plt.bar(x= range(1,10), height=percent_variance, tick_label=columns)
plt.ylabel('Percentage of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA Scree Plot - Resort Hotel')

y=percent_variance
for i, v in enumerate(y):
    xlocs=[i+1 for i in range(0,10)]
    plt.text(xlocs[i] - 0.25, v + 0.5, str(v)+'%')
    
plt.show()


# In the next part of this analysis we will repeat all the steps using City Hotel dataset.

# ### 9.6 Data Preparation (City hotel)
# Similarly as the previously analysed datased, the city hotel dataframe contains data from three consecutive years: 2015, 2016, and 2017.
# 
# Rememebering our business problem we are going to extract the following variables for our PCA:
# 
#     (1) is_canceled (int64)
#     (2) arrival_date_month (object)
#     (3) arrival_date_week_number (int64)
#     (4) arrival_date_day_of_month (int64)
#     (5) stays_in_weekend_nights (int64)
#     (6) stays_in_week_nights (int64)
#     (7) adults (int64)
#     (8) Children (int64)
#     (9) Babies (int64)
#     (10) adr (float64)
#     
# Our categorical variable is actually our target veriable so before conducting PCA we are going to separate this variable from the numerical features. 
# 
# But before we do this, we have to extract the 10 variables listed above from the main dataset.
# 
# We will use the same method as previously and we will call our new dataframe new_df_H2.

# In[211]:


df_H2 = domain_df[domain_df['hotel'] == 'City Hotel']
df_H2


# In[212]:


new_df_H2 = df_H2.drop(['hotel','arrival_date_year', 'country'], axis=1)
new_df_H2


# To avoid unnecessary problem when concatenating PC's dataframe with our target variable we will reset indexing for our dataframe. We will do this by using Pandas function .reset_index() and argument drop=True to avoid adding the column with an old index.

# In[213]:


new_df_H2 = new_df_H2.reset_index(drop=True)
new_df_H2


# Our new_df_H2 has 10 columns and 79330 records.

# #### Clean dataset  (City hotel)
# Before we move any further, we are going just to make sure that there are no NULL values in our dataset.

# In[214]:


new_df_H2.isnull().sum()


# #### Adjust dataset  for PCA (City hotel)
# As arrival_date_month is our target variable we have to separate it from the numerical features before conducting PCA.
# To do so we will use .drop() function.

# In[215]:


pca_new_H2 = new_df_H2.drop(['arrival_date_month'], axis=1)
pca_new_H2


# ### 9.7  PCA exporation (City hotel)
# Before we proceed with PCA we want to see if there is correlation between any variables in the above dataframe. One way of doing this is to create a heatmap. 
# 
# We also want to see if there are differences in results between two hotels.

# In[216]:


fig, ax = plt.subplots(figsize=(20,20))  
sns.heatmap(pca_new_H2.corr(), annot=True, fmt='.2f', linewidths=1, ax=ax)


# We can see clearly that there is difference between two hotels in relation to the correlation results. While in the Resort Hotel results we could see 2 variables that were moderately correlated, here none of the selected variables correlate with each other.
# 
# Let's what else is different.
# 
# As mentioned earlier, PCA is affected by the scale, so we have to scale our features.
# 
# We will use StandardScaler() to help us standardize the dataset’s features onto unit scale (mean = 0 and variance = 1) which is a requirement for the optimal performance of many machine learning algorithms.

# In[217]:


pca_H2 = StandardScaler().fit_transform(pca_new_H2)
pca_H2


# In[218]:


pca_H2.shape


# We have to check whether the normalized data has a mean of zero and a standard deviation of one.

# In[219]:


np.mean(pca_H2), np.std(pca_H2)


# It seems that it is.

# ### 9.8 PCA Projection to 2D (City hotel)
# 
# In this section, we will try to project this 9-dimensional dataset into 2-dimensions.
# 
# We will do this using sklearn.decomposition.PCA from scikit-learn library.

# In[220]:


pca_H2 = PCA(n_components=2)
princComp = pca_H2.fit_transform(pca_new_H2)
pcDf_H2 = pd.DataFrame(data = princComp, columns = ['PC1', 'PC2'])


# In[221]:


pcDf_H2


# So now our 9-dimensions is reduced into 2-dimensions. Next, we will try to plot this 2 dimensional data.
# 
# ##### Plot PCA graph
# 
# First, we will have to concatenate our target variable with the 2-dimensional dataframe above.

# In[222]:


finalDf_H2 = pd.concat([pcDf_H2, new_df_H2['arrival_date_month']], axis=1)
finalDf_H2


# In[223]:


# 2D plot

# First, we want to determine the size of our 2D figure
fig = plt.figure(figsize = (20,20))

# Then, we design our plot

ax = fig.add_subplot(1,1,1)

# Naming the lables on axis
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Components PCA - City Hotel', fontsize = 20)

# defining our targets and their colors
targets = ['January', 'February', 'March', 'April', 'May', 'June',
           'July', 'August', 'September', 'October', 'November', 'December']

colors = ['darkred','salmon','rosybrown','tomato','sandybrown','gold','olive','yellow',
          'darkolivegreen','darkseagreen','darkcyan','dodgerblue']

for target, color in zip(targets,colors):
    indicesToKeep = finalDf_H2['arrival_date_month'] == target
    ax.scatter(finalDf_H2.loc[indicesToKeep, 'PC1'], finalDf_H2.loc[indicesToKeep, 'PC2'], 
               c=color, s = 100, edgecolors='w')
    
ax.legend(targets,fontsize='xx-large', loc='best')
ax.grid()


# On this scatter plot we can see that each month is almost equally busy. There is also one outlier related to March data, which make it difficult to see the actual pattern. What we can say for sure is that each cluster of data overlap the next to it. Also, while the data from the Resort Hotel clearly indicated which months are the busisest, here data related to all months are almost equally distributed. In addition, because of one outlier we see this visualisation in a very small scale. If this outlier wasn't there we would be able to see more clearly a distribution of each cluster. This might be much better visible in 3D projection. 
# 
# As this outlier was bothering us, we have decided to check which record is actually causing the problem and if it affects any other part of our analysis.
# This is what we have found:
# 
# ![image.png](attachment:image.png)

# So, within the 79330 records there is one where adr value is insereted incorrectly. It seems that instead of 54.00, the value of 5400 was inserted. To make sure that our assumptions are correct we selected two other records from the same dataset that had similar characteristics. We can see that the average rate for the room category 'A' during the week days was in March 2016 between 40 and 57.73 if 2 adults were staying in the room.
# 
# To be on safe side, in our EDA we have concentrated mainly on our business question thus we acknowledged that there were 'cancelled' bookings in our dataset but they were not taken into consideration when we were looking at the months that are busy. 
# 
# For our PCA we took all bookings into consideration (including the cancelled one) as we wanted to see the clusters distribution, and thanks to that we actually found the reason for our strange statistics related to numerical data. As this is only one record which actually related to canelled booking we decided to remove it from the dataset. First, however, we will check the mean.

# In[224]:


pca_new_H2['adr'].mean()


# In[225]:


pca_new_H2[8455:8456]


# In[226]:


pca_new_H2=pca_new_H2.drop([8455])


# In[227]:


pca_new_H2['adr'].mean()


# So, mean didn't change much after replacemen, which is actually good. Let's see how our 2D projection will look right now. We will repeat all the necessary steps.
# First, we will check the shape of our updated dataframe.

# In[228]:


pca_new_H2.shape


# Next, we have to scale our data for PCA.

# In[229]:


pca_H2 = StandardScaler().fit_transform(pca_new_H2)
pca_H2


# Then we will check if the mean is 0 and sd=1

# In[230]:


np.mean(pca_H2), np.std(pca_H2)


# It is close enough, so now we will proceed with PCA.
# Now, we are going to reduce our 9-dimensions to 2-dimensions.

# In[231]:


pca_H2 = PCA(n_components=2)
princComp = pca_H2.fit_transform(pca_new_H2)
pcDf_H2 = pd.DataFrame(data = princComp, columns = ['PC1', 'PC2'])


# In[232]:


pcDf_H2


# Next, we are adding our target variable.

# In[233]:


finalDf_H2 = pd.concat([pcDf_H2, new_df_H2['arrival_date_month']], axis=1)
finalDf_H2


# In[234]:


# 2D Plot

# First, we want to determine the size of our 2D figure
fig = plt.figure(figsize = (20,20))

# Then, we design our plot

ax = fig.add_subplot(1,1,1)

# Naming the lables on axis
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Components PCA - City Hotel', fontsize = 20)

# defining our targets and their colors
targets = ['January', 'February', 'March', 'April', 'May', 'June',
           'July', 'August', 'September', 'October', 'November', 'December']

colors = ['darkred','salmon','rosybrown','tomato','sandybrown','gold','olive','yellow',
          'darkolivegreen','darkseagreen','darkcyan','dodgerblue']

for target, color in zip(targets,colors):
    indicesToKeep = finalDf_H2['arrival_date_month'] == target
    ax.scatter(finalDf_H2.loc[indicesToKeep, 'PC1'], finalDf_H2.loc[indicesToKeep, 'PC2'], 
               c=color, s = 100, edgecolors='w')
    
ax.legend(targets,fontsize='xx-large', loc='best')
ax.grid()


# As we can see the difference is huge. Now we can clearly see all the clusters. They seem a bit more messy than for the Resort Hotel but still the pattern is clear. The City Hotel is busy all year around. However, there are about 3 to 4 months where more data seems to be clustered. These are: May, June, July, August. It is close to the findings from our EDA analysis.
# 
# Before we move to 3D projection, we want to know explained variance ratio.

# In[235]:


print('Explained variation per principal component: {}'.format(pca_H2.explained_variance_ratio_))


# **From the above output we can see that PC1 holds only 85.66% of information while PC2 holds only 9.96%.**
# So, while projecting 9-dimensional data to a 2-dimensional data, we lost 4.38% of information. 
# 
# We hit our target of 95% so we could stop here, but we want to see how our dataset's prodjection would look like when described by 3 dimensions.
# 
# ### 9.9 PCA Projection to 3D (City hotel)

# In[236]:


pca_H2 = PCA(n_components=3)
principalComponents = pca_H2.fit_transform(pca_new_H2)
pcDf_H2 = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3'])


# In[237]:


pcDf_H2.head()


# In[238]:


finalDf_H2 = pd.concat([pcDf_H2, new_df_H2['arrival_date_month']], axis = 1)


# In[239]:


finalDf_H2.head()


# In[240]:


# Define the parameters for 3D plot

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

# Naming the lables on axis
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 Components PCA - City Hotel', fontsize = 20)

# defining our targets and their colors
targets = ['January', 'February', 'March', 'April', 'May', 'June',
           'July', 'August', 'September', 'October', 'November', 'December']

colors = ['darkred','salmon','rosybrown','tomato','sandybrown','gold','olive','yellow',
          'darkolivegreen','darkseagreen','darkcyan','dodgerblue']

for target, color in zip(targets,colors):
    indicesToKeep = finalDf_H2['arrival_date_month'] == target
    ax.scatter(finalDf_H2.loc[indicesToKeep, 'PC1'], finalDf_H2.loc[indicesToKeep, 'PC2'],
               finalDf_H2.loc[indicesToKeep, 'PC3'], c=color, s = 100, edgecolors='w')
    
ax.legend(targets, fontsize='xx-large', loc='best', bbox_to_anchor=(0.85, 0.8))
ax.grid()


# Three dimensional scatter plot clearly show that all months seem to be more less the same busy. We also can see that data are separated in less organised clusters then for the Resort Hotel. Now, let's see how much more information our 3 PCs hold.

# In[241]:


print('Explained variation per principal component: {}'.format(pca_H2.explained_variance_ratio_))


# **From the above output we can see that PC1 holds 85.66% of information, PC2 holds 9.96% of information, and PC3 holds only 4.19%.**
# 
# And like earlier, the other point to note is that while projecting 9-dimensional data to a 3-dimensional data, we further reduced the amount of lost information to 0.19%.
# 
# So, although a distribution of variation per principal component in both, Resort Hotel and City Hotel PCA is different, in both cases when using 3 compnent only approximately 0.2% of information is lost.

# ### 9.10 Some quick ways to check the number of Principal Components
# 
# In this section, similarly as during the earlier analysis wy are going to conduct 4 quick ways to determine the number of PC components.
# 
# **(1) The cumulative explained_variance_ratio_ as a function of the number of components**

# In[242]:


pca_H2_95 = PCA().fit(pca_new_H2)

plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()

xi = np.arange(1, 10, step=1)
y = np.cumsum(pca_H2_95.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 10, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance - City Hotel')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(2.0, 0.90, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()


# **(2) 95% n_components PCA**

# In[243]:


pca_H2_95 = PCA(n_components=0.95)
principalComponents = pca_H2_95.fit_transform(pca_new_H2)


# In[244]:


pcDf_H2 = pd.DataFrame(data = principalComponents)
pcDf_H2.head()


# **(3) Explained_Variance Table (Gouda, 2019)**

# In[245]:


pca=PCA()  
pca.n_components=9  
pca_data=pca.fit_transform(pca_new_H2)

# look at explainded variance of PCA components 
exp_var_cumsum=pd.Series(np.round(pca.explained_variance_ratio_.cumsum(),4)*100)

for index,var in enumerate(exp_var_cumsum):  
    print('if n_components= %d,   variance=%f' %(index,np.round(var,3)))


# **(4) Scree Plot**

# In[246]:


percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9']
plt.bar(x= range(1,10), height=percent_variance, tick_label=columns)
plt.ylabel('Percentage of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA Scree Plot - City Hotel')

y=percent_variance
for i, v in enumerate(y):
    xlocs=[i+1 for i in range(0,10)]
    plt.text(xlocs[i] - 0.25, v + 0.5, str(v)+'%')

plt.show()


# # Appendix

# ### From step 5: Scatterplot visualizations

# #### Some Graphs rejected due to poor correlation and further explanation
#  
#  The two graphs below are typical of many plots on the pairplot, horizontal or vertical bands
#  This can mean that the variables are constants or that we are looking at a comparison between 
#  categorical variables which may not be suited to correlation or regression
#  
#  http://www.unige.ch/ses/sococ/cl//stat/action/diagscat.html?

# In[247]:


# Seaborn relplot

sns.relplot(x='lead_time',y='children',data=df_num1)


# In[248]:


# Seaborn relplot

sns.relplot(x='total_of_special_requests',y='lead_time',data=df_num1)


# In[249]:


# Seaborn relplot

sns.relplot(x='arrival_date_week_number',y='arrival_date_day_of_month',data=df_num1)


# The mosaic pattern is confusing to the eye and as this is comparing constants, this may simply
# indicate these features, compared are not good candidates for investigating correlation

# #### Some graphs rejected due to poor correlation but otherwise initially visually interesting

# In[250]:


# Seaborn relplot

sns.relplot(x='lead_time',y='days_in_waiting_list',data=df_num1)


# In[251]:


# Seaborn relplot
# Seaborn lmplot

sns.relplot(x='lead_time',y='days_in_waiting_list',data=df_num1)
sns.lmplot(x='lead_time',y='days_in_waiting_list',data=df_num1)


# At a glance on pairplot this plot seemed like it might demonstrate some linearity but it is
# confused by the overlay of a dense horizontal constant-like pattern low on the Y axis and
# outliers and loosely correlated points. The line that can be drawn is not revealing and
# these features do not numerically exhibit correlation

# In[252]:


# Seaborn relplot
# Seaborn lmplot

sns.relplot(x='lead_time',y='adr',data=df_num1)
sns.lmplot(x='lead_time',y='adr',data=df_num1)


# This plot looked interesting from a glance at the pairplot however on closer inspection it 
# appears it may partly show a non-linear relationship between the two features, there is high
# density, overplotting, outliers and the two features are not numerically well correlated
# so any non-linear relationship is difficult to precisely identify

# #### Note
# All other subplots from the pairplot were rejected due to loosely clustered datapoints, constants or constants combined with outliers or the above or difficulty of establishing a relationship or generally low correlational relationships

# In[253]:


# Seaborn relplot
# Seaborn Scatterplot

x = df_num1['adults']
y = df_num1['adr']


fig, scatter = plt.subplots(figsize = (10,6), dpi = 100)
scatter = sns.scatterplot(x = x, y =y, data=df_num2 )
scatter.set_title('ADR vs adults', fontsize = 23, y =1.05);

sns.lmplot(x='adults',y='adr',data=df_num1)
ax = plt.gca()
ax.set_title('adr vs adults: resort hotel')


# ### From Step 6: Exploratory data analysis

# In[254]:


# outlier adr value 
pd.set_option('max_columns', None)
df.loc[48515:48517]


# In[255]:


# Seaborn Countplot for Countries that have booked the two types of hotels together. 

plt.figure(figsize=(16,30))
sns.countplot(y= 'country', data = confirmed_df)


# # Bibliography
# 
# Antonio, N., de Almeida, A. and Nunes, L. (2019) ‘Hotel Booking Demand Datasets’. Data in Brief, 22, pp. 41–49. DOI: 10.1016/j.dib.2018.11.126.
# 
# AV. (2016) PCA: Practical Guide to Principal Component Analysis in R & Python. Analytics Vidhya. Available at: https://www.analyticsvidhya.com/blog/2016/03/pca-practical-guide-principal-component-analysis-python/ (Accessed: 21 February 2021).
# 
# Data Camp. (2020) (Tutorial) Principal Component Analysis (PCA) in Python. DataCamp Community. Available at: https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python (Accessed: 21 February 2021).
# 
# Galarnyk, M. (2021) PCA Using Python (Scikit-Learn). Medium. Available at: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60 (Accessed: 27 February 2021).
# 
# Gouda, S. (2019) PCA(Principal Component Analysis) In Python. Medium. Available at: https://medium.com/@sarayupgouda/pca-principal-component-analysis-in-python-f9836c25acb9 (Accessed: 28 February 2021).
# 
# Jaadi, Z. (2020) A Step-by-Step Explanation of Principal Component Analysis. Built In. Available at: https://builtin.com/data-science/step-step-explanation-principal-component-analysis (Accessed: 21 February 2021).
# 
# Kindson The Genius. (2019) Principal Components Analysis(PCA) in Python - Step by Step. Kindson The Genius. Available at: https://www.kindsonthegenius.com/principal-components-analysispca-in-python-step-by-step/ (Accessed: 12 March 2021).
# 
# Matplotlib. (2021) List of Named Colors — Matplotlib 3.3.4 Documentation. matplotlib.org. Available at: https://matplotlib.org/stable/gallery/color/named_colors.html (Accessed: 24 February 2021).
# 
# mikulskibartosz. (2019) PCA — How to Choose the Number of Components?. Bartosz Mikulski. Available at: https://mikulskibartosz.name/pca-how-to-choose-the-number-of-components/ (Accessed: 24 February 2021).
# 
# Pandas. (2021a) Pandas.DataFrame.Reset_index — Pandas 1.2.3 Documentation. pandas.pydata.org. Available at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html (Accessed: 13 March 2021).
# 
# Pandas. (2021b) Pandas.Set_option — Pandas 1.2.2 Documentation. pandas.pydata.org. Available at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html (Accessed: 20 February 2021).
# 
# Pandey, S. (2019) Get List of Pandas Dataframe Columns Based on Data Type - Intellipaat. Available at: https://intellipaat.com/community/23269/get-list-of-pandas-dataframe-columns-based-on-data-type (Accessed: 12 March 2021).
# 
# Prabhakaran, S. (2019) Principal Component Analysis (PCA) - Better Explained. ML+. Available at: https://www.machinelearningplus.com/machine-learning/principal-components-analysis-pca-better-explained/ (Accessed: 25 February 2021).
# 
# Pramoditha, R. (2020) Principal Component Analysis (PCA) with Scikit-Learn. Medium. Available at: https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0 (Accessed: 24 February 2021).
# 
# Sarkar, D. (DJ). (2018) The Art of Effective Visualization of Multi-Dimensional Data. Medium. Available at: https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57 (Accessed: 24 February 2021).
# 
# scikit-learn. (2020) Sklearn.Decomposition.PCA — Scikit-Learn 0.24.1 Documentation. scikit-learn. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html (Accessed: 12 March 2021).
# 
# stackoverflow. (2021) Python - Get List of Pandas Dataframe Columns Based on Data Type. Stack Overflow. Available at: https://stackoverflow.com/questions/22470690/get-list-of-pandas-dataframe-columns-based-on-data-type (Accessed: 20 February 2021).
# 
# VanderPlas, J. (2020) In Depth: Principal Component Analysis | Python Data Science Handbook. Python Data Science Handbook. Available at: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html (Accessed: 27 February 2021).
# 

# In[ ]:




