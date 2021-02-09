##########################################################
# Customer Lifetime Value Calculation
##########################################################

##########################################################
# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
##########################################################

# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', 20)
# pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_excel("Datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

##################################################
# DATA PREPARATION
##################################################

# chose the ones by using ~ which do not contains "C"
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df['Quantity'] > 0)]
df.dropna(inplace=True)
df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_df = df.groupby('Customer ID').agg({'Invoice': lambda x: len(x),
                                         'Quantity': lambda x: x.sum(),
                                         'TotalPrice': lambda x: x.sum()})

# total transaction, total unit, total price
cltv_df.columns = ['total_transaction', 'total_unit', 'total_price']

cltv_df.head()


############################
# Average Order Value
############################

# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin

# Customer_Value = Average_Order_Value * Purchase_Frequency
cltv_df["avg_order_value"] = cltv_df["total_price"]/ cltv_df["total_transaction"]


#####################
#Purchase Frequency
#####################
#customer number:
cltv_df.shape[0]

cltv_df["purchase_frequency"] = cltv_df["total_transaction"]/cltv_df.shape[0]

##################
# Repeat Rate & Churn Rate
##################

#if total transaction is bigger than this means that these customers were here more than once
repeat_rate = cltv_df[cltv_df.total_transaction > 1].shape[0]/cltv_df.shape[0]
churn_rate = 1- repeat_rate


###############
#Profit Margin
################

# profit margin : 5 %
cltv_df["profit_margin"] = cltv_df["total_price"]* 0.05

##################
# Calculate Customer Lifetime Value
##################

# customer value:
cltv_df["CV"] = (cltv_df["avg_order_value"] * cltv_df["purchase_frequency"])/ churn_rate

cltv_df["CLTV"] = cltv_df["CV"] * cltv_df["profit_margin"]

cltv_df.sort_values("CLTV", ascending = False)

scaler = MinMaxScaler(feature_range= (1,100))
scaler.fit(cltv_df[["CLTV"]])
cltv_df["SCALED_CLTV"] = scaler.transform(cltv_df[["CLTV"]])


cltv_df.sort_values("CLTV", ascending = False)

cltv_df[["total_transaction", "total_unit","total_price","CLTV", "SCALED_CLTV"]].sort_values(by = "SCALED_CLTV",ascending= False).head()

cltv_df["Segment"] = pd.qcut(cltv_df["SCALED_CLTV"], 4, labels = ["D", "C", "B", "A"])

cltv_df.groupby("Segment")[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].agg(
    {"count", "mean", "sum"})

