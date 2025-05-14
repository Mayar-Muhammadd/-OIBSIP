#Mayar Muhammad Elsayed Muhammad Rashwan
import pandas as pd

# load the CSV file
df = pd.read_csv("sales.csv", encoding='ISO-8859-1')

# show first few rows
print(" First rows:")
print(df.head())


#1)Data Loading and Cleaning: Load the retail sales dataset.


# check for basic info and missing values
print("\n Data info:")
print(df.info())

print("\n Missing values:")
print(df.isnull().sum())

# 4. Fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Gender'].fillna('Unknown', inplace=True)
df['Total Amount'].fillna(df['Total Amount'].median(), inplace=True)
df['Quantity'].fillna(0, inplace=True)

# clean negative values in 'quantity' set them to 0 (or a reasonable number)
df['Quantity'] = df['Quantity'].apply(lambda x: max(x, 0))

# convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# remove duplicate rows
df.drop_duplicates(inplace=True)

# final check
print("\n Cleaned data:")
print(df)

print("\n Summary:")
print(df.describe())
#the data is already clean and nothing has changed. no need for the next step, but i'll do it anyways
df.to_csv("cleaneddata.csv", index=False)

#2)Descriptive Statistics: Calculate basic statistics (mean, median, mode, standard deviation).


# load your cleaned CSV file
df = pd.read_csv("cleaneddata.csv", encoding='ISO-8859-1')

# mean (Average)
mean_age = df['Age'].mean()
mean_amount = df['Total Amount'].mean()
mean_qty = df['Quantity'].mean()
mean_price = df['Price per Unit'].mean()


# median 
median_age = df['Age'].median()
median_amount = df['Total Amount'].median()
median_qty = df['Quantity'].median()
median_price = df['Price per Unit'].median()



# mode
mode_age = df['Age'].mode()[0]
mode_amount = df['Total Amount'].mode()[0]
mode_qty = df['Quantity'].mode()[0]
mode_price = df['Price per Unit'].mode()[0]


# standard deviation (how spread out the numbers are)
std_age = df['Age'].std()
std_amount = df['Total Amount'].std()
std_qty = df['Quantity'].std()
std_price = df['Price per Unit'].std()




# output 

print(f"Age => Mean: {mean_age:.2f}, Median: {median_age}, Mode: {mode_age}, Std: {std_age:.2f}")
print(f"Total Amount => Mean: {mean_amount:.2f}, Median: {median_amount}, Mode: {mode_amount}, Std: {std_amount:.2f}")
print(f"Quantity => Mean: {mean_qty:.2f}, Median: {median_qty}, Mode: {mode_qty}, Std: {std_qty:.2f}")
print(f"Price Per Unit => Mean: {mean_price:.2f}, Median: {median_price}, Mode: {mode_price}, Std: {std_price:.2f}")

#3) Time Series Analysis: Analyze sales trends over time using time series techniques.
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# convert date to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# set date as the index for time series operations
df.set_index('Date', inplace=True)

# optional: sort the index just in case
df.sort_index(inplace=True)

# resample to monthly sales (sum of 'total amount' per month)
monthly_sales = df['Total Amount'].resample('M').sum()

# plot the raw monthly sales trend
plt.figure(figsize=(10, 5))
monthly_sales.plot(marker='o', title="Monthly Sales Trend (Amount)")
plt.xlabel('Date')
plt.ylabel('Sales Amount')
plt.grid(True)
plt.tight_layout()
plt.show()

# rolling average (e.g. 3-month moving average)
rolling_avg = monthly_sales.rolling(window=3).mean()

# plot original vs rolling average
plt.figure(figsize=(10, 5))
monthly_sales.plot(label='Monthly Sales', color='skyblue')
rolling_avg.plot(label='3-Month Rolling Avg', color='orange')
plt.title('Monthly Sales with Rolling Average')
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




#4) Customer and Product Analysis: Analyze customer demographics and purchasing behavior.



import seaborn as sns

# load the dataset
df = pd.read_csv("cleaneddata.csv", encoding='ISO-8859-1')


# date column is datetime 
df['Date'] = pd.to_datetime(df['Date'])

#CUSTOMER ANALYSIS

# gender distribution
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(5, 5))
gender_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightpink'])
plt.title('Gender Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()

# age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=10, kde=True, color='purple')
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# purchases by gender
gender_purchase = df.groupby('Gender')['Total Amount'].sum()
plt.figure(figsize=(6, 4))
gender_purchase.plot(kind='bar', color=['blue', 'pink'])
plt.title('Total Purchase Amount by Gender')
plt.ylabel('Total Amount')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# average spending by age group
df['Age Group'] = pd.cut(df['Age'], bins=[0, 18, 25, 35, 50, 100],
                         labels=['<18', '18-25', '26-35', '36-50', '50+'])
age_group_avg = df.groupby('Age Group')['Total Amount'].mean()
plt.figure(figsize=(7, 4))
age_group_avg.plot(kind='bar', color='teal')
plt.title('Average Purchase Amount by Age Group')
plt.ylabel('Average Amount')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# PRODUCT ANALYSIS

# most popular product categories
product_counts = df['Product Category'].value_counts()
plt.figure(figsize=(8, 5))
product_counts.plot(kind='bar', color='orange')
plt.title('Most Purchased Product Categories')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# revenue per product category
product_revenue = df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
product_revenue.plot(kind='bar', color='green')
plt.title('Total Revenue by Product Category')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# average amount spent per product category
product_avg = df.groupby('Product Category')['Total Amount'].mean().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
product_avg.plot(kind='bar', color='darkred')
plt.title('Average Spend per Product Category')
plt.ylabel('Average Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#5 Visualization: Present insights through bar charts, line plots, and heatmaps.

# make sure 'Date' is datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
print(df.dtypes)
# Set 'Date' as the index of the DataFrame
df.set_index('Date', inplace=True)

# confirm that the index is now a DatetimeIndex
print(df.index)




# bar chart for Product Categories
# count of purchases per product category
product_counts = df['Product Category'].value_counts()
plt.figure(figsize=(8, 5))
product_counts.plot(kind='bar', color='orange')
plt.title('Most Purchased Product Categories')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# line Plot for Sales Over Time

# resample monthly sales
monthly_sales = df['Total Amount'].resample('ME').sum()

# Plot the sales trend over time
plt.figure(figsize=(10, 5))
monthly_sales.plot(marker='o', color='skyblue')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount')
plt.grid(True)
plt.tight_layout()
plt.show()

#  heatmap of correlations between numeric columns

# calculate correlations between numeric features (e.g., 'Age', 'Quantity', 'Price per Unit', 'Amount')
corr = df[['Age', 'Quantity', 'Price per Unit', 'Total Amount']].corr()

# plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
#6)Recommendations: Provide actionable recommendations based on the EDA.
#based on the **exploratory data analysis (EDA) of the dataset:

#1)target age groups with higher spending

#from the bar chart showing "Average Purchase Amount by Age Group", customers in the "<18" age group tend to spend more than other age groups.
#recommendation:

#marketing targeting: Focus marketing campaigns, offers, and promotions specifically towards customers aged<18 as they exhibit the highest average spending.
#product Development: Consider launching new products that appeal to this age. ex: clothing and beauty 
#2)gender specific marketing strategiy 

#the gender distribution pie chart shows a significant proportion of customers in the dataset, with slightly more females than males.
#recommendation:

#personalized marketing: create gender-targeted campaigns or product lines. For example, beauty and clothing category
#customer loyalty programs: build programs that specifically address gender preferences.

#3)focus on monthly sales trends

#the monthly sales trend line plot showed certain months with higher sales like may or lower sales like september 
#recommendation:
#promotions during low sales months:consider launching special promotions, discounts, or seasonal campaigns during those periods to boost sales.
#stock management: based on the sales trend, optimize inventory to ensure high-demand products are always available during peak months.


#4)Customer Segmentation for Personalized Offers

#the dataset contains valuable customer information like age, gender
#recommendation:
#segmentation: segment customers based on their purchasing behavior, age, gender and design personalized offers that helps specifically to these segments.
#loyalty Programs: develop loyalty programs tailored to frequent buyers, incentivizing repeat purchases with discounts, special access, or rewards points.

#5)optimize pricing strategy

#The price per unit varies across different product categories and age groups.
#recommendation:

#dynamic Pricing: implement a dynamic pricing strategy where the price is adjusted based on demand, competitor pricing, and customer segments.
#promotions and discounts: offer time-limited discounts to push sales for products that have high inventory but low demand.

#6)Improve Customer Experience with Data

#Observation:the dataset provides insights into customer behavior, like their spending patterns over time.(but the data i worked with has only unique customers with unique ids)
#recommendation:

  #customer Journey Mapping: use this data to map out the customer journey and identify touchpoints where customers drop off. This could help improve the website, user interface, or checkout process.
  #post-Purchase Engagement: consider sending follow-up emails or offering incentives for repeat purchases after a customer buys something.


#summary of recommendations:

#1. target marketing efforts towards the <18 age group to capitalize on higher spending.
#2. develop gender-targeted campaigns for a personalized shopping experience.
#3. launch promotions during months with lower sales to boost engagement.
#4. segment customers for personalized offers based on demographics and behavior.
#5. implement dynamic pricing strategies based on demand and customer segments.
#6. improve customer experience using insights from the data to improve the buying process and engage customers post-purchase.


