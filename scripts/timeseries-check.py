import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
df = pd.read_csv(
    "C:\\Users\\ssbap\\US-Consumer-Complaints-Forecasting\\data\\raw\\cleaned_consumer_complaints.csv"
    # parse_dates=['date_received'],
    # index_col='date_received'
)

# Quick peek at the data
print(df.shape)
weekly = df.resample('W').size()

monthly = df.resample('M').size()

plt.figure(figsize=(10, 4))
weekly.plot(title='Weekly Complaint Counts')
plt.ylabel('Number of Complaints')
plt.xlabel('Week')
plt.show()

plt.figure(figsize=(10, 4))
monthly.plot(title='Monthly Complaint Counts')
plt.ylabel('Number of Complaints')
plt.xlabel('Month')
plt.show()

rolling_4w = weekly.rolling(window=4).mean()
rolling_12m = monthly.rolling(window=12).mean()

plt.figure(figsize=(10, 4))
rolling_4w.plot(title='4-Week Rolling Mean of Weekly Complaints')
plt.ylabel('Rolling Mean (4 weeks)')
plt.xlabel('Week')
plt.show()

plt.figure(figsize=(10, 4))
rolling_12m.plot(title='12-Month Rolling Mean of Monthly Complaints')
plt.ylabel('Rolling Mean (12 months)')
plt.xlabel('Month')
plt.show()

monthly_df = monthly.to_frame(name='complaints')
monthly_df['month'] = monthly_df.index.month

plt.figure(figsize=(10, 6))
monthly_df.boxplot(column='complaints', by='month')
plt.title('Monthly Complaint Distribution')
plt.suptitle('')  # remove automatic subtitle
plt.xlabel('Month')
plt.ylabel('Number of Complaints')
plt.show()
result = seasonal_decompose(monthly, model='additive')

plt.rcParams.update({'figure.figsize': (10, 8)})
result.plot()
plt.show()
