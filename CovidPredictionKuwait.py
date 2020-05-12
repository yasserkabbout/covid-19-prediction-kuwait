# importing the required libraries
import pandas as pd

# Visualisation libraries
import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium 
from folium import plugins
from fbprophet import Prophet

# Manipulating the default plot size
plt.rcParams['figure.figsize'] = 10, 12


df = pd.read_csv('covid_19_clean_complete_kuwait.csv',parse_dates=['Date'])
df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)


confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()
deaths = df.groupby('Date').sum()['Deaths'].reset_index()
recovered = df.groupby('Date').sum()['Recovered'].reset_index()
casesPerDay = df.groupby('Date').sum()['CasesPerDay'].reset_index()
recoveredPerDay = df.groupby('Date').sum()['RecoveredPerDay'].reset_index()
deathPerDay = df.groupby('Date').sum()['DeathPerDay'].reset_index()
activeCases = df.groupby('Date').sum()['ActiveCases'].reset_index()
#italyCasesPerDay = df.groupby('Date').sum()['ItalyCasesPerDay'].reset_index()

fig = go.Figure()
#Plotting datewise cases per day

# fig.add_trace(go.Scatter(x=confirmed['Date'], y=confirmed['Confirmed'], mode='lines+markers', name='Confirmed',line=dict(color='blue', width=2)))
# fig.add_trace(go.Scatter(x=deaths['Date'], y=deaths['Deaths'], mode='lines+markers', name='Deaths', line=dict(color='Red', width=2)))
# fig.add_trace(go.Scatter(x=recovered['Date'], y=recovered['Recovered'], mode='lines+markers', name='Recovered', line=dict(color='Green', width=2)))
# fig.add_trace(go.Scatter(x=casesPerDay['Date'], y=casesPerDay['CasesPerDay'], mode='lines+markers', name='CasesPerDay',line=dict(color='orange', width=2)))

fig.add_trace(go.Bar(x=casesPerDay['Date'], y=casesPerDay['CasesPerDay']))
#fig.add_trace(go.Bar(x=italyCasesPerDay['Date'], y=italyCasesPerDay['ItalyCasesPerDay']))


fig.update_layout(title='Kuwait NCOVID-19 Cases', xaxis_tickfont_size=14,yaxis=dict(title='Number of Active Cases'))
fig.show()


# Initializing columns for prophet
confirmed.columns = ['ds','y']
deaths.columns =['ds','y']
recovered.columns=['ds','y']
casesPerDay.columns=['ds','y']
recoveredPerDay.columns = ['ds','y']
deathPerDay.columns = ['ds','y']
activeCases.columns = ['ds','y']
#italyCasesPerDay.columns = ['ds','y']

# initializing timeline for prophet
confirmed['ds'] = pd.to_datetime(confirmed['ds'])
deaths['ds'] = pd.to_datetime(deaths['ds'])
recovered['ds'] = pd.to_datetime(recovered['ds'])
casesPerDay['ds']=pd.to_datetime(casesPerDay['ds'])
recoveredPerDay['ds'] = pd.to_datetime(recoveredPerDay['ds'])
deathPerDay['ds'] = pd.to_datetime(deathPerDay['ds'])
activeCases['ds'] = pd.to_datetime(activeCases['ds'])
#italyCasesPerDay['ds'] = pd.to_datetime(ItalyCasesPerDay['ds'])

# prophet model
m = Prophet(mcmc_samples=300)
#m = Prophet(interval_width=0.95)
m.fit(casesPerDay)
#m.fit(italyCasesPerDay)
# predicting the future for 20 periods (20 days)
future = m.make_future_dataframe(periods=20)
future.tail()

#predicting the future with date, and upper and lower limit of y value
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m.plot(forecast)
plt.show()