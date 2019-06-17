
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from keras.regularizers import l2
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping, ModelCheckpoint


# # Data manipulation
# - classify response variable as the ground truth
# - set training and testing set

# In[2]:


df = pd.read_csv("C:/Users/Administrator/Desktop/SIT/course/BIA656/Project/Return.csv")
df = df.drop('Date', axis = 1)

df['SPY'] = np.where(df['SPY'] > 0, 1, 0)  # classify SPY as ground truth
df.head()
#df.isnull().sum()

# Split dataset into training and testing set
X = df.values[:, 0:56]
y = df.values[:, 56]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
print('sample size:', df.shape)
print('training size:', X_train.shape)
print('testing size:', X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Neural network with original data
# - Build model and tune parameters
# - Set early stop criteria to find the lowest lost

# In[3]:


# Build neural network model
lam = 0.01 # regularization parameter
model = Sequential()
model.add(Dense(12, input_dim = 56, kernel_regularizer = l2(lam), activation = 'relu', name = 'L2')) # first hidden layer
model.add(Dense(8, activation = 'relu', kernel_regularizer = l2(lam), name = 'L3')) # second hidden layer
model.add(Dense(1, activation = 'sigmoid', name = 'Output')) # output layer
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

# Use early stopping to find the best model
BEST_MODEL_FILEPATH="best_model"

# Define early stopping based on validation loss
# If three iterations in a row, validation loss is not improved compared with the previous one, stop training
# Mode='min' indicate the loss needs to decrease 
earlyStopping=EarlyStopping(monitor='val_loss',                             patience=3, verbose=2,                             mode='min')

# Define checkpoint to save best model which has max. validation acc
checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH,                              monitor='val_acc',                              verbose=2,                              save_best_only=True,                              mode='max')

# Fit model
training = model.fit(X_train, y_train, shuffle = True, 
                     epochs = 1000, batch_size = 36,
                     callbacks=[earlyStopping, checkpoint],
                     validation_data=[X_test, y_test], verbose = 2)


# In[4]:


# Covert the fitting history from dictionary to dataframe
history=pd.DataFrame.from_dict(training.history)
history.columns=["train_loss", "train_acc",             "val_loss", "val_acc"]
history.index.name='epoch'
#print(history)

# Plot fitting history
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3));
history[["train_acc", "val_acc"]].plot(ax=axes[0]);
history[["train_loss", "val_loss"]].plot(ax=axes[1]);
plt.show();

# Evaluate prediction
pred = model.predict(X_test)
pred = np.reshape(pred, -1)
pred = np.where(pred > 0.6, 1, 0)
print(metrics.classification_report(y_test, pred, labels = [0, 1]))


# # Neural network with PCA
# - Check correlation between variables to decide whether necessary to perform dimension reduction
# - Find proper number of principal components that contains most information of original data
# - Perform neural network again to see the performance of prediction model

# In[5]:


# Investigate predictors' correlation
sns.set(style="white")

# Compute the correlation matrix
corr = df.drop('SPY', axis = 1).corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show();


# In[6]:


# PCA
df = pd.read_csv("C:/Users/Administrator/Desktop/SIT/course/BIA656/Project/Return.csv")
df = df.drop('Date', axis = 1)
df['SPY'] = np.where(df['SPY'] > 0, 1, 0)
data = df.drop('SPY', axis = 1)

# Try to find the number of principle components which have 99% information of the original dataset
pca = PCA(n_components = 0.99)
pca.fit(data)
print('Variance ratio of each pc:\n', pca.explained_variance_ratio_, '\n')
print('Explained variance of each pc:\n', pca.explained_variance_, '\n')
print('Selected {} pcs'.format(pca.n_components_))

# Select the first 15 principle components according to the previous results
pca = PCA(n_components = 15)
pca.fit(data)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)


# In[7]:


# Build transformed dataset
dataDecomp = pca.transform(data)
dataDecomp = pd.DataFrame(dataDecomp)
df_pca = pd.concat([dataDecomp, df['SPY']], axis = 1)
df_pca.head()

# Fit in the same NN model with transformed dataset
X_pca = df_pca.values[:, 0:15]
y_pca = df_pca.values[:, 15]
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y_pca, test_size = 0.2, random_state = 123)
print(X_pca_train.shape, y_pca_train.shape, X_pca_test.shape, y_pca_test.shape)

lam = 0.01
model = Sequential()
model.add(Dense(12, input_dim = 15, kernel_regularizer = l2(lam), activation = 'relu', name = 'L2'))
model.add(Dense(8, activation = 'relu', kernel_regularizer = l2(lam), name = 'L3'))
model.add(Dense(1, activation = 'sigmoid', name = 'Output'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

BEST_MODEL_FILEPATH="best_model"
earlyStopping=EarlyStopping(monitor='val_loss',                             patience=3, verbose=2,                             mode='min')

checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH,                              monitor='val_acc',                              verbose=2,                              save_best_only=True,                              mode='max')

training_pca = model.fit(X_pca_train, y_pca_train, shuffle = True, 
                     epochs = 1000, batch_size = 36,
                     callbacks=[earlyStopping, checkpoint],
                     validation_data=[X_pca_test, y_pca_test], verbose = 2)


# In[8]:


# Plot fitting history
history_pca=pd.DataFrame.from_dict(training_pca.history)
history_pca.columns=["train_loss", "train_acc",                      "val_loss", "val_acc"]
history_pca.index.name='epoch'
#print(history_pca)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3));
history_pca[["train_acc", "val_acc"]].plot(ax=axes[0]);
history_pca[["train_loss", "val_loss"]].plot(ax=axes[1]);
plt.show();

# Evaluate prediction
pred_pca = model.predict(X_pca_test)
pred_pca = np.reshape(pred_pca, -1)
pred_pca = np.where(pred_pca > 0.6, 1, 0)
print(metrics.classification_report(y_pca_test, pred_pca, labels = [0, 1]))


# # Univariate time series analysis

# In[288]:


import statsmodels.api as sm
from pylab import rcParams
import itertools
from statsmodels.tsa.statespace.varmax import VARMAX


# ## Data manipulation
# - Transform time series data into the proper timestamp form

# In[ ]:


# Univariate time series analysis and forcasting
df1 = pd.read_csv("C:/Users/Administrator/Desktop/SIT/course/BIA656/Project/RAW_DATA.csv")
df1['Date'] = pd.to_datetime(df1.Date, format = '%Y-%m-%d')
ts = df1.loc[:, ['Date', 'SPY']]
#ts.isnull().sum()

# Indexing with time series
datetime = []
for date in ts['Date']:
    datetime.append(np.datetime64(date))
ts['Date_time'] = datetime
ts = ts.drop('Date', axis = 1)
ts = ts.set_index('Date_time')
#ts.head()
ts.index

# Use start of each month as the timestamp
y = ts['SPY'].resample('MS').mean()
#y.head()


# In[286]:


# Visualization
y.plot(figsize = (15, 6))
plt.show();

# Time-series decomposition: trend, seasonality, and noise
rcParams['figure.figsize'] = 12, 7
decomp = sm.tsa.seasonal_decompose(y, model = 'additive')
fig = decomp.plot()
plt.show();


# ## ARIMA model
# - Tune model parameters by grid search
# - Fit model and check diagnostics
# - Validate prediction with the ground truth

# In[202]:


# Forcasting with ARIMA(Autoregressive Integrated Moving Average)
# Grid-search best parameter for seasonality, trend, and noise
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y, order = param,                                           seasonal_order = param_seasonal,                                           enforce_stationarity = False,                                           enforce_invertibility = False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[203]:


# Fit model using a set of parameters with the lowest AIC
mod = sm.tsa.statespace.SARIMAX(y, order = (0, 1, 1),
                               seasonal_order = (0, 1, 1, 12),
                               enforce_stationarity = False,
                               enforce_invertibility = False)
results = mod.fit()
print(results.summary().tables[1])

# Model diagnostics
results.plot_diagnostics(figsize = (16, 8))
plt.show();


# In[291]:


# Validate forecast
pred = results.get_prediction(start = pd.to_datetime('2017-01-01'), dynamic = False)
pred_ci = pred.conf_int()

ax = y['2014':].plot(label = 'observed')
pred.predicted_mean.plot(ax = ax, label = 'One-step ahead Forcast', alpha = .7, figsize = (14, 7))
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color = 'k', alpha = .2)
ax.set_xlabel('Date')
ax.set_ylabel('SPY')
plt.legend()
plt.show();

# Numerically evaluate forecast with MSE and RMSE
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]

mse = ((y_forecasted - y_truth)**2).mean()
print('Mean squared error of our forcast is {}'.format(round(mse, 2)))
print('Root mean squared error of our forcast is {}'.format(round(np.sqrt(mse), 2)))
print('SPY range: {} - {}'.format(round(min(ts['SPY']), 1), round(max(ts['SPY']), 1)))

