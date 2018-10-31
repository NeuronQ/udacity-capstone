# Capstone Project
Machine Learning Engineer Nanodegree

## I. Definition [1-2 pages]

### Project overview [1-2 paragraphs]

Building **models that can predict the future evolution of easily tradeable financial tools such as stocks, or of crypto-currencies such as Bitcoin**, is a topic of great interest for traders and investors. Such predictions can guide both investment strategies and speculative trading. Using so called **deep leaning techniques** (multi-layered neural networks) to build such models is still in its infancy, and **and recurring neural networks** (RNNs) seem a good fit for this kind time-dependent data. In this report we are interested in building predictive models for trading, investigating whether we can obtain good enough predictions to guide potentially successful trading strategies. The models we are building and tuning are **long short-term memory (LSTM) RNNs**.

These problem we are attacking here is *just one particular case of applying machine learning to the prediction of time series data.* It is possible that work done on such models could generalize to more "socially useful" time series prediction problems, like *predicting electric energy consumption, water consumption or food prices.*

We will attempt to first **predict the price of Bitcoin (BTC)** in USD based on historical data of medium frequency (5 min) and low frequency (daily) comprised of the regular price values (open, high, low, close) and trading volume, augmented with extra data derived from the S&P 500 Index (SP500) taken as a rough general marker for the "health of US and West-European economy". (A mini-experiment in adding an extra feature representing "Bitcoin related sentiment of recent news" obtained from using Google Clod Natural Language API for sentiment analysis on a subset of news headlines filtered for Bitcoin related keywords was also carried out, but mainly as a proof-of concept for the technique - the actual historical news data was not enough and not diverse enough, and acquiring and processing better quality news data would have increased the scope of this project too much.)

Secondly, we will try to **predict the price of "Bitcoin-involved companies"** (that have either invested in BTC, or are connected to the BTC-ecosystem by producing mining hardware, or by providing relevant services) by using historical data augmented with **features extracted from recent past historical BTC price.**

The data sets used are publicly available historical data for the prices of stock and Bitcoin, acquired from the NASDAQ website, via the GDAX/CoinbasePro API, and from the Bitstamp (a Bitcoin exchange) website, and spans the 2015 - 2018 time interval.

### Problem statement [2-4 paragraphs]

The goals for this project are:
1. To build a model that predicts BTC-USD @ 5 min. future price change direction N points ahead with an accuracy good enough for a possible profitable trading strategy (based on variance we calculate the minimum accuracy required to make profit for a prediction N * 5 min. into the future - details in Metrics section below).
2. To build a model satisfying similar criteria for BTC-USD @ 1 day.
3. To see if the performance of the model BTC-USD @ 1 day can be improved by incorporating features derived from sentiment analysis or news headlines related to Bitcoin.
4. To build a model that predicts the future price of stocks picked from a set of possibly "Bitcoin involved" companies (see point 5.) based on historical daily stock price data(similar performance criteria as mentioned on 1.).
5. To add features derived from the recent evolution of Bitcoin price to the data used for the prediction of "Bitcoin involved companies" stocks' prices and see whether this improves prediction results.

The models will be LSTS RNNs and this project will attempt to identify a suitable network architecture, hyperparameters values, training data size, number of training epochs, and various data pre-processing strategies. Predictive performance will also be compared with two simpler models: a very basic linear-regression based model, and an ARIMA model.

### Metrics [0.5-1 pages]
Broadly speaking, we seek to minimize error and to increase accuracy.

The two most common error measures for regression problems are **mean absolute error (MAE)**:

$$
\text{MAE} = \frac{1}{N} \sum_{t=1}^N {
  | \hat{y_t} - y_t |
}
$$

and **root-mean-square error (RMSE):**

$$
\text{RMSE} = \sqrt {
  \frac {
    \sum_{t=1}^N {
      ( \hat{y_t} - y_t )^2
    }
  }{N}
}
$$

where:
- $t$ represents the index of a data point in the time-series data, corresponding to a moment in time
- $N$ is the total number of data points
- $y_t$ is true value at time/index $t$ (of predicted variable)
- $\hat{y_t}$ is predicted values at $t$

**RMSE is picked here** because it is also usable a loss function for training the RNN (being differentiable), and it thus it makes slightly more sense (despite some of the theoretical advantages of MAE).

RMSE is expressed in the units of the measures quantity, so we'll also use it normalized by the target value to get a unit-less / percentual value, called **mean absolute percentage error (MAPE)**:

$$
\text{MAPE} = \frac{100\%}{N} \sum_{t=1}^N {
  \frac {
    | \hat{y_t} - y_t |
  }{y_t}
}
$$

This will allow us to express statements like *"predicted values is on average within x% of target value"*. Practically speaking RMSE is still a better choice since we can compute things like "RMSE for constant prediction" (eg. "predict that value stays the same") that will also be used as reference, and which can't be computed as MAPE.

Now, practically speaking, in the simplest case, the predicted value would be fed into a trading strategy, which would decide what action to perform (like "buy" or "sell") depending on the predicted direction of the variation. We want to know the percent of the time this predicted direction is correct, the **direction prediction accuracy (DACC):**

$$
\text{DACC} = \frac{1}{N} \sum_{t=1}^{N}{ P_t }
$$

where:

$$
P_t = \begin{cases}
  1 & {
    \text{, if } (y_t - y_{t-k}) (\hat{y_t} - y_{t-k}) > 0
  }
\\
  0 & \text{, otherwise}
\end{cases}
$$

(Here $k$ represents "how many points ago was the last known value" or "how far into the future we are predicting". Therefore $y_{t-k}$ is $y$ at $k$ points ago, or "y at moment since we started predicting" or "start y".)

Based on variance ($\sigma$), we can figure out, under certain assumptions, what would be the **minimal predication accuracy** when predicting a certain interval into the future (given by $k$), **that can form the basis for a profitable trading strategy**. To note that this is in no way a certitude that such a prediction accuracy would drive a strategy that actually makes profit (only that an accuracy below such threshold would be statistically unlikely to have any change of driving a profitable strategy).

With:
- $\varepsilon$ - as the direction accuracy above 50% (eg. $DACC = 0.5 + \varepsilon$)
- $\text{fee}$ - as the percent transaction fee
- $\text{spread}$ - as the bid-ask spread (the difference between the price at which a unit is bough and the price at which it is sold at a moment in time, aka "how much you'd loose if you bough and instantly sold back a unit")
- $\sigma(k)$ - as the standard deviation for $k$ points

Considering that $100 \cdot (0.5 + \varepsilon)$ percent of the times we predict correctly, and $100 \cdot (0.5 - \varepsilon)$ we predict incorrectly, the amount gained on average for trading one unit ends up being $\sigma(0.5 + \varepsilon) - \sigma(0.5 - \varepsilon) = 2 \varepsilon \sigma$. Adding the trading costs and the condition for the profit to be positive we end up with:

$$
\text{Profit}(k, \varepsilon) = 2 \varepsilon \sigma(k) - \text{fee} - \text{spread} \geq 0
$$

$$
\min \varepsilon_{\text{Profit} \geq 0} (k) = \frac{
  \text{fee} + \text{spread}
}{
  2 \sigma(k)
}
$$

$$
\text{MinProfitableDACC}(k) = \frac{
  \text{fee} + \text{spread}
}{
  2 \sigma(k)
} + 0.5
$$

(In our Python code we use `min_eps_ct` for $(\text{fee}+\text{spread})/2$ since it's constant with respect to almost everything else.)

Since we only care about the sign of profit being positive in this discussion, we can take the traded quantity to be $1$ here omit it from the equations. Now we can use this $MinProfitableDACC$ as a **threshold of minimal change direction prediction accuracy that our models should aim to exceed**.
