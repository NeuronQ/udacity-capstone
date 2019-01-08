# Trading strategies

## From literature

1. Shah, Devavrat, and Kang Zhang. 2014. “Bayesian Regression and Bitcoin.” In Communication, Control, and Computing (Allerton), 2014 52nd Annual Allerton Conference On, 409–414. IEEE. http://ieeexplore.ieee.org/abstract/document/7028484/.

> Trading Strategy. The trading strategy is very simple: at each time, we either maintain position of +1 Bitcoin, 0 Bitcoin or −1 Bitcoin. At each time instance, we predict the average price movement over the 10 seconds interval, say ∆p, using Bayesian regression (precise details explained below) - if ∆p > t, a threshold, then we buy a bitcoin if current bitcoin position is ≤ 0; if ∆p < −t, then we sell a bitcoin if current position is ≥ 0; else do nothing. The choice of time steps when we make trading decisions as mentioned above are chosen carefully by looking at the recent trends. We skip details as they do not have first order effect on the performance.

2. Honchar, Alexandr. 2017. “Neural Networks for Algorithmic Trading. Correct Time Series Forecasting + Backtesting.” Medium (blog). May 11, 2017. https://medium.com/@alexrachnog/neural-networks-for-algorithmic-trading-1-2-correct-time-series-forecasting-backtesting-9776bfd9e589.

> The strategy I’ve tested is extremely simple: if our network says that price will go up, we buy the stock and sell it only after network says that price will go down and will wait for the next buying signal. The logic looks like:

```python
if np.argmax(pred) == 0 and not self.long_market:
     self.long_market = True
     signal = SignalEvent(1, sym, dt, 'LONG', 1.0)
     self.events.put(signal)
     print pred, 'LONG'
if np.argmax(pred) == 1 and self.long_market:
     self.long_market = False
     signal = SignalEvent(1, sym, dt, 'EXIT', 1.0)
     self.events.put(signal)
     print pred, 'EXIT'
```

## Distilled

1. pseudocode:

```
IF (increase is predicted):
  IF (position is negative / has $ to buy BTC):
    BUY BTC
ELSE IF (decrecrease is predicted):
  IF (position is positive / has BTC to sell):
    SELL BTC
```
