import numpy as np

from manifoldpy import api

MARKET_ID = "5c8i9riCl5o8S1N0m8qx"
market = api.get_market(MARKET_ID)
t0 = market.createdTime
tclose = market.closeTime

bets = api.get_all_bets(marketId=MARKET_ID)[::-1]
t = np.array([t0] + [bet.createdTime for bet in bets] + [bets[-1].createdTime])
# t = np.array([t0] + [bet.createdTime for bet in bets] + [tclose])
p = 100 * np.array([0.5] + [bet.probAfter for bet in bets])

# rounding:
within_two_of_edges = (p < 2) | (p > 98)
p[within_two_of_edges] = np.round(p[within_two_of_edges], 1)
p[~within_two_of_edges] = np.round(p[~within_two_of_edges], 0)

avg = (p * np.diff(t)).sum() / (t[-1] - t[0])

print(f"Market average is: {avg:.3f}")