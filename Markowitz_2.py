"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, gamma=0.3):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        start_date = self.price.index[0]

        if start_date < pd.Timestamp("2015-01-01"):
            lookback = 200   # for long sample 
        else:
            lookback = 300   # for short sample 
        
        top_k = 4
        # 月初 rebalancing
        month_groups = self.price.index.to_period("M")
        rebalance_dates = self.price.groupby(month_groups).head(1).index

        # 設 gamma
        gamma = self.gamma  

        for date in rebalance_dates:
            idx = self.price.index.get_loc(date)
            if idx < lookback:
                continue

            # lookback returns
            window = self.returns.iloc[idx - lookback: idx]

            asset_window = window[assets]
            cumret = (1 + asset_window).prod() - 1
            winners = cumret.nlargest(top_k).index

            cov = asset_window[winners].cov().values
            mu = asset_window[winners].mean().values
            n = len(winners)

            m = gp.Model()
            m.Params.OutputFlag = 0
            w = m.addMVar(n, lb=0.0)

            risk = w @ cov @ w
            ret = w @ mu

            m.setObjective(gamma * risk - ret, gp.GRB.MINIMIZE)
            m.addConstr(w.sum() == 1.0)
            m.optimize()

            opt_w = w.X

            row = pd.Series(0.0, index=self.price.columns)
            row[winners] = opt_w
            row[self.exclude] = 0
            self.portfolio_weights.loc[date] = row
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
