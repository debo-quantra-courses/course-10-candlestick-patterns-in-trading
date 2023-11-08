# For data manipulation
import numpy as np
import pandas as pd

# For data visulaisation
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

# Datetime manipulation
import datetime


# Function to calculate SL and TP
def exit_values(data, index, entry_price, r, n=1 ):
    if data['entry_signal'].iloc[index-1] == 1:
        SL = data["Low"].iloc[index - n:index].min()
        TP = entry_price + r * (entry_price - SL)

    elif data['entry_signal'].iloc[index-1] == -1:
        SL = data["High"].iloc[index - n:index].max()
        TP = entry_price + r * (entry_price - SL)

    return SL, TP


def backtesting(data, direction, rr):
    # Create dataframe for storing trades
    trades = pd.DataFrame(columns=['Entry_Date', 'Entry_Price', 'Exit_Date', 'Exit_Type', 'Exit_Price'])

    # Initialize current position, number of trades, cumulative pnl to 0
    current_position = 0
    trade_num = 0
    cum_pnl = 0

    # Set exit flag to False
    exit_flag = False

    for index in range(len(data)):
        # Check if entry_signal is 1 and there is no open position (Step 3)
        if (data['entry_signal'].iloc[index - 1] == 1 or data['entry_signal'].iloc[index - 1] == -1) and current_position == 0:

            # Set entry date and entry price (Step 3.1)
            entry_date = data.iloc[index].name
            entry_price = data['Open'].iloc[index]

            # Compute SL and TP for the trade (Step 3.2)
            stoploss, take_profit = exit_values(data, index, entry_price, rr)

            # Update current position to 1 (Step 3.3)
            current_position = data['entry_signal'].iloc[index - 1]

            # Increase number of trades by 1
            trade_num += 1

            # Print trade details
            print("-" * 30)
            print(f"Trade No: {trade_num} | Entry Date: {entry_date} | Entry Price: {entry_price}")

            # Check if there is an open position of the given timestamp (Step 4)
        elif current_position == 1 or current_position == -1:

            # Exit the trade if any of the exit condition is met (Step 4.1)

            if current_position == 1:
                if data['Close'].iloc[index] < stoploss:
                    exit_type = 'SL'
                    exit_flag = True

                elif data['Close'].iloc[index] > take_profit:
                    exit_type = 'TP'
                    exit_flag = True

            elif current_position == -1:
                if data['Close'].iloc[index] > stoploss:
                    exit_type = 'SL'
                    exit_flag = True

                elif data['Close'].iloc[index] < take_profit:
                    exit_type = 'TP'
                    exit_flag = True

                # Check if exit flag is true (Step 4.2)
            if exit_flag:
                # Set exit date and exit price (Step 4.2.1)
                exit_date = data.iloc[index].name
                exit_price = data['Close'].iloc[index]

                # Calculate pnl for the trade (Step 4.2.2)
                trade_pnl = current_position * round(exit_price - entry_price, 2)

                # Calculate cumulative pnl
                cum_pnl = round(cum_pnl, 2) + trade_pnl

                # Append the trade to trades dataframe (Step 4.2.3)
                trades = trades.append({'Entry_Date': entry_date, 'Entry_Price': entry_price,
                                        'Exit_Date': exit_date, 'Exit_Type': exit_type,
                                        'Exit_Price': exit_price, 'PnL': trade_pnl}, ignore_index=True)

                # Update current position to 0 and set exit flag to False (Step 4.2.4)
                current_position = 0
                exit_flag = False

                # Print trade details
                print(
                    f"Trade No: {trade_num} | Exit Type: {exit_type} | Date: {exit_date} | Exit Price: {exit_price} | PnL: {trade_pnl} | Cum PnL: {round(cum_pnl, 2)}")
    if direction == "long":
        # Append signals as 1 and 0 on entry and exit dates
        data.loc[data.index.isin(trades.Entry_Date), 'trade_signal'] = 1
        data.loc[data.index.isin(trades.Exit_Date), 'trade_signal'] = 0

        # Forward fill the NaN values
        data['trade_signal'] = data['trade_signal'].ffill(axis=0)

        # Set the remaining NaN values to 0
        data['trade_signal'] = data['trade_signal'].fillna(0)

    elif direction == "short":
        # Append signals as 1 and 0 on entry and exit dates
        data.loc[data.index.isin(trades.Entry_Date), 'trade_signal'] = -1
        data.loc[data.index.isin(trades.Exit_Date), 'trade_signal'] = 0

        # Forward fill the NaN values
        data['trade_signal'] = data['trade_signal'].ffill(axis=0)

        # Set the remaining NaN values to 0
        data['trade_signal'] = data['trade_signal'].fillna(0)

    return data, trades


def trade_level_analytics(trades):
    # Create dataframe to store trade analytics
    analytics = pd.DataFrame(index=['Strategy'])

    # Calculate total PnL
    analytics['Total PnL'] = trades.PnL.sum()

    # Print the value
    print("Total PnL: ", analytics['Total PnL'][0])

    # Number of total trades
    analytics['total_trades'] = len(trades)

    # Profitable trades
    analytics['Number of Winners'] = len(trades.loc[trades.PnL > 0])

    # Loss-making trades
    analytics['Number of Losers'] = len(trades.loc[trades.PnL <= 0])

    # Win percentage
    analytics['Win (%)'] = 100 * analytics['Number of Winners'] / analytics.total_trades

    # Loss percentage
    analytics['Loss (%)'] = 100 * analytics['Number of Losers'] / analytics.total_trades

    # Per trade profit/loss of winning trades
    analytics['per_trade_PnL_winners'] = trades.loc[trades.PnL > 0].PnL.mean()

    # Per trade profit/loss of losing trades
    analytics['per_trade_PnL_losers'] = np.abs(trades.loc[trades.PnL <= 0].PnL.mean())

    # Convert entry time and exit time to datetime format
    trades['Entry Date'] = pd.to_datetime(trades['Entry_Date'])
    trades['Exit Date'] = pd.to_datetime(trades['Exit_Date'])

    # Calculate holding period for each trade
    holding_period = trades['Exit Date'] - trades['Entry Date']

    # Calculate their mean
    analytics['Average holding time'] = holding_period.mean()

    # Calculate profit factor
    analytics['Profit Factor'] = (analytics['Win (%)'] / 100 * analytics['per_trade_PnL_winners']) / (
            analytics['Loss (%)'] / 100 * analytics['per_trade_PnL_losers'])

    return analytics


def performance_metrics(data, direction, t=1500):
    if direction == "long":

        # Plot the close prices of the stock
        close_plot = data.Close[-t:].plot(figsize=(15, 7), color='blue')

        # Plot the signal
        signal_plot = data.trade_signal[-t:].plot(figsize=(15, 7), secondary_y=True, ax=close_plot, style='green')

        # Highlight the holding periods of the long positions
        plt.fill_between(data.Close[-t:].index, 0, 1, where=(data.trade_signal[-t:] > 0), color='green', alpha=0.1, lw=0)

        # Set title
        plt.title('Buy Signals for a Long-only Strategy', fontsize=14)

        # Plot xlabel
        close_plot.set_xlabel('Date', fontsize=12)

        # Plot ylabels
        close_plot.set_ylabel('Price ($)', fontsize=12)
        signal_plot.set_ylabel('Signal', fontsize=12)

        # Legend of the plot
        plt.legend(["Signal", "Long"], loc="upper left")

        # Set title
        plt.title('Buy Signals for a Long-only Strategy', fontsize=14)

        # Display the graph
        plt.show()

    if direction == "short":

        # Plot the close prices of the stock
        close_plot = data.Close[-t:].plot(figsize=(15, 7), color='blue')

        # Plot the signal
        signal_plot = data.trade_signal[-t:].plot(figsize=(15, 7), secondary_y=True, ax=close_plot, style='red')

        # Highlight the holding periods of the long positions
        plt.fill_between(data.Close[-t:].index, 0, 1, where=(data.trade_signal[-t:] > 0), color='green', alpha=0.1,
                         lw=0)

        # Set title
        plt.title('Sell Signals for a Short-only Strategy', fontsize=14)

        # Plot xlabel
        close_plot.set_xlabel('Date', fontsize=12)

        # Plot ylabels
        close_plot.set_ylabel('Price ($)', fontsize=12)
        signal_plot.set_ylabel('Signal', fontsize=12)

        # Legend of the plot
        plt.legend(["Signal", "Short"], loc="upper left")

        # Set title
        plt.title('Sell Signals for a Short-only Strategy', fontsize=14)

        # Display the graph
        plt.show()


    data['strategy_returns'] = data.trade_signal.shift(1) * data.Close.pct_change()

    # Plot daily returns
    data.strategy_returns.plot(figsize=(15, 7), color='green')
    plt.title('Return of One Data Point', fontsize=14)
    plt.ylabel('Returns', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.show()

    # Plot cumulative returns
    (data.strategy_returns + 1).cumprod().plot(figsize=(15, 7), color='black')
    plt.title('Cumulative Returns', fontsize=14)
    plt.ylabel('Returns (in times)', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.show()

    # Create a data to store performance metrics
    performance_metrics = pd.DataFrame(index=['Strategy'])

    # Number of trading candles per day
    n = data.loc[datetime.datetime.strftime(data.index[-1].date(), '%Y-%m-%d')].shape[0]
    # Set a risk-free rate
    risk_free_rate = 0.02 / (252 * n)
    # Calculate Sharpe ratio
    performance_metrics['Sharpe Ratio'] = np.sqrt(252 * n) * (np.mean(data.strategy_returns) -
                                                              (risk_free_rate)) / np.std(data.strategy_returns)

    # Compute the cumulative maximum
    data['Peak'] = (data['strategy_returns'] + 1).cumprod().cummax()
    # Compute the Drawdown
    data['Drawdown'] = (((data['strategy_returns'] + 1).cumprod() - data['Peak']) / data['Peak'])
    # Compute the maximum drawdown
    performance_metrics['Maximum Drawdown'] = "{0:.2f}%".format((data['Drawdown'].min()) * 100)

    # Plot maximum drawdown
    data['Drawdown'].plot(figsize=(15, 7), color='red')
    # Set the title and axis labels
    plt.title('Drawdowns', fontsize=14)
    plt.ylabel('Drawdown(%)', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.fill_between(data['Drawdown'].index, data['Drawdown'].values, color='red')
    plt.show()

    return performance_metrics
