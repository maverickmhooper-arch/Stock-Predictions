import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact_manual

# Set up the UI layout using native IPython widgets
@interact_manual(
    Ticker=widgets.Text(value='NVDA', description='Ticker:'),
    Years=widgets.IntSlider(value=5, min=1, max=10, step=1, description='Years:'),
    Bull_Rev=widgets.FloatText(value=0.15, description='Bull Rev G:'),
    Bull_Net=widgets.FloatText(value=0.25, description='Bull Net G:'),
    Bull_PE=widgets.FloatText(value=30.0, description='Bull P/E:'),
    Base_Rev=widgets.FloatText(value=0.12, description='Base Rev G:'),
    Base_Net=widgets.FloatText(value=0.20, description='Base Net G:'),
    Base_PE=widgets.FloatText(value=25.0, description='Base P/E:'),
    Bear_Rev=widgets.FloatText(value=0.05, description='Bear Rev G:'),
    Bear_Net=widgets.FloatText(value=0.05, description='Bear Net G:'),
    Bear_PE=widgets.FloatText(value=15.0, description='Bear P/E:')
)
def run_valuation(Ticker, Years, Bull_Rev, Bull_Net, Bull_PE,
                  Base_Rev, Base_Net, Base_PE, Bear_Rev, Bear_Net, Bear_PE):

    ticker_symbol = Ticker.strip().upper()
    if not ticker_symbol:
        print("Please enter a valid stock ticker.")
        return

    stock = yf.Ticker(ticker_symbol)

    try:
        current_price = stock.fast_info.get('last_price') or stock.info.get('currentPrice')
        shares_out = stock.info.get('sharesOutstanding')

        quarterly_financials = stock.quarterly_financials
        if quarterly_financials.empty or len(quarterly_financials.columns) < 4:
            raise ValueError("Not enough quarterly financial data available.")

        quarterly_financials = quarterly_financials.sort_index(axis=1, ascending=False)
        total_rev = quarterly_financials.loc['Total Revenue'].head(4).sum()
        total_net = quarterly_financials.loc['Net Income'].head(4).sum()

    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    time_steps = np.arange(0, Years + 1)
    scenarios = {
        'Bull': {'color': 'green', 'rev_g': Bull_Rev, 'net_g': Bull_Net, 'pe': Bull_PE},
        'Base': {'color': 'blue', 'rev_g': Base_Rev, 'net_g': Base_Net, 'pe': Base_PE},
        'Bear': {'color': 'red', 'rev_g': Bear_Rev, 'net_g': Bear_Net, 'pe': Bear_PE}
    }

    print(f"\n--- {ticker_symbol} | Current Price: ${current_price:.2f} ---")
    plt.figure(figsize=(10, 6))

    for case, config in scenarios.items():
        rev_path = total_rev * (1 + config['rev_g'])**time_steps
        eps_path = (total_net * (1 + config['net_g']) ** time_steps) / shares_out
        price_path = eps_path * config['pe']

        final_price = price_path[-1]
        cagr = (final_price / current_price)**(1/Years) - 1

        plt.plot(time_steps, price_path, label=f"{case} (Final: ${final_price:.2f}, CAGR: {cagr:.1%})", color=config['color'], marker='o')

    plt.axhline(y=current_price, color='black', linestyle='--', alpha=0.5, label='Current Price')
    plt.title(f"{ticker_symbol} {Years}-Year Projection")
    plt.xlabel("Years Out"); plt.ylabel("Stock Price ($)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.show()
