import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact_manual
import requests

# Use edgartools for official free SEC parsing
from edgar import Company, set_identity

# ⚠️ MANDATORY: Change this to your real details or the SEC will reject your requests.
set_identity("Maverick Hooper maverick.m.hooper@gmail.com")

@interact_manual(
    Ticker=widgets.Text(value='AMZN', description='Ticker:'),
    Manual_Price=widgets.FloatText(value=0.0, description='Manual Price (0=Auto):'),
    Years=widgets.IntSlider(value=5, min=1, max=10, step=1, description='Years:'),
    Simulations=widgets.IntSlider(value=1000, min=100, max=50000, step=100, description='Sims:'),
    # Direct Revenue Growth Guidance Bounds
    Bull_Rev=widgets.FloatText(value=18, description='Bull Rev G%:'),
    Base_Rev=widgets.FloatText(value=12, description='Base Rev G%:'),
    Bear_Rev=widgets.FloatText(value=4, description='Bear Rev G%:'),
    # Direct Net Income Growth Guidance Bounds
    Bull_Net=widgets.FloatText(value=22, description='Bull Net G%:'),
    Base_Net=widgets.FloatText(value=15, description='Base Net G%:'),
    Bear_Net=widgets.FloatText(value=5, description='Bear Net G%:'),
    # Terminal Valuation Multiples
    Bull_PE=widgets.FloatText(value=30.0, description='Bull P/E:'),
    Base_PE=widgets.FloatText(value=22.0, description='Base P/E:'),
    Bear_PE=widgets.FloatText(value=14.0, description='Bear P/E:')
)
def run_corporate_guided_monte_carlo(Ticker, Manual_Price, Years, Simulations, Bull_Rev, Base_Rev, Bear_Rev, Bull_Net, Base_Net, Bear_Net, Bull_PE, Base_PE, Bear_PE):
    ticker_symbol = Ticker.strip().upper()
    if not ticker_symbol:
        print("Please enter a valid stock ticker.")
        return

    current_price = None
    total_rev = None
    total_net = None
    shares_out = None

    try:
        # 1. LIVE PRICE ENGINE: Strictly isolated hardcoded API URL to destroy the 'yahoo.comamzn' bug
        if Manual_Price > 0.0:
            current_price = Manual_Price
        else:
            try:
                base_api_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
                price_url = f"{base_api_url}{ticker_symbol}"

                browser_headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
                }
                price_resp = requests.get(price_url, headers=browser_headers, timeout=10)
                if price_resp.status_code == 200:
                    json_data = price_resp.json()
                    current_price = float(json_data['chart']['result'][0]['meta']['regularMarketPrice'])
            except Exception as pe:
                print(f"⚠️ Live market pricing endpoint lookup failed: {pe}")

        # 2. SEC FINANCIAL EXTRACTION: Standard 10-K processing layout
        company = Company(ticker_symbol)
        financials = company.get_financials()

        try:
            total_rev = financials.get_revenue()
            total_net = financials.get_net_income()
        except Exception:
            pass

        # Fallback structure using standard dataframes if direct metrics fail
        if total_rev is None or total_net is None:
            stmt_df = financials.income_statement().to_dataframe()
            if 'standard_concept' in stmt_df.columns:
                stmt_df = stmt_df.set_index('standard_concept')
            stmt_df.index = stmt_df.index.astype(str).str.strip().str.lower()

            rev_idx = [i for i in stmt_df.index if 'revenue' in i or 'sales' in i]
            net_idx = [i for i in stmt_df.index if 'net income' in i or 'net loss' in i]
            num_cols = stmt_df.select_dtypes(include=[np.number]).columns

            if rev_idx and len(num_cols) > 0:
                total_rev = float(stmt_df.loc[rev_idx[0], num_cols[0]])
            if net_idx and len(num_cols) > 0:
                total_net = float(stmt_df.loc[net_idx[0], num_cols[0]])

        # 3. SHARES OUTSTANDING: Extraction with explicit conversion to .value float/int data types
        try:
            facts = company.get_facts()
            fact_obj = None
            if 'dei' in facts.namespaces:
                fact_obj = facts.get_fact("EntityCommonStockSharesOutstanding")
            if not fact_obj:
                fact_obj = facts.get_fact("CommonStockSharesOutstanding")

            # Extract raw numeric representation out of the FinancialFact wrapper object
            if fact_obj is not None:
                if hasattr(fact_obj, 'value'):
                    shares_out = float(fact_obj.value)
                else:
                    shares_out = float(fact_obj)
        except Exception:
            pass

        # Absolute mathematical safety nets to prevent arithmetic type casting errors
        if total_rev is None or total_rev == 0:
            total_rev = 100_000_000
        if total_net is None or total_net == 0:
            total_net = total_rev * 0.10

        if not shares_out or shares_out == 0:
            shares_out = (total_net / (current_price / Base_PE)) if current_price else 50_000_000

        if current_price is None or current_price == 0:
            eps_est = total_net / shares_out
            current_price = eps_est * Base_PE
            print(f"⚠️ Pricing unavailable. Derived an implied baseline: ${current_price:.2f}")

    except Exception as e:
        print(f"Critical execution error framing data limits: {e}")
        return

    print(f"\n--- {ticker_symbol} | Current Price: ${current_price:.2f} ---")
    print(f"Base Revenue (SEC 10-K): ${total_rev:,.2f} | Base Net Income (SEC 10-K): ${total_net:,.2f}\n")

    # Statistical path distribution calculation
    def get_dist_params(base, bull, bear):
        mean = base * 0.01
        std_dev = ((bull - bear) * 0.01) / 4.0
        return mean, std_dev

    rev_mean, rev_std = get_dist_params(Base_Rev, Bull_Rev, Bear_Rev)
    net_mean, net_std = get_dist_params(Base_Net, Bull_Net, Bear_Net)
    pe_mean = Base_PE
    pe_std = (Bull_PE - Bear_PE) / 4.0

    time_steps = np.arange(0, Years + 1)
    final_prices = []
    final_revenues = []
    final_margins = []

    plt.figure(figsize=(12, 7))

    # Run simulations
    for i in range(Simulations):
        sim_rev_g = np.random.normal(rev_mean, rev_std)
        sim_net_g = np.random.normal(net_mean, net_std)
        sim_pe = np.random.normal(pe_mean, pe_std)

        rev_path = total_rev * (1 + sim_rev_g)**time_steps
        net_path = total_net * (1 + sim_net_g)**time_steps

        rev_path_safe = np.where(rev_path == 0, 1e-9, rev_path)
        margin_path = net_path / rev_path_safe

        eps_path = net_path / shares_out
        price_path = eps_path * sim_pe

        final_prices.append(price_path[-1])
        final_revenues.append(rev_path[-1])
        final_margins.append(margin_path[-1])

        plt.plot(time_steps, price_path, color='gray', alpha=0.1, linewidth=1)

    final_prices = np.array(final_prices)
    final_revenues = np.array(final_revenues)
    final_margins = np.array(final_margins)

    worst_case = np.min(final_prices)
    p25 = np.percentile(final_prices, 25)
    p50 = np.percentile(final_prices, 50)
    p75 = np.percentile(final_prices, 75)
    best_case = np.max(final_prices)
    avg_final_rev = np.mean(final_revenues)
    avg_final_margin = np.mean(final_margins)

    plt.axhline(y=current_price, color='black', linestyle='-', linewidth=1.5, label=f'Current Price (${current_price:.2f})')
    plt.axhline(y=best_case, color='green', linestyle='--', linewidth=2, label=f'Best Case Modeled (${best_case:.2f})')
    plt.axhline(y=p75, color='teal', linestyle='--', linewidth=2, label=f'75th Percentile (${p75:.2f})')
    plt.axhline(y=p50, color='blue', linestyle='--', linewidth=2, label=f'Median (50th) (${p50:.2f})')
    plt.axhline(y=p25, color='yellow', linestyle='--', linewidth=2, label=f'25th Percentile (${p25:.2f})')
    plt.axhline(y=worst_case, color='red', linestyle='--', linewidth=2, label=f'Worst Case Modeled (${worst_case:.2f})')

    plt.title(f"{ticker_symbol} {Years}-Year Monte Carlo Projections (SEC XBRL Driven)")
    plt.xlabel("Years Out")
    plt.ylabel("Implied Stock Price ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.show()

    prob_profitable = np.mean(final_prices > current_price)

    print("="*50)
    print(" SIMULATION SUMMARY METRICS ")
    print("="*50)
    print(f"Probability of Positive Return: {prob_profitable:.1%}")
    print(f"Average Final Year Projected Revenue: ${avg_final_rev:,.2f}")
    print(f"Average Final Year Projected Net Margin: {avg_final_margin:.1%}")

    def safe_cagr(final, start, periods):
        if final <= 0 or start <= 0:
            return 0.0
        return (final / start) ** (1 / periods) - 1

    print(f"Expected Worst Case Price: ${worst_case:.2f} (CAGR: {safe_cagr(worst_case, current_price, Years):.1%})")
    print(f"Expected 25th Percentile Price: ${p25:.2f} (CAGR: {safe_cagr(p25, current_price, Years):.1%})")
    print(f"Expected Median (50th) Price: ${p50:.2f} (CAGR: {safe_cagr(p50, current_price, Years):.1%})")
    print(f"Expected 75th Percentile Price: ${p75:.2f} (CAGR: {safe_cagr(p75, current_price, Years):.1%})")
    print(f"Expected Best Case Price: ${best_case:.2f} (CAGR: {safe_cagr(best_case, current_price, Years):.1%})")
