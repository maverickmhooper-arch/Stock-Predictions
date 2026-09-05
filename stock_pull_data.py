import pandas as pd
import requests
import numpy as np
import warnings
import json

# Silences generic pandas warnings to keep console output pristine
warnings.simplefilter(action='ignore', category=FutureWarning)

# -------------------------------------------------------------------------
# CONFIGURATION & HEADERS (SEC COMPLIANT)
# -------------------------------------------------------------------------
USER_AGENT = "Maverick Hooper maverick.m.hooper@gmail.com"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate"
}


def get_cik_from_ticker(ticker: str) -> str:
    """Fetches official SEC ticker mapping file and retrieves the 10-digit CIK."""
    ticker = ticker.upper().strip()
    url = "https://www.sec.gov/files/company_tickers.json"

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            raise Exception(f"SEC server returned status code {response.status_code}")
        ticker_data = response.json()
    except Exception:
        url_alt = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url_alt, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            raise Exception("SEC denied connection. Check network state or User-Agent rules.")
        ticker_data = response.json()

    for entry in ticker_data.values():
        if entry["ticker"] == ticker:
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker '{ticker}' not found in SEC database.")


def get_company_facts(cik: str) -> dict:
    """Fetches the comprehensive JSON payload of XBRL facts for a given CIK."""
    base_url = "https://data.sec.gov/api/xbrl/companyfacts/"
    filename = f"CIK{cik}.json"
    full_url = base_url + filename

    response = requests.get(full_url, headers=HEADERS, timeout=15)

    if response.status_code == 200:
        try:
            return response.json()
        except json.JSONDecodeError:
            raise Exception("Failed parsing corporate XBRL payload. Returned text was invalid JSON.")
    else:
        raise Exception(f"Failed to fetch SEC facts for CIK {cik}. Status code: {response.status_code}")


def extract_pure_quarterly_metric(facts_json: dict, standard_tags: list, normalized_name: str) -> pd.DataFrame:
    """Extracts, isolates, and mathematically derives pure 3-month financial values."""
    data_points = None

    for tag in standard_tags:
        try:
            data_points = facts_json["facts"]["us-gaap"][tag]["units"]["USD"]
            break
        except KeyError:
            continue

    if not data_points:
        return pd.DataFrame()

    df = pd.DataFrame(data_points)
    df = df[df["form"].isin(["10-Q", "10-K"])].copy()
    if df.empty:
        return pd.DataFrame()

    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["days"] = (df["end"] - df["start"]).dt.days

    # Deduplicate restated overlapping rows, preserving the newest data
    df = df.sort_values(by=["end", "filed"]).drop_duplicates(subset=["start", "end"], keep="last")

    # Isolate standalone 3-month entries (~1 quarter)
    df_3m = df[(df["days"] >= 80) & (df["days"] <= 105)].copy()

    # Isolate compound YTD entries (6-month, 9-month, and 12-month flows)
    df_ytd = df[(df["days"] > 105)].copy()

    calculated_quarters = []
    for _, ytd_row in df_ytd.iterrows():
        # FIXED: Match rows sharing the exact same start date that are ~1 quarter shorter
        match = df[
            (df["start"] == ytd_row["start"]) &
            (df["days"] >= ytd_row["days"] - 105) &
            (df["days"] <= ytd_row["days"] - 80)
        ]
        if not match.empty:
            match_row = match.sort_values(by="filed").iloc[-1]
            pure_value = ytd_row["val"] - match_row["val"]

            # Reconstruct proper quarter tags
            fp = "Q4" if (ytd_row["form"] == "10-K" or ytd_row["days"] > 300) else ytd_row["fp"]

            calculated_quarters.append({
                "fy": ytd_row["fy"],
                "fp": fp,
                "end": ytd_row["end"],
                "val": pure_value
            })

    df_calc = pd.DataFrame(calculated_quarters)
    df_clean_3m = df_3m[["fy", "fp", "end", "val"]].copy()

    if not df_calc.empty:
        df_clean_3m = pd.concat([df_clean_3m, df_calc], ignore_index=True)

    df_clean_3m["end"] = pd.to_datetime(df_clean_3m["end"])
    df_clean_3m = df_clean_3m.sort_values(by="end").drop_duplicates(subset=["fy", "fp"], keep="last")

    df_final = df_clean_3m.rename(columns={"fy": "year", "fp": "quarter", "end": "period_end", "val": normalized_name})
    return df_final[["year", "quarter", "period_end", normalized_name]]

def build_advanced_financial_model(ticker: str) -> pd.DataFrame:
    """Pulls corporate statements and builds full scale financial analysis metrics."""
    cik = get_cik_from_ticker(ticker)
    facts = get_company_facts(cik)

    # XBRL Tag Matrices — Configured to map across mixed retail/tech architectures
    tags_revenue = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet", "OperatingRevenueRevenue"]
    tags_cogs = ["CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfRevenue", "CostOfGoodsSoldDepreciationAndAmortization"]
    tags_gross_profit = ["GrossProfit", "GrossMargin"]
    tags_opex = ["OperatingExpenses", "OperatingCostsAndExpenses"]
    tags_op_income = ["OperatingIncomeLoss", "OperatingProfit"]
    tags_net_income = ["NetIncomeLoss", "NetIncome"]

    # Extract foundational dataframes (Using corrected Part 1 extraction logic)
    df_rev = extract_pure_quarterly_metric(facts, tags_revenue, "Revenue")
    df_cogs = extract_pure_quarterly_metric(facts, tags_cogs, "COGS")
    df_gp = extract_pure_quarterly_metric(facts, tags_gross_profit, "Gross_Profit")
    df_opex = extract_pure_quarterly_metric(facts, tags_opex, "Op_Expenses")
    df_op = extract_pure_quarterly_metric(facts, tags_op_income, "Operating_Income")
    df_net = extract_pure_quarterly_metric(facts, tags_net_income, "Net_Income")

    # Sequence merge metrics on distinct year and quarter signatures
    final_df = df_rev
    for next_df in [df_cogs, df_gp, df_opex, df_op, df_net]:
        if not next_df.empty:
            final_df = pd.merge(final_df, next_df.drop(columns=["period_end"], errors="ignore"), on=["year", "quarter"], how="outer")

    if final_df.empty:
        raise ValueError(f"No usable financial records found for ticker: {ticker.upper()}")

    # Format dates and chronologically sort the dataframe rows
    final_df = final_df.sort_values(by=["year", "quarter"]).reset_index(drop=True)
    final_df["year"] = final_df["year"].fillna(0).astype(int).astype(str)

    # Structural math validation checks to fill corporate taxonomy gaps
    if "Revenue" in final_df.columns and "COGS" in final_df.columns and "Gross_Profit" not in final_df.columns:
        final_df["Gross_Profit"] = final_df["Revenue"] - final_df["COGS"]
    if "Revenue" in final_df.columns and "Gross_Profit" in final_df.columns and "COGS" not in final_df.columns:
        final_df["COGS"] = final_df["Revenue"] - final_df["Gross_Profit"]
    if "Gross_Profit" in final_df.columns and "Operating_Income" in final_df.columns and "Op_Expenses" not in final_df.columns:
        final_df["Op_Expenses"] = final_df["Gross_Profit"] - final_df["Operating_Income"]
    if "Gross_Profit" in final_df.columns and "Op_Expenses" in final_df.columns and "Operating_Income" not in final_df.columns:
        final_df["Operating_Income"] = final_df["Gross_Profit"] - final_df["Op_Expenses"]

    core_metrics = ["Revenue", "COGS", "Gross_Profit", "Op_Expenses", "Operating_Income", "Net_Income"]
    existing_metrics = [m for m in core_metrics if m in final_df.columns]

    # Calculate standard consecutive Trailing Twelve Month sums (window=4)
    for col in existing_metrics:
        final_df[f"{col}_TTM"] = final_df[col].rolling(window=4).sum()

    # --- ADVANCED CALCULATIONS FRAMEWORK (QUARTERLY & TTM VARIATIONS) ---
    for suffix in ["", "_TTM"]:
        rev = final_df[f"Revenue{suffix}"] if f"Revenue{suffix}" in final_df.columns else np.nan
        gp = final_df[f"Gross_Profit{suffix}"] if f"Gross_Profit{suffix}" in final_df.columns else np.nan
        op = final_df[f"Operating_Income{suffix}"] if f"Operating_Income{suffix}" in final_df.columns else np.nan
        net = final_df[f"Net_Income{suffix}"] if f"Net_Income{suffix}" in final_df.columns else np.nan

        # Margins
        final_df[f"Gross_Margin{suffix}"] = gp / rev
        final_df[f"Operating_Margin{suffix}"] = op / rev
        final_df[f"Net_Margin{suffix}"] = net / rev

        # Growth metrics (Year-over-Year shifts)
        shift_periods = 4
        final_df[f"Revenue_Growth_YoY{suffix}"] = final_df[f"Revenue{suffix}"].pct_change(shift_periods, fill_method=None)
        final_df[f"Operating_Income_Growth_YoY{suffix}"] = final_df[f"Operating_Income{suffix}"].pct_change(shift_periods, fill_method=None)
        final_df[f"Net_Income_Growth_YoY{suffix}"] = final_df[f"Net_Income{suffix}"].pct_change(shift_periods, fill_method=None)

        # Incremental Margin Changes (dProfit / dRevenue)
        delta_rev = final_df[f"Revenue{suffix}"].diff(1)
        final_df[f"Incremental_Gross_Margin{suffix}"] = final_df[f"Gross_Profit{suffix}"].diff(1) / delta_rev
        final_df[f"Incremental_Operating_Margin{suffix}"] = final_df[f"Operating_Income{suffix}"].diff(1) / delta_rev

    return final_df


def output_formatted_summary(df: pd.DataFrame, ticker: str):
    """Formats and prints the terminal report into highly readable, clean metrics summaries."""
    print(f"\n🚀 Analysis Pipeline Complete for {ticker.upper()}:")

    display_cols = [
        "year", "quarter", "Revenue_TTM", "Revenue_Growth_YoY_TTM",
        "Gross_Margin_TTM", "Operating_Margin_TTM", "Net_Margin_TTM",
        "Incremental_Operating_Margin_TTM"
    ]

    valid_cols = [c for c in display_cols if c in df.columns]

    # Isolate calculated blocks to strip blank rolling buffer zones
    if "Revenue_TTM" in df.columns:
        sub_df = df[valid_cols].dropna(subset=["Revenue_TTM"]).copy()
    else:
        sub_df = df[valid_cols].copy()

    if sub_df.empty:
        print("No completed calculations available to display.")
        return

    formatted_df = pd.DataFrame()
    formatted_df["Year"] = sub_df["year"]
    formatted_df["Quarter"] = sub_df["quarter"]

    if "Revenue_TTM" in sub_df.columns:
        formatted_df["Revenue TTM"] = sub_df["Revenue_TTM"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    if "Revenue_Growth_YoY_TTM" in sub_df.columns:
        formatted_df["Revenue YoY Growth"] = sub_df["Revenue_Growth_YoY_TTM"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    if "Gross_Margin_TTM" in sub_df.columns:
        formatted_df["Gross Margin"] = sub_df["Gross_Margin_TTM"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    if "Operating_Margin_TTM" in sub_df.columns:
        formatted_df["Operating Margin"] = sub_df["Operating_Margin_TTM"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    if "Net_Margin_TTM" in sub_df.columns:
        formatted_df["Net Margin"] = sub_df["Net_Margin_TTM"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    if "Incremental_Operating_Margin_TTM" in sub_df.columns:
        formatted_df["Incremental Op Margin"] = sub_df["Incremental_Operating_Margin_TTM"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")

    print(formatted_df.tail(12).to_string(index=False))


if __name__ == "__main__":
    target_ticker = input("Enter Stock Ticker Symbol (e.g., AMZN, NVDA, MSFT): ")
    run = False
    try:
        while True:
            df_model = build_advanced_financial_model(target_ticker)
            output_formatted_summary(df_model, target_ticker)
            run = input("Continue? Click ENTER. ")
            if run:
              break
            target_ticker = input("Enter Stock Ticker Symbol (e.g., AMZN, NVDA, MSFT): ")
    except Exception as e:
        print(f"\n❌ Error executing pipeline: {e}")

