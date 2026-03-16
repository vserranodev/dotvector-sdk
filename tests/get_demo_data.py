from __future__ import annotations
from pysindy import SINDy, STLSQ, PolynomialLibrary, FiniteDifference
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


INDEX_TICKER = "^GSPC"
DEFAULT_TICKERS = [
    # US Tech & Magnificent 7
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",
    
    # S&P 500 Indices & ETFs
    "^GSPC", "^DJI", "^IXIC", "^RUT", "SPY", "QQQ", "DIA", "IWM",
    
    # Major Banks & Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "SCHW", "BLK",
    
    # Consumer Staples & Retail
    "WMT", "COST", "HD", "LOW", "TGT", "MCD", "SBUX", "NKE", "TJX", "ORLY",
    
    # Healthcare Leaders
    "JNJ", "PFE", "ABBV", "TMO", "DHR", "UNH", "CI", "BMY", "GILD", "AMGN",
    
    # Energy Giants
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "DVN", "MPC", "PSX", "VLO",
    
    # Industrials
    "GE", "CAT", "HON", "UNP", "UPS", "FDX", "LMT", "NOC", "RTX", "BA",
    
    # Communication Services
    "CMCSA", "VZ", "T", "DIS", "TMUS", "CHTR", "FOXA", "DASH", "UBER", "LYV",
    
    # Semiconductors
    "AMD", "QCOM", "TXN", "INTC", "MU", "ADI", "LRCX", "AMAT", "KLAC", "SNPS",
    
    # Consumer Discretionary
    "HD", "LOW", "NKE", "SBUX", "MCD", "BKNG", "MAR", "HLT", "EXPE", "RCL",
    
    # Utilities
    "NEE", "SO", "DUK", "AEP", "EXC", "XEL", "WEC", "ED", "D", "PEG",
    
    # Materials
    "LIN", "SHW", "ECL", "VMC", "MLM", "NTR", "CF", "DOW", "PPG", "APD",
    
    # Real Estate
    "PLD", "AMT", "CCI", "EQIX", "SPG", "O", "DLR", "ARE", "VTR", "WY",
    
    # More S&P 500 Additions
    "CRM", "ADBE", "INTU", "NOW", "ORCL", "ACN", "IBM", "PANW", "SNOW", "PLTR",
    
    # European Stocks
    "ASML", "SAP", "SHEL", "AZN", "NVO", "SNY", "UL", "NSRGY", "NVS", "TM",
    
    # Spanish IBEX Stocks (from search context)
    "ACS", "ACX", "BBVA", "BKT", "ANA", "CABK", "ENG", "NTGY", "GRF", "FER",
    "REE", "ITX", "REP", "IBE", "IDR", "MAP", "TEF", "SCYR", "SAB", "SAN",
    
    # Crypto & Gold (ETFs)
    "GBTC", "MARA", "RIOT", "COIN", "GLD", "SLV", "GDX", "USO", "TLT", "HYG",
    
    # International
    "BABA", "PDD", "BIDU", "TCEHY", "NIO", "BYDDY", "TSM", "SNE", "HMC", "TM",
    
    # Additional Popular
    "PYPL", "SQ", "ROKU", "ZM", "DOCU", "SHOP", "ZS", "NET", "CRWD", "DDOG",
    "OKTA", "MDB", "TEAM", "U", "SE", "MELI", "NU", "DBX", "ZS", "AFRM"
]


def search_tickers(query: str, ticker_list: list[str] | None = None) -> list[str]:
    symbols = ticker_list if ticker_list is not None else DEFAULT_TICKERS
    query = query.strip().upper()
    if not query:
        return symbols
    return [symbol for symbol in symbols if query in symbol.upper()]


def get_sp500_tickers(limit: int | None = None, ticker_list: list[str] | None = None) -> list[str]:
    symbols = ticker_list if ticker_list is not None else DEFAULT_TICKERS
    cleaned = [symbol.replace(".", "-") for symbol in symbols]
    return cleaned[:limit] if limit else cleaned


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1d"] = out.groupby("Ticker")["Close"].pct_change()
    out["log_return_1d"] = np.log(out["Close"]).groupby(out["Ticker"]).diff()
    out["high_low_range"] = (out["High"] - out["Low"]) / out["Close"].replace(0, np.nan)
    out["open_close_gap"] = (out["Close"] - out["Open"]) / out["Open"].replace(0, np.nan)
    out["volatility_20"] = (
        out.groupby("Ticker")["log_return_1d"].rolling(20).std().reset_index(level=0, drop=True)
    )
    out["volume_z20"] = (
        out.groupby("Ticker")["Volume"]
        .transform(lambda s: (s - s.rolling(20).mean()) / s.rolling(20).std())
    )
    return out


def _build_horizontal_ticker_dataset(df: pd.DataFrame) -> pd.DataFrame:
    wide_open = (
        df.pivot(index="date", columns="Ticker", values="Open")
        .sort_index(axis=1)
        .reset_index()
    )
    wide_close = (
        df.pivot(index="date", columns="Ticker", values="Close")
        .sort_index(axis=1)
        .reset_index()
    )

    wide_open = wide_open.rename(
        columns={
            ticker: f"asset_open_{ticker.lower().replace('^', '').replace('-', '_')}"
            for ticker in wide_open.columns
            if ticker != "date"
        }
    )
    wide_close = wide_close.rename(
        columns={
            ticker: f"asset_close_{ticker.lower().replace('^', '').replace('-', '_')}"
            for ticker in wide_close.columns
            if ticker != "date"
        }
    )

    return wide_open.merge(wide_close, on="date", how="inner")


def download_sp500_demo_dataframe(
    start: str = "2015-01-01",
    end: str | None = None,
    limit_tickers: int = 100,
    output_csv: str | Path = "demo_sindy_data.csv",
    ticker_list: list[str] | None = None,
    ticker_query: str | None = None,
) -> pd.DataFrame:
    selected = ticker_list if ticker_list is not None else DEFAULT_TICKERS
    if ticker_query:
        selected = search_tickers(ticker_query, selected)
    tickers = get_sp500_tickers(limit=limit_tickers, ticker_list=selected)
    # Asegura incluir siempre el S&P 500 sin duplicados.
    all_tickers = list(dict.fromkeys(tickers + [INDEX_TICKER]))
    raw = yf.download(
        tickers=all_tickers,
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=True,
        progress=True,
        threads=True,
    )

    frames: list[pd.DataFrame] = []
    expected_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ]

    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            continue
        ticker_df = raw[ticker].copy()
        ticker_df["Ticker"] = ticker
        ticker_df = ticker_df.reset_index().rename(columns={"Date": "date"})
        for col in expected_cols:
            if col not in ticker_df.columns:
                ticker_df[col] = np.nan
        ticker_df = ticker_df[["date", "Ticker"] + expected_cols]
        frames.append(ticker_df)

    if not frames:
        raise RuntimeError("No se pudo descargar datos para los tickers seleccionados.")

    data_long = pd.concat(frames, ignore_index=True)
    data_long = data_long.sort_values(["Ticker", "date"]).reset_index(drop=True)

    data = _build_horizontal_ticker_dataset(data_long)

    if INDEX_TICKER in raw.columns.get_level_values(0):
        index_df = raw[INDEX_TICKER].copy().reset_index().rename(columns={"Date": "date"})
        for col in expected_cols:
            if col not in index_df.columns:
                index_df[col] = np.nan
        index_df = index_df[["date"] + expected_cols]
        index_df = index_df.rename(columns={col: f"sp500_{col.lower().replace(' ', '_')}" for col in expected_cols})
        data = data.merge(index_df, on="date", how="left")

    data = data.sort_values("date").reset_index(drop=True)

    output_csv = Path(output_csv)
    data.to_csv(output_csv, index=False)
    return data


def load_demo_dataframe(csv_path: str | Path = "demo_sindy_data.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path, parse_dates=["date"])


if __name__ == "__main__":
    df = download_sp500_demo_dataframe(limit_tickers=10, output_csv="demo_sindy_data.csv")
    print(df.head())
    print(f"\nFilas: {len(df):,}")
    print("CSV guardado en: demo_sindy_data.csv")