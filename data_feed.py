"""
BTC 5-min OHLCV data from Coinbase public API (no account needed, US-friendly).
Also includes a Polymarket helper to list active BTC markets.
"""

import time
import requests
import pandas as pd
from config import SYMBOL, INTERVAL, NUM_CANDLES, POLYMARKET_API


# ─── Coinbase ─────────────────────────────────────────────────────────────────

def fetch_coinbase_candles(symbol: str = SYMBOL,
                           granularity: int = INTERVAL,
                           total_candles: int = NUM_CANDLES) -> pd.DataFrame:
    """
    Fetch historical 5-min OHLCV from Coinbase Advanced Trade public API.
    Coinbase returns max 300 candles per request — we loop to get more.
    """
    url      = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    all_data = []
    end_time = int(time.time())
    max_per_req = 300

    print(f"Fetching {total_candles} candles for {symbol} ({granularity}s interval)...")

    while len(all_data) < total_candles:
        start_time = end_time - (max_per_req * granularity)
        params = {
            "granularity": granularity,
            "start": start_time,
            "end": end_time,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break

        all_data.extend(batch)
        end_time = start_time
        time.sleep(0.25)   # respect rate limits

    df = pd.DataFrame(all_data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df = df.tail(total_candles).reset_index(drop=True)

    print(f"  Fetched {len(df)} candles  |  {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    return df


# ─── Polymarket ───────────────────────────────────────────────────────────────

def fetch_btc_polymarkets() -> pd.DataFrame:
    """
    Fetch active Polymarket markets related to Bitcoin.
    Returns a DataFrame with market slug, question, yes/no prices.
    """
    url    = f"{POLYMARKET_API}/markets"
    params = {"active": "true", "closed": "false", "limit": 100}

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    markets = data.get("data", [])
    btc_markets = [m for m in markets if "bitcoin" in m.get("question", "").lower()
                   or "btc" in m.get("question", "").lower()]

    if not btc_markets:
        print("No active BTC markets found on Polymarket.")
        return pd.DataFrame()

    rows = []
    for m in btc_markets:
        tokens = m.get("tokens", [])
        yes_price = next((t["price"] for t in tokens if t.get("outcome") == "Yes"), None)
        no_price  = next((t["price"] for t in tokens if t.get("outcome") == "No"),  None)
        rows.append({
            "question":       m.get("question"),
            "condition_id":   m.get("condition_id"),
            "yes_price":      yes_price,
            "no_price":       no_price,
            "end_date":       m.get("end_date_iso"),
            "volume":         m.get("volume"),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = fetch_coinbase_candles()
    print(df.tail())

    print("\nActive BTC Polymarkets:")
    pm = fetch_btc_polymarkets()
    if not pm.empty:
        print(pm[["question", "yes_price", "no_price"]].to_string())
