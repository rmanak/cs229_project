import pandas as pd
import datetime
import pytz
import subprocess
import sys
from io import StringIO


def find_unix_timestamps(date):
    # Convert string date to datetime and adjust times
    one_day_before = date - datetime.timedelta(days=1)
    while one_day_before.weekday() not in range(1, 6):
        one_day_before = one_day_before - datetime.timedelta(days=1)
    three_days_after = date + datetime.timedelta(days=3)
    while three_days_after.weekday() not in range(1, 6):
        three_days_after = three_days_after + datetime.timedelta(days=1)

    # Convert to Unix timestamp
    period1 = date_to_eastern_unix_timestamp(one_day_before)
    period2 = date_to_eastern_unix_timestamp(three_days_after)

    return period1, period2


def date_to_eastern_unix_timestamp(date):
    # Define the time zone
    tz = pytz.timezone("US/Eastern")
    return int(tz.localize(date.replace(hour=21, minute=0, second=0)).timestamp())


def download_stock_data(symbol, period1, period2):
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    # Set User-Agent to prevent HTTP 429: https://stackoverflow.com/a/78111952/207384
    # Use subprocess to run the curl command and capture its output
    curl_command = ["curl", "-v", url]
    try:
        # Run the command, capture stdout, and decode bytes to a string
        result = subprocess.run(
            curl_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )

        # Access the standard output
        curl_output = result.stdout

        # Print or process the output as needed
    except subprocess.CalledProcessError as e:
        print(f"Error running curl: {e}", file=sys.stderr)
        return None
    frame = pd.read_csv(StringIO(curl_output))
    # Add Symbol column and put at front
    frame["Symbol"] = symbol
    frame["Close"] = frame["Close"].astype(float)
    frame = frame[["Symbol"] + list(frame.columns)[:-1]]
    return frame

def download_sofr_data(period1, period2):
    start_date = datetime.datetime.utcfromtimestamp(period1).replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d")
    end_date = datetime.datetime.utcfromtimestamp(period2).replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d")
    url = f"https://markets.newyorkfed.org/read?startDt={start_date}&endDt={end_date}&eventCodes=520&productCode=50&sort=postDt:-1,eventCode:1&format=csv"
    # Set User-Agent to prevent HTTP 429: https://stackoverflow.com/a/78111952/207384
    # Use subprocess to run the curl command and capture its output
    curl_command = ["curl", "-v", url]
    try:
        # Run the command, capture stdout, and decode bytes to a string
        result = subprocess.run(
            curl_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )

        # Access the standard output
        curl_output = result.stdout

        # Print or process the output as needed
    except subprocess.CalledProcessError as e:
        print(f"Error running curl: {e}", file=sys.stderr)
        return None
    frame = pd.read_csv(StringIO(curl_output))
    # Add Symbol and Date columns and put at front
    frame["Symbol"] = "SOFR"
    frame["Close"] = frame["Rate (%)"].astype(float)
    frame["Date"] = pd.to_datetime(frame["Effective Date"], format="%m/%d/%Y").astype(str)
    front_cols = ["Symbol", "Date"]
    frame = frame[front_cols + [col for col in frame.columns if col not in front_cols]]
    return frame


def compute_alpha(frame, stock_index, risk_free_symbol):
    return_frame = pd.DataFrame(columns=["Symbol", "Return", "Market Return", "Risk-Free Return", "Excess Return"])
    return_frame.set_index(["Symbol"], inplace=True)
    for symbol in set(x[0] for x in frame.index):
        if (symbol == risk_free_symbol) or (symbol == stock_index):
            continue
        # Compute the risk-adjusted return
        symbol_frame = frame.loc[symbol]
        symbol_min_date = symbol_frame.index.min()
        symbol_max_date = symbol_frame.index.max()
        symbol_return = (symbol_frame.loc[symbol_max_date]["Close"] - symbol_frame.loc[symbol_min_date]["Close"]) / symbol_frame.loc[symbol_min_date]["Close"]
        stock_index_return = (frame.loc[stock_index, symbol_max_date]["Close"] - frame.loc[stock_index, symbol_min_date]["Close"]) / frame.loc[stock_index, symbol_min_date]["Close"]

        risk_free_frame = frame.loc[risk_free_symbol]
        risk_free_frame = risk_free_frame.loc[(risk_free_frame.index >= symbol_min_date) & (risk_free_frame.index <= symbol_max_date)]
        risk_free_return = 1
        max_i = len(risk_free_frame.index.values) - 1
        for i, date in enumerate(risk_free_frame.index.values):
            if i == max_i: break
            risk_free_return *= (1 + risk_free_frame.loc[date]["Close"] / 100)**(1./365)
        risk_free_return = risk_free_return - 1

        return_frame.loc[symbol, "Return"] = symbol_return
        return_frame.loc[symbol, "Market Return"] = stock_index_return
        return_frame.loc[symbol, "Risk-Free Return"] = risk_free_return
        return_frame.loc[symbol, "Excess Return"] = (symbol_return - risk_free_return) - (stock_index_return - risk_free_return)
    return return_frame

def main(input_file, start_symbol):
    # Read input CSV
    dataframe = pd.read_csv(input_file, dtype={"Symbol": str, "Date": str, "Year": str})

    # Iterate over rows
    start_symbol_encountered = False
    output_frame = None
    for i, (index, row) in enumerate(dataframe.iterrows()):
        # if i > 1:
        #     break
        stock_symbol, raw_date, raw_year = row["Symbol"], row["Date"], row["Year"]
        if start_symbol:
            if stock_symbol == start_symbol:
                start_symbol_encountered = True
            elif not start_symbol_encountered:
                print(
                    f"Skipping {stock_symbol} until start symbol {start_symbol} encountered...",
                    file=sys.stderr,
                )
                continue
        date = pd.to_datetime(
            "{}, {}".format(row["Date"], row["Year"]), format="%b. %d, %Y"
        )
        # Find UNIX timestamps
        period1, period2 = find_unix_timestamps(date)

        # Download stock data
        try:
            stock_data = download_stock_data(stock_symbol, period1, period2)
        except Exception as e:
            print(f"Error downloading stock data for {stock_symbol}: {e}", file=sys.stderr)
            continue
        if stock_data is not None:
            if output_frame is None:
                output_frame = stock_data
            else:
                output_frame = pd.merge(output_frame, stock_data, how="outer")
            # Extract and print Close price
            print(f"Stock: {stock_symbol}", file=sys.stderr)
            print(stock_data[["Date", "Close"]], file=sys.stderr)
        else:
            print(
                f"Couldn't download or extract data for {stock_symbol}", file=sys.stderr
            )
    # Now download S&P 500 (SPY)
    min_date = pd.to_datetime(output_frame["Date"].min()) - datetime.timedelta(days=1)
    max_date = pd.to_datetime(output_frame["Date"].max())
    period1 = date_to_eastern_unix_timestamp(min_date)
    period2 = date_to_eastern_unix_timestamp(max_date)
    sp500_data = download_stock_data("SPY", period1, period2)
    sofr_data = download_sofr_data(period1, period2)
    output_frame = pd.merge(output_frame, sp500_data, how="outer")
    output_frame = pd.merge(output_frame, sofr_data, how="outer")
    output_frame.set_index(["Symbol", "Date"], inplace=True)
    output_frame.sort_index(inplace=True)
    alpha_frame = compute_alpha(output_frame, stock_index="SPY", risk_free_symbol="SOFR")
    print(output_frame, file=sys.stderr)
    with open(data_output_file, "w") as f:
        f.write(output_frame.to_csv())
    print(alpha_frame, file=sys.stderr)
    with open(return_output_file, "w") as f:
        f.write(alpha_frame.to_csv())

if __name__ == "__main__":
    input_file = sys.argv[1]
    start_symbol = None
    data_output_file = sys.argv[2]
    return_output_file = sys.argv[3]
    main(input_file, start_symbol)
