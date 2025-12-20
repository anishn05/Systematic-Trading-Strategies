import yaml
import logging
from pathlib import Path
import pandas as pd

from data.data_loader import DataLoader
from src.strategies.sma_strategy import SMAStrategy
from src.backtest.backtester import Backtester
from src.analytics.performance import PerformanceAnalyzer
from src.execution.execution import Order, Fill, ExecutionModel

# ---------------------------------------------------
# Setup logging
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# ---------------------------------------------------
# Load configuration
# ---------------------------------------------------
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

#MAIN
def main():

    logger.info("Starting algorithmic trading system run")
    # -------------------------------
    # 1. Load config
    # -------------------------------
    config = load_config("config/settings.yaml")
    initial_capital = config["capital"]["initial_capital"]
    leverage = config["capital"]["leverage"]

    #CHANGE INTPUTS HERE!!
    symbol = "AAPL"
    sma_params = config["strategy"]["sma"]
    strategy = SMAStrategy(
        short_window=sma_params["short_window"],
        long_window=sma_params["long_window"]
        #position_size=sma_params["position_size"]
    )


    # -------------------------------
    # 2. Load market data
    # -------------------------------
    data_loader = DataLoader(
        data_dir="data",
        frequency=config["backtest"]["frequency"]
    )
    market_data = data_loader.load_csv(
        symbol=symbol,
        timeframe = config["backtest"]["frequency"],
        parse_dates=False
        #start_date="2018-01-01",
        #end_date="2024-01-01"
    )
    logger.info(f"Loaded {len(market_data)} rows of data for {symbol}")
    # -------------------------------
    # 3. Instantiate backtester
    # -------------------------------
    backtester = Backtester(
        data=market_data,
        strategy=strategy,
        settings=config
        #leverage=leverage,
        #transaction_cost_bps=config["costs"]["transaction_cost_bps"],
        #allow_partial_fills=config["backtest"]["allow_partial_fills"]
    )
    # -------------------------------
    # 4. Run backtest
    # -------------------------------
    results = backtester.run()
    logger.info("Backtest completed")
    # -------------------------------
    # 5. Display results
    # -------------------------------
    print("\n===== BACKTEST SUMMARY =====")
    print(f"Final equity: £{results['final_equity']:.2f}")
    print(f"Total return: {results['total_return']:.2%}")
    print(f"Number of trades: {results['num_trades']}")
    print(f"Max drawdown: {results['max_drawdown']:.2%}")


    perf = PerformanceAnalyzer(
    equity_curve=results["equity_curve"],
    returns=results["returns"],
    trades=results["trades"],
    )

    summary = perf.summary()

    print("\n===== PERFORMANCE SUMMARY =====")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.4f}")
        else:
            print(f"{k:25s}: {v}")

    # -------------------------------
    # 7. Save results
    # -------------------------------
    output_path = Path("reports/performance")
    output_path.mkdir(parents=True, exist_ok=True)
    results["equity_curve"].to_csv(output_path / f"equity_curve_{symbol}.csv")
    results["trades"].to_csv(output_path / f"trades_{symbol}.csv")
    pd.Series(summary).to_csv(output_path / f"summary_metrics_{symbol}.csv")
    logger.info("Results saved to reports/performance/")

if __name__ == "__main__":
    main()



