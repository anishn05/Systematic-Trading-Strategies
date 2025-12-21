import yaml
import logging
from pathlib import Path
import pandas as pd

from data.data_loader import DataLoader
from data.data_validator import DataValidator
from src.strategies.sma_strategy import SMAStrategy
from src.backtest.backtester import Backtester
from src.analytics.performance import PerformanceAnalyzer

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

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    logger.info("Starting algorithmic trading system run")

    # -------------------------------
    # 1. Load config
    # -------------------------------
    config = load_config("config/settings.yaml")
    initial_capital = config["capital"]["initial_capital"]
    leverage = config["capital"]["leverage"]

    # CHANGE INPUTS HERE
    symbols = ["AAPL"]  # single asset
    sma_params = config["strategy"]["sma"]

    strategy = SMAStrategy(
        short_window=sma_params["short_window"],
        long_window=sma_params["long_window"]
    )

    # -------------------------------
    # 2. Load market data
    # -------------------------------
    data_loader = DataLoader(
        data_dir="data",
        frequency=config["backtest"]["frequency"]
    )

    # multi-asset loading
    market_data = data_loader.load_multiple(symbols=symbols, timeframe=config["backtest"]["frequency"])

    validator = DataValidator()
    validator.validate_multiple(market_data)
    logger.info("All assets validated successfully!")

    # -------------------------------
    # 3. Run backtests
    # -------------------------------
    for symbol in symbols:
        logger.info(f"Backtesting {symbol} with {len(market_data[symbol])} rows of data")
        
        backtester = Backtester(
            data=market_data[symbol],
            strategy=strategy,
            settings=config
        )

        results = backtester.run()
        logger.info(f"Backtest for {symbol} completed")

        # -------------------------------
        # 4. Display Backtest Summary
        # -------------------------------
        print(f"\n===== BACKTEST SUMMARY ({symbol}) =====")
        print(f"Final equity       : £{results['final_equity']:.4f}")
        print(f"Total return       : {results['total_return']:.4%}")
        print(f"Number of trades   : {results['num_trades']}")
        print(f"Max drawdown       : {results['max_drawdown']:.4%}")

        # -------------------------------
        # 5. Performance Summary
        # -------------------------------
        perf = PerformanceAnalyzer(
            equity_curve=results["equity_curve"],
            returns=results["returns"],
            trades=results["trades"],
        )
        summary = perf.summary()

        print(f"\n===== PERFORMANCE SUMMARY ({symbol}) =====")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"{k:25s}: {v:.4f}")
            else:
                print(f"{k:25s}: {v}")

        # -------------------------------
        # 6. Save results
        # -------------------------------
        output_path = Path("reports/performance")
        output_path.mkdir(parents=True, exist_ok=True)
        results["equity_curve"].to_csv(output_path / f"equity_curve_{symbol}.csv")
        results["trades"].to_csv(output_path / f"trades_{symbol}.csv")
        pd.Series(summary).to_csv(output_path / f"summary_metrics_{symbol}.csv")
        logger.info(f"Results saved for {symbol} to reports/performance/")

if __name__ == "__main__":
    main()
