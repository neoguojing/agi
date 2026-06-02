import os

class InvestmentMonitor:
    def __init__(self):
        # TSLA Trigger points: Warning $418.34, Critical $360, Emergency $315, Liquidation $270
        self.tsla_triggers = {
            "warning": 418.34,
            "critical": 360.00,
            "emergency": 315.00,
            "liquidation": 270.00
        }
        self.rklb_volatility_limit = 0.25
        self.vix_limit = 20.0

    def check_tsla_status(self, current_price):
        if current_price <= self.tsla_triggers["liquidation"]:
            return "LIQUIDATION"
        elif current_price <= self.tsla_triggers["emergency"]:
            return "EMERGENCY"
        elif current_price <= self.tsla_triggers["critical"]:
            return "CRITICAL"
        elif current_price <= self.tsla_triggers["warning"]:
            return "WARNING"
        return "STABLE"

    def check_rklb_status(self, current_volatility, current_vix):
        if current_vix > self.vix_limit or current_volatility > self.rklb_volatility_limit:
            return "LIQUIDATE_RKLB"
        return "STABLE"

if __name__ == "__main__":
    monitor = InvestmentMonitor()
    current_tsla = 415.88
    print(f"TSLA Status: {monitor.check_tsla_status(current_tsla)}")
