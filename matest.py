import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 1. 配置参数 ----------------------
# Tushare token（需替换为自己的，注册地址：https://tushare.pro/register?reg=467877）
TS_TOKEN = "847769c027878def477fd864c8c27bcd92ff367ca560450e935fde1c"
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 回测标的与时间
STOCK_CODE = "601899.SH"  # 紫金矿业（可替换为300502.SZ新易盛/002304.SZ洋河股份）
START_DATE = "20201201"  # 回测起始时间
END_DATE = "20251205"  # 回测结束时间

# 均线参数
MA5 = 5  # 5日均线
MA7 = 7  # 7日均线
MA20 = 20  # 20日均线

# 交易成本（单次）
COMMISSION_RATE = 0.0005  # 佣金0.05%
STAMP_TAX_RATE = 0.001  # 印花税0.1%（仅卖出收取）
SLIPPAGE_RATE = 0.0002  # 滑点0.02%


# ---------------------- 2. 获取并预处理数据 ----------------------
def get_stock_data(code, start_date, end_date):
    """获取日线数据并计算均线（全版本兼容）"""
    # 获取日线数据
    df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
    # 按交易日期升序排列
    df = df.sort_values("trade_date").reset_index(drop=True)
    # 转换日期格式
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # 计算均线（前20天会有NaN）
    df["ma5"] = df["close"].rolling(window=MA5).mean()
    df["ma7"] = df["close"].rolling(window=MA7).mean()
    df["ma20"] = df["close"].rolling(window=MA20).mean()
    # 修复fillna警告：用bfill()替代fillna(method="bfill")
    df = df.bfill()  # 向后填充空值（优先用后续数据填充前期均线空值）
    return df


# 获取数据
df = get_stock_data(STOCK_CODE, START_DATE, END_DATE)


# ---------------------- 3. 计算交易信号 ----------------------
def calculate_signals(df):
    """计算金叉/死叉信号"""
    # 1. 5/20金叉信号（初始入场）：5日均线上穿20日均线
    df["golden_cross_520"] = (df["ma5"].shift(1) < df["ma20"].shift(1)) & (df["ma5"] > df["ma20"])
    # 2. 5/7死叉信号（离场）：5日均线下穿7日均线
    df["death_cross_57"] = (df["ma5"].shift(1) > df["ma7"].shift(1)) & (df["ma5"] < df["ma7"])
    # 3. 再入场信号：20日均线之上 + 5日均线上穿7日均线
    df["golden_cross_57"] = (df["close"] > df["ma20"]) & (df["ma5"].shift(1) < df["ma7"].shift(1)) & (
                df["ma5"] > df["ma7"])
    return df


df = calculate_signals(df)


# ---------------------- 4. 执行交易逻辑 ----------------------
def backtest_strategy(df):
    """回测策略，计算收益（修复expanding参数问题）"""
    # 初始化变量
    position = 0  # 持仓状态：0空仓，1持仓
    cash = 100000  # 初始资金10万
    shares = 0  # 持仓数量
    trade_records = []  # 交易记录
    equity_curve = []  # 资产净值曲线

    for idx, row in df.iterrows():
        current_date = row["trade_date"]
        open_price = row["open"]  # 次日开盘价（实际用当日open模拟次日开盘）
        close_price = row["close"]

        # ------------ 初始入场：5/20金叉 + 空仓 ------------
        if row["golden_cross_520"] and position == 0:
            # 计算可买数量（整手，100股/手）
            buy_price = open_price * (1 + SLIPPAGE_RATE)  # 买入滑点
            buy_amount = cash // (buy_price * 100) * 100
            if buy_amount > 0:
                cost = buy_amount * buy_price * (1 + COMMISSION_RATE)  # 买入成本（含佣金）
                cash -= cost
                shares = buy_amount
                position = 1
                trade_records.append({
                    "date": current_date,
                    "type": "初始买入",
                    "price": buy_price,
                    "shares": buy_amount,
                    "cash": cash
                })

        # ------------ 离场：5/7死叉 + 持仓 ------------
        elif row["death_cross_57"] and position == 1:
            # 卖出持仓
            sell_price = open_price * (1 - SLIPPAGE_RATE)  # 卖出滑点
            sell_amount = shares
            revenue = sell_amount * sell_price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)  # 卖出收益（含佣金+印花税）
            cash += revenue
            shares = 0
            position = 0
            trade_records.append({
                "date": current_date,
                "type": "卖出",
                "price": sell_price,
                "shares": sell_amount,
                "cash": cash
            })

        # ------------ 再入场：5/7金叉 + 空仓 + 20日均线之上 ------------
        elif row["golden_cross_57"] and position == 0:
            # 计算可买数量
            buy_price = open_price * (1 + SLIPPAGE_RATE)
            buy_amount = cash // (buy_price * 100) * 100
            if buy_amount > 0:
                cost = buy_amount * buy_price * (1 + COMMISSION_RATE)
                cash -= cost
                shares = buy_amount
                position = 1
                trade_records.append({
                    "date": current_date,
                    "type": "再买入",
                    "price": buy_price,
                    "shares": buy_amount,
                    "cash": cash
                })

        # 计算当日资产净值（持仓时=现金+股票市值，空仓时=现金）
        if position == 1:
            equity = cash + shares * close_price
        else:
            equity = cash
        equity_curve.append(equity)

    # 最终平仓（若仍持仓）
    if position == 1:
        final_sell_price = df.iloc[-1]["close"] * (1 - SLIPPAGE_RATE)
        final_revenue = shares * final_sell_price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)
        cash += final_revenue
        shares = 0
        position = 0
        trade_records.append({
            "date": df.iloc[-1]["trade_date"],
            "type": "最终平仓",
            "price": final_sell_price,
            "shares": shares,
            "cash": cash
        })
        # 更新最后一天净值
        equity_curve[-1] = cash

    # 计算收益指标
    initial_equity = 100000
    final_equity = cash
    total_return = (final_equity - initial_equity) / initial_equity * 100  # 累计收益率（%）

    # 修复max_periods警告：新版Pandas expanding()无需该参数
    equity_series = pd.Series(equity_curve)
    peak = equity_series.expanding().max()  # 移除max_periods=None，功能完全一致
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = drawdown.min()  # 最大回撤（%）

    # 交易次数
    trade_count = len(trade_records)

    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "trade_count": trade_count,
        "trade_records": trade_records,
        "equity_curve": equity_curve,
        "final_equity": final_equity
    }


# 执行回测
backtest_result = backtest_strategy(df)

# ---------------------- 5. 计算持有不动收益 ----------------------
# 持有不动：首日开盘买入，最后一日收盘卖出
initial_open = df.iloc[0]["open"]
final_close = df.iloc[-1]["close"]
# 初始10万可买数量
hold_buy_amount = 100000 // (initial_open * 100) * 100
hold_cost = hold_buy_amount * initial_open * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
hold_revenue = hold_buy_amount * final_close * (1 - COMMISSION_RATE - STAMP_TAX_RATE - SLIPPAGE_RATE)
hold_final_equity = 100000 - hold_cost + hold_revenue
hold_total_return = (hold_final_equity - 100000) / 100000 * 100

# ---------------------- 6. 输出结果 ----------------------
print("=" * 50)
print(f"回测标的：{STOCK_CODE}")
print(f"回测时间：{START_DATE} 至 {END_DATE}")
print("=" * 50)
# 策略收益
print(f"【波段策略】")
print(f"累计收益率：{backtest_result['total_return']:.2f}%")
print(f"最大回撤：{backtest_result['max_drawdown']:.2f}%")
print(f"交易次数：{backtest_result['trade_count']}次")
print(f"最终资产：{backtest_result['final_equity']:.2f}元")
# 持有不动收益
print(f"\n【持有不动】")
print(f"累计收益率：{hold_total_return:.2f}%")
print(f"最终资产：{hold_final_equity:.2f}元")
print("=" * 50)

# ---------------------- 7. 可视化收益曲线 ----------------------
plt.figure(figsize=(12, 6))
# 策略净值曲线
strategy_equity = pd.Series(backtest_result["equity_curve"])
strategy_equity = strategy_equity / strategy_equity.iloc[0]  # 归一化到1
# 持有不动净值曲线（精准计算）
hold_equity = []
initial_hold_cash = 100000
hold_shares = initial_hold_cash // (initial_open * 100) * 100  # 初始持仓数量
for idx, row in df.iterrows():
    current_close = row["close"]
    # 持有不动的净值 = 剩余现金 + 持仓市值
    remaining_cash = initial_hold_cash - (hold_shares * initial_open * (1 + COMMISSION_RATE + SLIPPAGE_RATE))
    current_equity = remaining_cash + hold_shares * current_close
    hold_equity.append(current_equity / initial_hold_cash)  # 归一化

plt.plot(df["trade_date"], strategy_equity, label=f"波段策略（收益率：{backtest_result['total_return']:.2f}%）",
         color="red")
plt.plot(df["trade_date"], hold_equity, label=f"持有不动（收益率：{hold_total_return:.2f}%）", color="blue")
plt.title(f"{STOCK_CODE} 策略回测收益曲线")
plt.xlabel("日期")
plt.ylabel("归一化净值（初始=1）")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
