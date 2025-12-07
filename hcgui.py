import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

warnings.filterwarnings('ignore')  # 屏蔽无关警告

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MABacktestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("均线策略回测系统")
        self.root.geometry("1200x800")

        # 初始化变量
        self.pro = None
        self.backtest_result = None
        self.hold_result = None
        self.df = None

        # 构建界面
        self._create_widgets()

    def _create_widgets(self):
        # ========== 1. 输入区域 ==========
        input_frame = ttk.LabelFrame(self.root, text="参数设置", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        # 1.1 基础参数
        ttk.Label(input_frame, text="Tushare Token：").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.token_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.token_var, width=30).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="股票代码：").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.stock_code_var = tk.StringVar(value="601899.SH")  # 默认紫金矿业
        ttk.Entry(input_frame, textvariable=self.stock_code_var, width=15).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(input_frame, text="开始日期(YYYYMMDD)：").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.start_date_var = tk.StringVar(value="20221201")
        ttk.Entry(input_frame, textvariable=self.start_date_var, width=15).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="结束日期(YYYYMMDD)：").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.end_date_var = tk.StringVar(value="20251205")
        ttk.Entry(input_frame, textvariable=self.end_date_var, width=15).grid(row=1, column=3, padx=5, pady=5)

        # 1.2 均线参数
        ttk.Label(input_frame, text="短期均线1(MA)：").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.ma5_var = tk.StringVar(value="5")
        ttk.Entry(input_frame, textvariable=self.ma5_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="短期均线2(MA)：").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        self.ma7_var = tk.StringVar(value="7")
        ttk.Entry(input_frame, textvariable=self.ma7_var, width=10).grid(row=2, column=3, padx=5, pady=5)

        ttk.Label(input_frame, text="长期均线(MA)：").grid(row=2, column=4, sticky=tk.W, padx=5, pady=5)
        self.ma20_var = tk.StringVar(value="20")
        ttk.Entry(input_frame, textvariable=self.ma20_var, width=10).grid(row=2, column=5, padx=5, pady=5)

        # 1.3 交易成本
        ttk.Label(input_frame, text="佣金率(‰)：").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.commission_var = tk.StringVar(value="0.5")  # 0.5‰ = 0.05%
        ttk.Entry(input_frame, textvariable=self.commission_var, width=10).grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="印花税率(‰)：").grid(row=3, column=2, sticky=tk.W, padx=5, pady=5)
        self.stamp_var = tk.StringVar(value="1")  # 1‰ = 0.1%
        ttk.Entry(input_frame, textvariable=self.stamp_var, width=10).grid(row=3, column=3, padx=5, pady=5)

        ttk.Label(input_frame, text="滑点率(‰)：").grid(row=3, column=4, sticky=tk.W, padx=5, pady=5)
        self.slippage_var = tk.StringVar(value="0.2")  # 0.2‰ = 0.02%
        ttk.Entry(input_frame, textvariable=self.slippage_var, width=10).grid(row=3, column=5, padx=5, pady=5)

        # 1.4 执行按钮
        ttk.Button(input_frame, text="开始回测", command=self.run_backtest).grid(row=4, column=0, columnspan=6, pady=10)

        # ========== 2. 结果展示区域 ==========
        result_frame = ttk.Frame(self.root)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 2.1 数值结果
        result_text_frame = ttk.LabelFrame(result_frame, text="回测结果", padding=10)
        result_text_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.result_text = scrolledtext.ScrolledText(result_text_frame, width=40, height=20)
        self.result_text.pack(fill=tk.Y, expand=True)

        # 2.2 可视化结果
        plot_frame = ttk.LabelFrame(result_frame, text="收益曲线", padding=10)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _validate_params(self):
        """验证输入参数"""
        try:
            # 基础参数
            token = self.token_var.get().strip()
            if not token:
                raise ValueError("Tushare Token不能为空")

            stock_code = self.stock_code_var.get().strip()
            if not stock_code:
                raise ValueError("股票代码不能为空")

            start_date = self.start_date_var.get().strip()
            end_date = self.end_date_var.get().strip()
            if len(start_date) != 8 or len(end_date) != 8:
                raise ValueError("日期格式必须为YYYYMMDD")

            # 均线参数
            ma5 = int(self.ma5_var.get())
            ma7 = int(self.ma7_var.get())
            ma20 = int(self.ma20_var.get())
            if ma5 <= 0 or ma7 <= 0 or ma20 <= 0:
                raise ValueError("均线参数必须为正整数")

            # 交易成本
            commission = float(self.commission_var.get()) / 1000  # 转成小数
            stamp = float(self.stamp_var.get()) / 1000
            slippage = float(self.slippage_var.get()) / 1000
            if commission < 0 or stamp < 0 or slippage < 0:
                raise ValueError("交易成本参数不能为负数")

            return {
                "token": token,
                "stock_code": stock_code,
                "start_date": start_date,
                "end_date": end_date,
                "ma5": ma5,
                "ma7": ma7,
                "ma20": ma20,
                "commission": commission,
                "stamp": stamp,
                "slippage": slippage
            }
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return None

    def _get_stock_data(self, params):
        """获取并预处理股票数据"""
        try:
            pro = ts.pro_api(params["token"])
            start_date = params["start_date"]

            start = datetime.strptime(start_date, "%Y%m%d")
            new_date = start - timedelta(days=50)
            new_start_date = new_date.strftime("%Y%m%d")

            df = pro.daily(
                ts_code=params["stock_code"],
                start_date=new_start_date,
                end_date=params["end_date"]
            )
            if df.empty:
                raise ValueError("未获取到股票数据，请检查代码和日期")

            df = df.sort_values("trade_date").reset_index(drop=True)
            df["trade_date"] = pd.to_datetime(df["trade_date"])

            # 计算均线
            df["ma5"] = df["close"].rolling(window=params["ma5"]).mean()
            df["ma7"] = df["close"].rolling(window=params["ma7"]).mean()
            df["ma20"] = df["close"].rolling(window=params["ma20"]).mean()
            df = df.bfill()  # 填充空值
            df = df[df["trade_date"] >= start]
            self.pro = pro
            return df
        except Exception as e:
            messagebox.showerror("数据获取失败", f"错误信息：{str(e)}")
            return None

    def _calculate_signals(self, df, params):
        """计算交易信号"""
        # 金叉/死叉信号
        df["golden_cross_520"] = (df["ma5"].shift(1) < df["ma20"].shift(1)) & (df["ma5"] > df["ma20"])
        df["death_cross_57"] = (df["ma5"].shift(1) > df["ma7"].shift(1)) & (df["ma5"] < df["ma7"])
        df["golden_cross_57"] = (df["close"] > df["ma20"]) & (df["ma5"].shift(1) < df["ma7"].shift(1)) & (
                    df["ma5"] > df["ma7"])
        return df

    def _backtest_strategy(self, df, params):
        """执行策略回测"""
        position = 0  # 0空仓，1持仓
        cash = 100000  # 初始资金10万
        shares = 0
        trade_records = []
        equity_curve = []

        for idx, row in df.iterrows():
            open_price = row["open"]
            close_price = row["close"]

            # 初始入场：5/20金叉 + 空仓
            if row["golden_cross_520"] and position == 0:
                buy_price = open_price * (1 + params["slippage"])
                buy_amount = cash // (buy_price * 100) * 100
                if buy_amount > 0:
                    cost = buy_amount * buy_price * (1 + params["commission"])
                    cash -= cost
                    shares = buy_amount
                    position = 1
                    trade_records.append({
                        "date": row["trade_date"],
                        "type": "初始买入",
                        "price": buy_price,
                        "shares": buy_amount
                    })

            # 离场：5/7死叉 + 持仓
            elif row["death_cross_57"] and position == 1:
                sell_price = open_price * (1 - params["slippage"])
                sell_amount = shares
                revenue = sell_amount * sell_price * (1 - params["commission"] - params["stamp"])
                cash += revenue
                shares = 0
                position = 0
                trade_records.append({
                    "date": row["trade_date"],
                    "type": "卖出",
                    "price": sell_price,
                    "shares": sell_amount
                })

            # 再入场：5/7金叉 + 空仓 + 20日均线之上
            elif row["golden_cross_57"] and position == 0:
                buy_price = open_price * (1 + params["slippage"])
                buy_amount = cash // (buy_price * 100) * 100
                if buy_amount > 0:
                    cost = buy_amount * buy_price * (1 + params["commission"])
                    cash -= cost
                    shares = buy_amount
                    position = 1
                    trade_records.append({
                        "date": row["trade_date"],
                        "type": "再买入",
                        "price": buy_price,
                        "shares": buy_amount
                    })

            # 计算当日净值
            if position == 1:
                equity = cash + shares * close_price
            else:
                equity = cash
            equity_curve.append(equity)

        # 最终平仓
        if position == 1:
            final_sell_price = df.iloc[-1]["close"] * (1 - params["slippage"])
            final_revenue = shares * final_sell_price * (1 - params["commission"] - params["stamp"])
            cash += final_revenue
            shares = 0
            trade_records.append({
                "date": df.iloc[-1]["trade_date"],
                "type": "最终平仓",
                "price": final_sell_price,
                "shares": shares
            })
            equity_curve[-1] = cash

        # 计算收益指标
        initial_equity = 100000
        final_equity = cash
        total_return = (final_equity - initial_equity) / initial_equity * 100

        # 最大回撤
        equity_series = pd.Series(equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100
        max_drawdown = drawdown.min()

        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "trade_count": len(trade_records),
            "final_equity": final_equity,
            "equity_curve": equity_curve,
            "trade_records": trade_records
        }

    def _calculate_hold_return(self, df, params):
        """计算持有不动收益"""
        initial_open = df.iloc[0]["open"]
        final_close = df.iloc[-1]["close"]

        # 初始10万可买数量
        hold_buy_amount = 100000 // (initial_open * 100) * 100
        hold_cost = hold_buy_amount * initial_open * (1 + params["commission"] + params["slippage"])
        hold_revenue = hold_buy_amount * final_close * (1 - params["commission"] - params["stamp"] - params["slippage"])
        hold_final_equity = 100000 - hold_cost + hold_revenue
        hold_total_return = (hold_final_equity - 100000) / 100000 * 100

        # 计算持有不动净值曲线
        hold_equity = []
        initial_hold_cash = 100000
        hold_shares = initial_hold_cash // (initial_open * 100) * 100
        for idx, row in df.iterrows():
            current_close = row["close"]
            remaining_cash = initial_hold_cash - (
                        hold_shares * initial_open * (1 + params["commission"] + params["slippage"]))
            current_equity = remaining_cash + hold_shares * current_close
            hold_equity.append(current_equity / initial_hold_cash)

        return {
            "total_return": hold_total_return,
            "final_equity": hold_final_equity,
            "equity_curve": hold_equity
        }

    def _plot_result(self, df, strategy_equity, hold_equity, params):
        """绘制收益曲线"""
        self.ax.clear()
        # 归一化策略净值
        strategy_equity_norm = pd.Series(strategy_equity) / strategy_equity[0]
        # 绘制曲线
        self.ax.plot(df["trade_date"], strategy_equity_norm,
                     label=f"波段策略（收益率：{self.backtest_result['total_return']:.2f}%）",
                     color="red", linewidth=1.5)
        self.ax.plot(df["trade_date"], hold_equity,
                     label=f"持有不动（收益率：{self.hold_result['total_return']:.2f}%）",
                     color="blue", linewidth=1.5)
        # 美化图表
        self.ax.set_title(f"{params['stock_code']} 均线策略回测收益曲线", fontsize=12)
        self.ax.set_xlabel("日期", fontsize=10)
        self.ax.set_ylabel("归一化净值（初始=1）", fontsize=10)
        self.ax.legend(loc="upper left")
        self.ax.grid(True, alpha=0.3)
        self.ax.tick_params(axis='x', rotation=45)
        # 更新画布
        self.fig.tight_layout()
        self.canvas.draw()

    def _show_result_text(self, params):
        """显示数值结果（修复f-string语法错误）"""
        self.result_text.delete(1.0, tk.END)
        # 核心修复：把错误的 MA{params['ma7']} 改为 {params['ma7']}
        result_str = f"""【回测参数】
股票代码：{params['stock_code']}
回测时间：{params['start_date']} - {params['end_date']}
均线参数：MA{params['ma5']}/{params['ma7']}/{params['ma20']}
交易成本：佣金{params['commission'] * 1000}‰ | 印花税{params['stamp'] * 1000}‰ | 滑点{params['slippage'] * 1000}‰

【波段策略结果】
累计收益率：{self.backtest_result['total_return']:.2f}%
最大回撤：{self.backtest_result['max_drawdown']:.2f}%
交易次数：{self.backtest_result['trade_count']}次
最终资产：{self.backtest_result['final_equity']:.2f}元

【持有不动结果】
累计收益率：{self.hold_result['total_return']:.2f}%
最终资产：{self.hold_result['final_equity']:.2f}元

【对比结论】
{('波段策略收益更高' if self.backtest_result['total_return'] > self.hold_result['total_return'] else '持有不动收益更高')}
收益差值：{abs(self.backtest_result['total_return'] - self.hold_result['total_return']):.2f}个百分点
"""
        self.result_text.insert(tk.END, result_str)

    def run_backtest(self):
        """主回测流程"""
        # 1. 验证参数
        params = self._validate_params()
        if not params:
            return

        # 2. 获取数据
        self.df = self._get_stock_data(params)
        if self.df is None:
            return

        # 3. 计算信号
        self.df = self._calculate_signals(self.df, params)

        # 4. 执行策略回测
        self.backtest_result = self._backtest_strategy(self.df, params)

        # 5. 计算持有不动收益
        self.hold_result = self._calculate_hold_return(self.df, params)

        # 6. 展示结果
        self._plot_result(self.df, self.backtest_result["equity_curve"], self.hold_result["equity_curve"], params)
        self._show_result_text(params)

        messagebox.showinfo("回测完成", "策略回测已执行完毕，结果已展示！")


if __name__ == "__main__":
    root = tk.Tk()
    app = MABacktestGUI(root)
    root.mainloop()
