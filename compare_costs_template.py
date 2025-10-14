import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ====== 数据输入（示例） ======
# 你可以直接把四组数据粘贴到下方，数据结构需一致：
# Header 为：Wall time,Step,Value
# 将示例数据替换为你的真实数据即可。

data_a = """
Wall time,Step,Value
1759900000.0,16000,150.0
1759900100.0,32000,200.0
1759900200.0,48000,250.0
"""

data_b = """
Wall time,Step,Value
1759900000.0,16000,140.0
1759900100.0,32000,210.0
1759900200.0,48000,260.0
"""

data_c = """
Wall time,Step,Value
1759900000.0,16000,155.0
1759900100.0,32000,205.0
1759900200.0,48000,255.0
"""

data_d = """
Wall time,Step,Value
1759900000.0,16000,130.0
1759900100.0,32000,190.0
1759900200.0,48000,240.0
"""

# ====== 可自定义的标签与配色 ======
labels = ["origin1", "origin2", "processive", "50%update"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def load_df(text: str) -> pd.DataFrame:
    """从 CSV 文本载入为 DataFrame。"""
    return pd.read_csv(StringIO(text))


def main():
    # 载入四组数据
    dfs = [load_df(data_a), load_df(data_b), load_df(data_c), load_df(data_d)]

    # 绘图
    plt.figure(figsize=(10, 6))
    for df, label, color in zip(dfs, labels, colors):
        plt.plot(df["Step"], df["Value"], label=label, color=color, linewidth=2)

    # 美化图表
    plt.title("Cost Comparison (4 Curves)", fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Cost", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # 保存与展示
    plt.tight_layout()
    plt.savefig("comparison_costs.png")
    plt.show()


if __name__ == "__main__":
    main()