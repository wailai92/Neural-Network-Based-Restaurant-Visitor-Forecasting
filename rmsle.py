import numpy as np
import matplotlib.pyplot as plt

# 三筆模型的 RMSLE
rmsle_values = {
    "Full Batch (RMSLE = 0.3326)": 0.3326,
    "Batch 1024 (RMSLE = 0.2237)": 0.2237,
    "Batch 256 (RMSLE = 0.2204)": 0.2204
}

# 真實來客數範圍
true_y = np.arange(1, 301)

# 畫圖
plt.figure(figsize=(10, 6))

for label, rmsle in rmsle_values.items():
    log_y = np.log1p(true_y)
    log_y_upper = log_y + rmsle
    #log_y_lower = log_y - rmsle

    y_upper = np.expm1(log_y_upper)
    #y_lower = np.expm1(log_y_lower)

    # 誤差百分比（上下對稱）
    percent_error_upper = (y_upper - true_y) / true_y * 100
    #percent_error_lower = (true_y - y_lower) / true_y * 100

    # 畫上下誤差線
    plt.plot(true_y, percent_error_upper, label=f'{label} %')
    #plt.plot(true_y, percent_error_lower, label=f'{label} -%')

plt.xlabel("Actual Visitors")
plt.ylabel("Error (%)")
plt.title("% Error vs. Actual Visitors ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
