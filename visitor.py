import pandas as pd
import matplotlib.pyplot as plt

# 讀取資料
df = pd.read_csv("air_visit_data.csv")

# 畫圖
plt.figure(figsize=(10,6))
plt.hist(df['visitors'], bins=100, color='skyblue', edgecolor='black')
plt.xlabel("Number of Visitors")
plt.ylabel("Frequency")
plt.title("Distribution of Visitors in air_visit_data.csv")
plt.grid(True)
plt.tight_layout()
plt.show()
