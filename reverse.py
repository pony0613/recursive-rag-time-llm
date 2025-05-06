import pandas as pd

# 读取CSV文件
input_file = "cryptonews.csv"  # 替换为你的文件路径
output_file = "sorted_cryptonews.csv"  # 输出文件路径

# 使用 pandas 读取 CSV 文件
df = pd.read_csv(input_file)

# 确保 'date' 列是 datetime 格式
df['date'] = pd.to_datetime(df['date'])

# 按照 'date' 列逆序排序
df_sorted = df.sort_values(by='date', ascending=True)

# 将排序后的数据保存到新的 CSV 文件
df_sorted.to_csv(output_file, index=False)

print(f"新闻已按照时间戳逆序排序，并保存到文件：{output_file}")