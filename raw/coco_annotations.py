import json
import os

# 读取JSON文件
with open(f'raw1/01.json') as file:
    data = json.load(file)

    # 获取shapes中的points值
points = []
for shape in data['shapes']:
    points.extend(shape['points'][0] + [0.9])

print(points)