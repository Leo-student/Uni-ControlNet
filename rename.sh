#!/bin/bash

# 遍历所有的 mask*.png 文件
for file in mask*.png; do
    # 提取数字部分
    echo "$file"
    # number=$(echo "$file" | sed 's/mask\([0-9]*\)\.png/\1/')
    # # 构建新的文件名
    # new_name="input${number}.png"
    # # 重命名文件
    # mv "$file" "$new_name"
done

echo "All files have been renamed."
