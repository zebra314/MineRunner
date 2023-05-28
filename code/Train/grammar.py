string = "(0, 0) (0, 0) (0, 0) (1, 0) (0, -1) (0, 0) (0, 0) (2, 0) (0, 0) (0, 0) (0, 0) (0, 0) (1, 0) (0, 0) (0, 0) (0, 0) (0, 0)"
data_list = []

for pair in string.split():
    pair = pair.strip('()')  # 去除括号
    a, b = pair.split(',')
    a = int(a.strip())
    b = int(b.strip())
    
    if '-' in a:  # 判断a是否为负数
        a = int(a)  # 转换为整数
    if '-' in b:  # 判断b是否为负数
        b = int(b)  # 转换为整数
    
    data_list.append((a, b))  # 存储为元组形式的列表
