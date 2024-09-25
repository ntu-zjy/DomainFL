import random

random.seed(42)
# 定义域和类别
domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
num_classes = 50

# 为每个域生成一个包含两个随机类别的字典
domain_class_dict = {}

for domain in domains:
    # 从50个类别中随机选择2个类别
    selected_classes = random.sample(range(num_classes), 2)
    domain_class_dict[domain] = {selected_classes[0]:0,  # random.randint(1, 100),  # 随机生成1到100的样本数量
                                 selected_classes[1]: 0} # random.randint(1, 100)

# 打印结果
for domain, class_dict in domain_class_dict.items():
    print(f"Domain: {domain}, Selected Classes: {class_dict}")
