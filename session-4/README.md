# PyTest & TDD Practice

这个项目包含 Python 基础练习和对应的 PyTest 测试用例。

## 项目结构

```
session-4/
├── practice_pytest_TDD.py    # Python 练习实现
├── tests/
│   ├── __init__.py
│   └── test_all.py           # 测试用例
├── requirements.txt          # 依赖包
└── README.md                 # 说明文档
```

## 练习内容

1. **列表练习**：查找列表中的重复元素
2. **集合练习**：返回只在第一个集合中的元素
3. **元组练习**：返回所有元素平方后的元组
4. **字典练习**：合并两个字典，相同键的值相加
5. **面向对象练习**：实现简单的 ToDo 列表类
6. **函数练习**：展平嵌套列表（一层）

## 如何运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行所有测试

```bash
pytest tests/test_all.py -v
```

### 3. 运行特定测试

```bash
pytest tests/test_all.py::test_find_duplicates -v
```

### 4. 查看测试覆盖率（可选）

```bash
pip install pytest-cov
pytest tests/test_all.py --cov=practice_pytest_TDD
```

## 测试结果

所有 6 个测试都应该通过：

- ✅ test_find_duplicates
- ✅ test_difference_set
- ✅ test_square_tuple
- ✅ test_merge_dicts
- ✅ test_todo_class
- ✅ test_flatten_list_once

## CI/CD

项目配置了 GitHub Actions，每次推送代码时会自动运行测试。

查看测试状态：进入 GitHub 仓库 → Actions 标签页
