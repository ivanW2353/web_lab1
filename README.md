# 信息检索实验系统

这是一个基于 Python 的信息检索实验系统，实现了布尔检索和向量空间模型检索两种检索方法。

## 功能特性

- **数据解析**: 解析 Meetup 数据集的 XML 文件
- **文本规范化**: 支持 NLTK 和内置文本处理器
- **倒排索引**: 高效的倒排索引构建和查询
- **布尔检索**: 支持 AND、OR、NOT 布尔操作
- **向量检索**: 基于 TF-IDF 的向量空间模型检索
- **缓存机制**: 自动缓存处理结果，加速后续运行

## 系统要求

- Python 3.7+
- Windows/Linux/macOS

## 安装步骤

1. **克隆或下载项目**
   ```bash
   cd web_lab1
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **准备数据集**
   - 确保 `Meetup/All_Unpack/` 目录存在
   - 将 Meetup XML 数据文件放入该目录

## 快速开始

直接运行主程序：

```bash
python main.py
```

首次运行时，系统会：
1. 自动下载 NLTK 所需数据（如果使用 NLTK）
2. 解析 XML 数据文件
3. 进行文本规范化处理
4. 构建倒排索引
5. 执行示例查询
6. 保存索引到文件

## 项目结构

```
web_lab1/
│
├── main.py                 # 主程序入口
├── data_processor.py       # 数据解析模块
├── text_normalizer.py      # 文本规范化模块
├── inverted_index.py       # 倒排索引模块
├── boolean_retrieval.py    # 布尔检索模块
├── vector_retrieval.py     # 向量检索模块
├── nltk_downloader.py      # NLTK 数据下载工具
├── requirements.txt        # 项目依赖
├── README.md               # 本文档
│
└── Meetup/                 # 数据目录
    ├── All_Unpack/         # XML 数据文件
    ├── cache/              # 缓存文件（自动生成）
    └── inverted_index.json # 倒排索引文件（自动生成）
```

## 模块说明

### 1. DataProcessor (data_processor.py)
- 解析 Meetup XML 数据文件
- 提取事件名称、描述、分组信息
- 支持缓存机制，避免重复解析

### 2. TextNormalizer (text_normalizer.py)
- 文本规范化处理（分词、去停用词、词干提取）
- 自动检测 NLTK 可用性
- 降级到内置处理器（如果 NLTK 不可用）
- 支持缓存规范化结果

### 3. InvertedIndex (inverted_index.py)
- 构建倒排索引
- 记录词项位置和频率
- 计算 TF-IDF 权重
- 支持索引保存和加载

### 4. BooleanRetrieval (boolean_retrieval.py)
- 支持布尔操作：AND、OR、NOT
- 基于倒排索引的快速检索
- 查询格式示例：
  - `"party"` - 单词查询
  - `"meeting and group"` - AND 操作
  - `"tech or computer"` - OR 操作
  - `"not business"` - NOT 操作

### 5. VectorRetrieval (vector_retrieval.py)
- 基于 TF-IDF 的向量空间模型
- 余弦相似度计算
- 返回 Top-K 相关文档
- 简化实现，不依赖 scikit-learn

## 使用示例

### 布尔检索

```python
from boolean_retrieval import BooleanRetrieval

# 初始化检索器
boolean_searcher = BooleanRetrieval(inverted_index)

# 执行查询
results = boolean_searcher.search("party")
results = boolean_searcher.search("meeting and group")
results = boolean_searcher.search("tech or computer")
results = boolean_searcher.search("not business")
```

### 向量检索

```python
from vector_retrieval import VectorRetrieval

# 初始化检索器
vector_searcher = VectorRetrieval(documents)
vector_searcher.build_tfidf_vectors()

# 执行查询
results, search_time = vector_searcher.search("technology conference", top_k=10)

# 显示结果
for doc_id, score in results:
    print(f"文档: {doc_id}, 相似度: {score:.4f}")
```

## 缓存机制

系统实现了三层缓存机制：

1. **文档缓存**: 缓存解析后的 XML 文档
2. **规范化缓存**: 缓存文本规范化结果
3. **索引缓存**: 缓存倒排索引

缓存文件存储在 `Meetup/cache/` 目录，使用 MD5 哈希命名，确保数据一致性。

## 配置选项

### 限制处理文件数量

在 `main.py` 中修改：

```python
processor = DataProcessor(data_path, max_files=1000)  # 只处理前 1000 个文件
```

### 禁用缓存

```python
processor.parse_event_files(use_cache=False)
normalizer.process_documents(documents, use_cache=False)
inverted_index.build_index(normalized_docs, use_cache=False)
```

### 修改缓存目录

```python
processor = DataProcessor(data_path, cache_dir="custom_cache")
normalizer = TextNormalizer(cache_dir="custom_cache")
inverted_index = InvertedIndex(cache_dir="custom_cache")
```

## 性能优化

- **批量处理**: 文档按批次处理，显示进度
- **缓存机制**: 避免重复计算，大幅提升速度
- **增量显示**: 处理过程实时显示进度
- **内存优化**: 使用生成器和字典优化内存使用

## 故障排除

### NLTK 数据下载失败

如果 NLTK 数据下载失败，系统会自动降级到内置处理器，不影响基本功能。

手动下载 NLTK 数据：

```bash
python nltk_downloader.py
```

### 缓存文件损坏

删除缓存目录重新生成：

```powershell
# Windows PowerShell
Remove-Item -Recurse -Force Meetup\cache
```

```bash
# Linux/macOS
rm -rf Meetup/cache
```

### 内存不足

减少处理文件数量：

```python
processor = DataProcessor(data_path, max_files=100)
```

## 实验结果示例

```
=== 信息检索实验系统启动 ===

步骤1: 数据解析
------------------------------
找到 8742 个XML文件
限制解析前 10000 个文件
解析进度: 100/8742 200/8742 ... 8742/8742
数据解析完成: 成功 8740, 错误 2, 耗时 98.34秒

步骤2: 文本规范化
------------------------------
开始文本规范化处理 (使用NLTK处理器)...
文本规范化处理完成，耗时 45.56秒

步骤3: 构建倒排索引
------------------------------
开始构建倒排索引，共 8740 个文档...
倒排索引构建完成，共 15234 个词项，耗时 12.34秒

步骤4: 布尔检索测试
------------------------------
查询: 'party' -> 1234 结果, 耗时: 0.0012秒
查询: 'meeting and group' -> 456 结果, 耗时: 0.0008秒
查询: 'tech or computer' -> 789 结果, 耗时: 0.0010秒
查询: 'not business' -> 7234 结果, 耗时: 0.0015秒

步骤5: 向量空间模型检索
------------------------------
查询: 'technology conference'
找到 3 个相关文档 (耗时: 1.2345秒)
  1. [相似度: 0.8234] Tech Meetup Conference 2023
  2. [相似度: 0.7456] Software Engineering Summit
  3. [相似度: 0.6789] AI and Machine Learning Workshop

查询: 'business meeting'
找到 3 个相关文档 (耗时: 1.1234秒)
  1. [相似度: 0.7890] Professional Business Network
  2. [相似度: 0.7123] Startup Entrepreneurs Meetup
  3. [相似度: 0.6543] Corporate Networking Event

查询: 'social event party'
找到 3 个相关文档 (耗时: 1.2567秒)
  1. [相似度: 0.8012] Weekend Social Gathering
  2. [相似度: 0.7345] Community Party Event
  3. [相似度: 0.6789] Fun Social Meetup

步骤6: 保存索引
------------------------------
索引已保存到 Meetup/inverted_index.json，耗时 5.67秒

============================================================
实验完成!
============================================================
总运行时间: 165.34 秒
处理文档数: 8740
索引词项数: 15234

各阶段耗时统计:
  数据解析: 98.34 秒
  文本规范化: 45.56 秒
  索引构建: 12.34 秒
  索引保存: 5.67 秒
  平均布尔检索: 0.0011 秒
  平均向量检索: 1.2049 秒
  TF-IDF构建: 23.45 秒

缓存目录: Meetup/cache
缓存文件数量: 3
```

## 技术细节

### 倒排索引结构

```json
{
  "term": {
    "doc_id": [position1, position2, ...]
  }
}
```

### TF-IDF 计算

- **TF (词频)**: `count / total_terms_in_doc`
- **IDF (逆文档频率)**: `log(total_docs / docs_containing_term)`
- **TF-IDF**: `TF × IDF`

### 余弦相似度

```
similarity = (A · B) / (||A|| × ||B||)
```

其中 A 和 B 是查询和文档的 TF-IDF 向量。

## 依赖说明

- **nltk**: 自然语言处理工具包（可选，系统会自动降级）
- **numpy**: 数值计算（向量检索需要）
- **scikit-learn**: 机器学习库（可选，使用简化版 TF-IDF）

## 注意事项

1. 首次运行需要较长时间构建索引
2. 后续运行会使用缓存，速度显著提升
3. 缓存文件可能占用较多磁盘空间
4. 向量检索比布尔检索耗时更多，但结果更相关
5. 可以通过 `max_files` 参数限制处理文件数量进行快速测试

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

如有问题，请提交 Issue。

---

**注意**: 本项目为信息检索课程实验项目，仅用于学习和研究目的。
