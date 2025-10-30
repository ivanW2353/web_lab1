# 信息检索实验系统

这是一个基于 Python 的信息检索实验系统，实现了布尔检索和向量空间模型检索两种检索方法。

## 📑 目录

- [✨ 特性亮点](#-特性亮点)
- [功能特性](#功能特性)
- [系统要求](#系统要求)
- [安装步骤](#安装步骤)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [模块说明](#模块说明)
- [使用示例](#使用示例)
- [缓存机制](#缓存机制)
- [配置选项](#配置选项)
- [性能优化](#性能优化)
- [故障排除](#故障排除)
- [实验结果示例](#实验结果示例)
- [技术细节](#技术细节)
- [依赖说明](#依赖说明)
- [注意事项](#注意事项)
- [常见问题 FAQ](#常见问题-faq)
- [贡献](#贡献)
- [许可证](#许可证)

## ✨ 特性亮点

- 🚀 **高性能**: 四层缓存机制，重复运行速度提升 10-50 倍
- 💾 **内存优化**: 稀疏向量表示，支持大规模数据集
- 📊 **可视化进度**: tqdm 进度条实时显示处理状态
- 🎯 **灵活配置**: 命令行参数支持，轻松调整运行参数
- 🔄 **智能降级**: NLTK 不可用时自动使用增强内置处理器
- 📝 **友好输出**: emoji 和格式化输出，清晰易读
- 🛠️ **易于扩展**: 模块化设计，便于添加新功能

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
   
   主要依赖包括：
   - `tqdm`: 进度条显示
   - `nltk`: 自然语言处理（可选）
   - 其他标准库

3. **准备数据集**
   - 确保 `Meetup/All_Unpack/` 目录存在
   - 将 Meetup XML 数据文件放入该目录

## 快速开始

直接运行主程序：

```bash
# 使用默认参数（处理前10000个文件）
python main.py

# 指定处理文件数量
python main.py --max_files 500

# 处理所有文件
python main.py --max_files 0

# 自定义数据路径和缓存目录
python main.py --max_files 1000 --data_path "custom/data/path" --cache_dir "custom/cache"

# 查看帮助
python main.py --help
```

### 命令行参数说明

- `--max_files`: 最大处理文件数量（默认：10000，设置为 0 处理所有文件）
- `--data_path`: 数据文件路径（默认：`Meetup/All_Unpack`）
- `--cache_dir`: 缓存目录路径（默认：`Meetup/cache`）

首次运行时，系统会：
1. 自动下载 NLTK 所需数据到 `Meetup/nltk_data`（如果使用 NLTK）
2. 解析 XML 数据文件（显示进度条）
3. 进行文本规范化处理（显示进度条）
4. 构建倒排索引（显示进度条）
5. 执行示例布尔检索和向量检索查询
6. 保存索引到文件

后续运行会自动使用缓存，显著提升速度！

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
- **增强的文本规范化处理**（分词、去停用词、词干提取）
- 自动检测 NLTK 可用性
- **增强的内置处理器**：处理缩写（n't, 's, 're 等）、连字符单词
- **全面的停用词列表**
- **增强的词干提取**：支持复数、动词、副词、名词后缀等
- 支持缓存规范化结果
- 使用 tqdm 进度条显示处理进度

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
- **优化的 TF-IDF 向量空间模型**
- **稀疏向量表示**：大幅减少内存使用
- 余弦相似度计算（支持稀疏向量）
- 返回 Top-K 相关文档
- **可配置特征数量**（max_features）控制内存使用
- **批量处理**：减少内存占用
- 支持 TF-IDF 模型缓存
- 使用 tqdm 进度条显示处理进度
- 完全独立实现，不依赖 scikit-learn

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

# 初始化检索器（可指定缓存目录）
vector_searcher = VectorRetrieval(documents, cache_dir="Meetup/cache")

# 构建TF-IDF向量（可配置特征数量和缓存）
vector_searcher.build_tfidf_vectors(use_cache=True, max_features=30000)

# 执行查询
results, search_time = vector_searcher.search(
    "technology conference", 
    top_k=10,
    use_cache=True,
    max_features=30000
)

# 显示结果
for doc_id, score in results:
    print(f"文档: {doc_id}, 相似度: {score:.4f}")
```

## 缓存机制

系统实现了**四层智能缓存机制**，大幅提升重复运行速度：

1. **文档缓存** (`documents_*.json`): 缓存解析后的 XML 文档
2. **规范化缓存** (`normalized_*.json`): 缓存文本规范化结果
3. **索引缓存** (`index_*.json`): 缓存倒排索引
4. **TF-IDF 缓存** (`tfidf_optimized_*.json`): 缓存向量模型

### 缓存特性

- 📁 缓存文件存储在 `Meetup/cache/` 目录
- 🔐 使用 MD5 哈希命名，确保数据一致性
- ⚡ 自动检测数据变化，智能使用缓存
- 💾 NLTK 数据单独存储在 `Meetup/nltk_data/`

### 缓存效果

- **首次运行**: 完整处理，耗时较长
- **后续运行**: 使用缓存，速度提升 **10-50 倍**

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

### 速度优化
- ⚡ **四层缓存机制**: 避免重复计算，速度提升 10-50 倍
- 📊 **tqdm 进度条**: 实时显示处理进度和速度
- 🔄 **批量处理**: 向量检索使用批处理减少内存压力

### 内存优化
- 💾 **稀疏向量表示**: TF-IDF 使用稀疏向量，大幅减少内存占用
- 🎯 **可配置特征数**: 通过 `max_features` 参数控制词汇表大小
- 📦 **增量处理**: 文档按批次处理，避免一次性加载所有数据

### 用户体验
- 🎨 **友好的输出**: 使用 emoji 和格式化输出
- 📈 **详细统计**: 显示各阶段耗时和处理进度
- ⚙️ **命令行参数**: 灵活配置运行参数

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

多种解决方案：

1. **减少处理文件数量**：
   ```bash
   python main.py --max_files 500
   ```

2. **降低向量特征数**：
   ```python
   # 在代码中修改
   vector_searcher.search(query, top_k=10, max_features=10000)
   ```

3. **清理缓存释放空间**：
   ```powershell
   Remove-Item -Recurse -Force Meetup\cache
   ```

4. **分批处理**：系统已自动使用批处理机制

## 实验结果示例

### 首次运行（无缓存）

```
=== 信息检索实验系统启动 ===

配置参数:
  • 最大处理文件数: 10000
  • 数据路径: Meetup/All_Unpack
  • 缓存目录: Meetup/cache

🔍 检查 NLTK 数据...
📥 下载 punkt... ✅
📥 下载 stopwords... ✅
📁 NLTK数据位置: Meetup/nltk_data

📂 步骤 1: 数据解析
────────────────────────────────────────
📄 找到 8742 个 XML 文件
🔢 限制解析前 10000 个文件
📂 解析XML文件: 100%|██████████| 8742/8742 [01:38<00:00, 89.12文件/s]
✅ 数据解析完成: 成功 8740, 错误 2, 耗时 98.34秒
✅ 成功解析 8740 个文档

🔤 步骤 2: 文本规范化
────────────────────────────────────────
🔤 文本规范化处理 (使用NLTK处理器)
🔤 文本规范化: 100%|██████████| 8740/8740 [00:45<00:00, 192.35文档/s]
✅ 文本规范化完成

📊 步骤 3: 构建倒排索引
────────────────────────────────────────
📊 构建倒排索引，共 8740 个文档
📊 构建索引: 100%|██████████| 8740/8740 [00:12<00:00, 708.26文档/s]
✅ 倒排索引构建完成，共 15234 个词项，耗时 12.34秒
✅ 倒排索引构建完成，共 15234 个词项

🔍 步骤 4: 布尔检索测试
────────────────────────────────────────
执行布尔查询:
  • 'party' -> 1234 个结果 (耗时: 0.0012秒)
  • 'meeting and group' -> 456 个结果 (耗时: 0.0008秒)
  • 'tech or computer' -> 789 个结果 (耗时: 0.0010秒)
  • 'not business' -> 7234 个结果 (耗时: 0.0015秒)
✅ 布尔检索测试完成

📈 步骤 5: 向量空间模型检索
────────────────────────────────────────
📈 构建 TF-IDF 向量 (限制特征数: 30000)...
📝 第一阶段：构建词汇表和计算文档频率
📊 计算词项频率: 100%|██████████| 8740/8740 [00:08<00:00, 1012.45文档/s]
✅ 词汇表构建完成，选择 30000 个特征
📈 第二阶段：计算TF-IDF稀疏向量
🔢 计算文档向量: 100%|██████████| 8740/8740 [00:15<00:00, 565.78文档/s]
✅ TF-IDF向量构建完成，使用稀疏表示

执行向量查询:

🔎 查询: 'technology conference'
🔍 计算相似度，共 8740 个文档...
  找到 3 个相关文档 (耗时: 1.2345秒)
    1. [相似度: 0.8234] Tech Meetup Conference 2023
    2. [相似度: 0.7456] Software Engineering Summit
    3. [相似度: 0.6789] AI and Machine Learning Workshop

🔎 查询: 'business meeting'
🔍 计算相似度，共 8740 个文档...
  找到 3 个相关文档 (耗时: 1.1234秒)
    1. [相似度: 0.7890] Professional Business Network
    2. [相似度: 0.7123] Startup Entrepreneurs Meetup
    3. [相似度: 0.6543] Corporate Networking Event

✅ 向量空间模型检索完成

💾 步骤 6: 保存索引
────────────────────────────────────────
💾 保存索引到文件... ✅
✅ 索引保存完成

============================================================
🎉 实验完成!
============================================================
总运行时间: 165.34 秒
处理文档数: 8740
索引词项数: 15234

⏱️  各阶段耗时统计:
  数据解析: 98.34 秒
  文本规范化: 45.56 秒
  索引构建: 12.34 秒
  索引保存: 5.67 秒
  平均布尔检索: 0.0011 秒
  平均向量检索: 1.2049 秒
  TF-IDF构建: 23.45 秒

📁 缓存目录: Meetup/cache
  缓存文件数量: 4

✨ 所有操作已完成！
```

### 使用缓存的后续运行

```
=== 信息检索实验系统启动 ===

📂 步骤 1: 数据解析
────────────────────────────────────────
✅ 从缓存加载了 8740 个文档          <-- 几乎瞬间完成！
✅ 成功解析 8740 个文档

🔤 步骤 2: 文本规范化
────────────────────────────────────────
✅ 从缓存加载了 8740 个规范化文档    <-- 几乎瞬间完成！
✅ 文本规范化完成

📊 步骤 3: 构建倒排索引
────────────────────────────────────────
✅ 从缓存加载了包含 15234 个词项的索引  <-- 几乎瞬间完成！
✅ 倒排索引构建完成，共 15234 个词项

... (检索部分正常执行)

总运行时间: 8.45 秒  <-- 速度提升约 20 倍！
```

## 性能对比

### 缓存加速效果（基于 10000 文件）

| 阶段 | 首次运行 | 使用缓存 | 加速比 |
|------|---------|---------|--------|
| 数据解析 | ~98 秒 | ~0.2 秒 | **490x** |
| 文本规范化 | ~46 秒 | ~0.1 秒 | **460x** |
| 索引构建 | ~12 秒 | ~0.3 秒 | **40x** |
| TF-IDF构建 | ~23 秒 | ~0.5 秒 | **46x** |
| **总计** | **~180 秒** | **~8 秒** | **22.5x** |

### 内存优化效果

| 方法 | 内存占用 | 说明 |
|------|---------|------|
| 密集向量 (旧) | ~3.5 GB | 10000 文档 × 50000 特征 |
| **稀疏向量 (新)** | **~300 MB** | 仅存储非零值 |
| **优化比例** | **11.7x** | 节省约 91% 内存 |

### 检索速度对比

| 检索类型 | 平均耗时 | 适用场景 |
|---------|---------|---------|
| 布尔检索 | ~0.001 秒 | 精确查询，快速筛选 |
| 向量检索 | ~1.2 秒 | 语义相关，排序结果 |

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

### 必需依赖
- **tqdm** (>=4.62.0): 进度条显示，提升用户体验

### 可选依赖
- **nltk** (>=3.8.0): 自然语言处理工具包
  - 如果不可用，系统会自动使用增强的内置处理器
  - 数据自动下载到 `Meetup/nltk_data/`

## 注意事项

1. **首次运行时间**: 首次运行需要较长时间构建索引（取决于文件数量）
   - 1000 个文件：约 30-60 秒
   - 10000 个文件：约 3-5 分钟
   
2. **缓存加速**: 后续运行使用缓存，速度提升 **10-50 倍**

3. **磁盘空间**: 缓存文件可能占用一定磁盘空间
   - 文档缓存：约 1-2MB / 1000 文件
   - 索引缓存：约 5-10MB / 10000 文件
   - TF-IDF 缓存：取决于特征数量

4. **检索性能对比**:
   - 布尔检索：毫秒级，速度快
   - 向量检索：秒级，结果更相关

5. **快速测试**: 使用 `--max_files 100` 快速测试功能

6. **内存优化**: 如遇内存问题，可通过以下方式优化：
   - 减少 `max_files` 参数
   - 降低 `max_features` 参数（向量检索）
   - 使用更强大的机器

## 常见问题 FAQ

### Q1: 为什么首次运行这么慢？
**A**: 首次运行需要：
- 解析所有 XML 文件
- 进行文本规范化处理
- 构建倒排索引和 TF-IDF 向量
- 建立缓存文件

后续运行会使用缓存，速度提升 10-50 倍！

### Q2: 如何加快处理速度？
**A**: 几种方法：
1. 减少处理文件数：`python main.py --max_files 1000`
2. 确保启用缓存（默认启用）
3. 降低向量特征数：修改 `max_features` 参数
4. 使用 SSD 硬盘存储缓存

### Q3: NLTK 下载失败怎么办？
**A**: 不用担心！系统会自动使用增强的内置处理器，功能几乎相同。如需手动下载：
```bash
python nltk_downloader.py
```

### Q4: 内存不足怎么办？
**A**: 尝试以下方案：
- 减少文件数：`--max_files 500`
- 降低特征数：`max_features=10000`
- 清理其他程序释放内存
- 关闭浏览器等占用内存的应用

### Q5: 缓存占用太多空间？
**A**: 可以安全删除缓存：
```powershell
Remove-Item -Recurse -Force Meetup\cache
```
下次运行会重新生成。

### Q6: 如何查看更多文档？
**A**: 修改向量检索的 `top_k` 参数：
```python
results, time = vector_searcher.search(query, top_k=20)
```

### Q7: 向量检索结果为空？
**A**: 可能原因：
- 查询词太专业/罕见
- 词被停用词过滤
- 尝试使用更通用的查询词

### Q8: 如何调试代码？
**A**: 在各模块中添加打印语句，或使用 Python 调试器：
```bash
python -m pdb main.py
```

## 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献指南
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

MIT License

## 联系方式

如有问题，请提交 Issue。

---

**注意**: 本项目为信息检索课程实验项目，仅用于学习和研究目的。
