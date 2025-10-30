import os
import time

def main():
    """主函数"""
    print("=== 信息检索实验系统启动 ===\n")
    
    # 数据路径和缓存路径
    data_path = "Meetup/All_Unpack"
    cache_dir = "Meetup/cache"  # 修改缓存目录到Meetup文件夹下
    
    if not os.path.exists(data_path):
        print(f"错误: 数据路径 '{data_path}' 不存在!")
        print("请确保Meetup/All_Unpack目录存在并包含XML文件")
        return
    
    # 首先尝试下载 NLTK 数据（静默）
    try:
        from nltk_downloader import NLTKDataDownloader
        NLTKDataDownloader.download_required_data()
    except Exception as e:
        print(f"NLTK 数据检查跳过: {e}")
    
    # 导入模块
    from data_processor import DataProcessor
    from text_normalizer import TextNormalizer
    from inverted_index import InvertedIndex
    from boolean_retrieval import BooleanRetrieval
    
    # 尝试导入向量检索
    try:
        from vector_retrieval import VectorRetrieval
        VECTOR_AVAILABLE = True
    except ImportError:
        print("[INFO] 向量检索不可用，跳过此功能")
        VECTOR_AVAILABLE = False
    
    total_start_time = time.time()
    
    # 1. 数据解析
    print("\n步骤1: 数据解析")
    print("-" * 30)
    processor = DataProcessor(data_path, max_files=10000, cache_dir=cache_dir)
    processor.parse_event_files(use_cache=True)  # 启用缓存
    documents = processor.get_documents()
    
    if not documents:
        print("错误: 没有成功解析任何文档!")
        return
    
    print(f"成功解析 {len(documents)} 个文档")
    
    # 2. 文本规范化
    print("\n步骤2: 文本规范化")
    print("-" * 30)
    normalizer = TextNormalizer(cache_dir=cache_dir)
    normalized_docs = normalizer.process_documents(documents, use_cache=True)  # 启用缓存
    
    # 3. 构建倒排索引
    print("\n步骤3: 构建倒排索引")
    print("-" * 30)
    inverted_index = InvertedIndex(cache_dir=cache_dir)
    inverted_index.build_index(normalized_docs, use_cache=True)  # 启用缓存
    
    # 4. 布尔检索测试
    print("\n步骤4: 布尔检索测试")
    print("-" * 30)
    boolean_searcher = BooleanRetrieval(inverted_index)
    
    test_queries = [
        "party",
        "meeting and group", 
        "tech or computer",
        "not business"
    ]
    
    for query in test_queries:
        start_time = time.time()
        results = boolean_searcher.search(query)
        end_time = time.time()
        search_time = end_time - start_time
        print(f"查询: '{query}' -> {len(results)} 结果, 耗时: {search_time:.4f}秒")
    
    # 5. 向量空间模型检索（如果可用）
    vector_searcher = None
    if VECTOR_AVAILABLE:
        print("\n步骤5: 向量空间模型检索")
        print("-" * 30)
        vector_searcher = VectorRetrieval(documents)
        
        vector_queries = [
            "technology conference",
            "business meeting", 
            "social event party"
        ]
        
        for query in vector_queries:
            print(f"\n查询: '{query}'")
            try:
                # 确保正确接收返回值
                results, search_time = vector_searcher.search(query, top_k=3)
                print(f"找到 {len(results)} 个相关文档 (耗时: {search_time:.4f}秒)")
                
                for i, (doc_id, score) in enumerate(results[:3], 1):
                    doc_name = documents[doc_id]['name']
                    if len(doc_name) > 50:
                        doc_name = doc_name[:50] + "..."
                    print(f"  {i}. [相似度: {score:.4f}] {doc_name}")
            except ValueError as e:
                print(f"向量检索错误: {e}")
                print("跳过此查询...")
    else:
        print("\n步骤5: 向量空间模型检索")
        print("-" * 30)
        print("向量检索功能不可用，跳过此步骤")
    
    # 6. 保存索引
    print("\n步骤6: 保存索引")
    print("-" * 30)
    inverted_index.save_index("Meetup/inverted_index.json")  # 索引也保存到Meetup目录
    
    # 统计信息
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)
    print(f"总运行时间: {total_time:.2f} 秒")
    print(f"处理文档数: {len(documents)}")
    print(f"索引词项数: {len(inverted_index.index)}")
    
    # 各阶段耗时
    print("\n各阶段耗时统计:")
    
    # 数据解析时间
    processor_times = processor.get_processing_times()
    if 'data_parsing' in processor_times:
        print(f"  数据解析: {processor_times['data_parsing']:.2f} 秒")
    
    # 文本规范化时间
    normalizer_times = normalizer.get_processing_times()
    if 'text_normalization' in normalizer_times:
        print(f"  文本规范化: {normalizer_times['text_normalization']:.2f} 秒")
    
    # 索引构建时间
    index_times = inverted_index.get_processing_times()
    if 'index_building' in index_times:
        print(f"  索引构建: {index_times['index_building']:.2f} 秒")
    
    # 索引保存时间
    if 'index_saving' in index_times:
        print(f"  索引保存: {index_times['index_saving']:.2f} 秒")
    
    # 布尔检索时间统计
    boolean_times = boolean_searcher.get_processing_times()
    if 'search_times' in boolean_times and boolean_times['search_times']:
        avg_bool_time = sum(boolean_times['search_times']) / len(boolean_times['search_times'])
        print(f"  平均布尔检索: {avg_bool_time:.4f} 秒")
    
    # 向量检索时间统计
    if vector_searcher and hasattr(vector_searcher, 'get_processing_times'):
        vector_times = vector_searcher.get_processing_times()
        if 'search_times' in vector_times and vector_times['search_times']:
            avg_vector_time = sum(vector_times['search_times']) / len(vector_times['search_times'])
            print(f"  平均向量检索: {avg_vector_time:.4f} 秒")
        
        if 'tfidf_building' in vector_times:
            print(f"  TF-IDF构建: {vector_times['tfidf_building']:.2f} 秒")
    
    # 缓存使用情况
    print(f"\n缓存目录: {cache_dir}")
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        print(f"缓存文件数量: {len(cache_files)}")

if __name__ == "__main__":
    main()