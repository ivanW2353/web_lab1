import os
import time
import argparse
from tqdm import tqdm

def main():
    """主函数"""
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='信息检索实验系统')
    parser.add_argument('--max_files', type=int, default=10000, 
                       help='最大处理文件数量 (默认: 10000)')
    parser.add_argument('--data_path', type=str, default="Meetup/All_Unpack",
                       help='数据文件路径 (默认: Meetup/All_Unpack)')
    parser.add_argument('--cache_dir', type=str, default="Meetup/cache",
                       help='缓存目录 (默认: Meetup/cache)')
    
    args = parser.parse_args()
    
    print("=== 信息检索实验系统启动 ===\n")
    
    # 使用参数
    data_path = args.data_path
    cache_dir = args.cache_dir
    max_files = args.max_files
    
    print(f"配置参数:")
    print(f"  • 最大处理文件数: {max_files}")
    print(f"  • 数据路径: {data_path}")
    print(f"  • 缓存目录: {cache_dir}\n")
    
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据路径 '{data_path}' 不存在!")
        print("请确保 Meetup/All_Unpack 目录存在并包含 XML 文件")
        return
    
    # 首先尝试下载 NLTK 数据
    try:
        from nltk_downloader import NLTKDataDownloader
        print("🔍 检查 NLTK 数据...")
        success = NLTKDataDownloader.download_required_data()
        if success:
            print("✅ NLTK数据检查完成")
        else:
            print("⚠️  NLTK数据部分缺失，使用内置处理器")
    except Exception as e:
        print(f"⚠️  NLTK数据检查跳过: {e}")
    
    # 导入模块
    from data_processor import DataProcessor
    from text_normalizer import TextNormalizer
    from inverted_index import InvertedIndex
    from boolean_retrieval import BooleanRetrieval
    
    # 尝试导入向量检索
    try:
        from vector_retrieval import VectorRetrieval
        VECTOR_AVAILABLE = True
    except ImportError as e:
        VECTOR_AVAILABLE = False
        print("ℹ️  向量检索功能不可用，跳过此功能")
    
    total_start_time = time.time()
    
    # 1. 数据解析
    print("\n📂 步骤 1: 数据解析")
    print("─" * 40)
    
    processor = DataProcessor(data_path, max_files=max_files, cache_dir=cache_dir)
    processor.parse_event_files(use_cache=True)
    documents = processor.get_documents()
    
    if not documents:
        print("❌ 错误: 没有成功解析任何文档!")
        return
    
    print(f"✅ 成功解析 {len(documents)} 个文档")
    
    # 2. 文本规范化
    print("\n🔤 步骤 2: 文本规范化")
    print("─" * 40)
    
    normalizer = TextNormalizer(cache_dir=cache_dir)
    normalized_docs = normalizer.process_documents(documents, use_cache=True)
    
    print("✅ 文本规范化完成")
    
    # 3. 构建倒排索引
    print("\n📊 步骤 3: 构建倒排索引")
    print("─" * 40)
    
    inverted_index = InvertedIndex(cache_dir=cache_dir)
    inverted_index.build_index(normalized_docs, use_cache=True)
    
    print(f"✅ 倒排索引构建完成，共 {len(inverted_index.index)} 个词项")
    
    # 4. 布尔检索测试
    print("\n🔍 步骤 4: 布尔检索测试")
    print("─" * 40)
    
    boolean_searcher = BooleanRetrieval(inverted_index)
    
    test_queries = [
        "party",
        "meeting and group", 
        "tech or computer",
        "not business"
    ]
    
    print("执行布尔查询:")
    for query in test_queries:
        start_time = time.time()
        results = boolean_searcher.search(query)
        end_time = time.time()
        search_time = end_time - start_time
        print(f"  • '{query}' -> {len(results)} 个结果 (耗时: {search_time:.4f}秒)")
    
    print("✅ 布尔检索测试完成")
    
    # 5. 向量空间模型检索（如果可用）
    vector_searcher = None
    if VECTOR_AVAILABLE:
        print("\n📈 步骤 5: 向量空间模型检索")
        print("─" * 40)
        
        vector_searcher = VectorRetrieval(documents, cache_dir=cache_dir)
        
        vector_queries = [
            "technology conference",
            "business meeting", 
            "social event party"
        ]
        
        print("执行向量查询:")
        for query in vector_queries:
            print(f"\n🔎 查询: '{query}'")
            
            try:
                # 使用优化的向量检索，限制特征数量
                results, search_time = vector_searcher.search(query, top_k=3, use_cache=True, max_features=30000)
                print(f"  找到 {len(results)} 个相关文档 (耗时: {search_time:.4f}秒)")
                
                for i, (doc_id, score) in enumerate(results[:3], 1):
                    doc_name = documents[doc_id]['name']
                    if len(doc_name) > 50:
                        doc_name = doc_name[:50] + "..."
                    print(f"    {i}. [相似度: {score:.4f}] {doc_name}")
                    
            except ValueError as e:
                print(f"❌ 向量检索错误: {e}")
                print("跳过此查询...")
            except MemoryError as e:
                print(f"❌ 内存不足: {e}")
                print("尝试进一步减少特征数量...")
                # 尝试使用更少的特征
                try:
                    results, search_time = vector_searcher.search(query, top_k=3, use_cache=True, max_features=10000)
                    print(f"  找到 {len(results)} 个相关文档 (耗时: {search_time:.4f}秒)")
                    
                    for i, (doc_id, score) in enumerate(results[:3], 1):
                        doc_name = documents[doc_id]['name']
                        if len(doc_name) > 50:
                            doc_name = doc_name[:50] + "..."
                        print(f"    {i}. [相似度: {score:.4f}] {doc_name}")
                except:
                    print("❌ 向量检索失败，跳过此查询")
        
        print("✅ 向量空间模型检索完成")
    else:
        print("\n📈 步骤 5: 向量空间模型检索")
        print("─" * 40)
        print("ℹ️  向量检索功能不可用，跳过此步骤")
    
    # 6. 保存索引
    print("\n💾 步骤 6: 保存索引")
    print("─" * 40)
    
    inverted_index.save_index("Meetup/inverted_index.json")
    
    print("✅ 索引保存完成")
    
    # 统计信息
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print("\n" + "=" * 60)
    print("🎉 实验完成!")
    print("=" * 60)
    
    print(f"总运行时间: {total_time:.2f} 秒")
    print(f"处理文档数: {len(documents)}")
    print(f"索引词项数: {len(inverted_index.index)}")
    
    # 各阶段耗时
    print("\n⏱️  各阶段耗时统计:")
    
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
    print(f"\n📁 缓存目录: {cache_dir}")
    
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        print(f"  缓存文件数量: {len(cache_files)}")
    
    print(f"\n✨ 所有操作已完成！")

if __name__ == "__main__":
    main()