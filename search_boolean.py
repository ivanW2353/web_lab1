import os
import argparse
from inverted_index import InvertedIndex
from boolean_retrieval import BooleanRetrieval
from data_processor import DataProcessor


def run_query(index_path: str, query: str, documents: dict):
    if not os.path.exists(index_path):
        print(f"❌ 找不到索引文件: {index_path}")
        print("请先运行构建阶段: python build.py")
        return 2

    inv = InvertedIndex()
    inv.load_index(index_path)

    searcher = BooleanRetrieval(inv)
    results = searcher.search(query)

    print(f"🔎 查询: {query}")
    print(f"✅ 命中文档数: {len(results)}")
    if results:
        print("示例前 10 个结果:")
        for i, doc_id in enumerate(list(results)[:10], 1):
            if documents and doc_id in documents:
                name = documents[doc_id]['name']
                if len(name) > 50:
                    name = name[:50] + '...'
                print(f"  {i}. [{doc_id}] {name}")
            else:
                print(f"  {i}. {doc_id}")

    return 0


def main():
    parser = argparse.ArgumentParser(description='布尔检索')
    parser.add_argument('--index', type=str, default='Meetup/inverted_index.json', help='倒排索引文件路径')
    parser.add_argument('--query', type=str, default=None, help='查询语句（支持 and/or/not）')
    parser.add_argument('--data_path', type=str, default='Meetup/All_Unpack', help='数据路径（用于显示文档名称）')
    parser.add_argument('--cache_dir', type=str, default='Meetup/cache', help='缓存目录')
    parser.add_argument('--max_files', type=int, default=10000, help='最大文件数量（与构建时保持一致）')
    args = parser.parse_args()

    # 加载文档信息
    print("📂 加载文档信息...")
    max_files = args.max_files if args.max_files > 0 else None
    processor = DataProcessor(data_path=args.data_path, max_files=max_files, cache_dir=args.cache_dir)
    documents = {}
    if processor.load_documents_from_cache():
        documents = processor.get_documents()
        print(f"✅ 已加载 {len(documents)} 个文档信息\n")
    else:
        print("⚠️  未找到文档缓存，将只显示文档ID\n")

    if args.query:
        return run_query(args.index, args.query, documents)

    # 交互式模式
    print('=== 布尔检索 (交互模式) ===')
    print("输入查询，或输入 'exit' 退出")

    while True:
        try:
            q = input('query> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n再见!')
            break
        if not q or q.lower() in {'exit', 'quit', ':q'}:
            break
        run_query(args.index, q, documents)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
