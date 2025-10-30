import os
import argparse
from nltk_downloader import NLTKDataDownloader
from data_processor import DataProcessor
from vector_retrieval import VectorRetrieval


def ensure_documents(data_path: str, cache_dir: str, max_files: int, use_cache_only: bool):
    dp = DataProcessor(data_path=data_path, max_files=max_files, cache_dir=cache_dir)
    # 尝试从缓存加载
    if dp.load_documents_from_cache():
        return dp.get_documents()

    if use_cache_only:
        print('❌ 未找到文档缓存，且启用了 --use_cache_only')
        return {}

    print('⚠️  未找到缓存，尝试解析数据以供向量检索使用...')
    dp.parse_event_files(use_cache=True)
    return dp.get_documents()


def main():
    parser = argparse.ArgumentParser(description='向量检索（基于优化 TF-IDF）')
    parser.add_argument('--data_path', type=str, default='Meetup/All_Unpack', help='数据路径（用于加载/生成缓存的文档）')
    parser.add_argument('--cache_dir', type=str, default='Meetup/cache', help='缓存目录')
    parser.add_argument('--max_files', type=int, default=10000, help='最大处理文件数量（与构建时保持一致，0 为全部）')
    parser.add_argument('--top_k', type=int, default=10, help='返回前K个文档')
    parser.add_argument('--max_features', type=int, default=30000, help='TF-IDF 词汇表大小上限')
    parser.add_argument('--query', type=str, default=None, help='查询语句（不区分大小写）')
    parser.add_argument('--use_cache_only', action='store_true', help='仅使用缓存的文档，不重新解析')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"❌ 数据路径不存在: {args.data_path}")
        return 1

    # 准备 NLTK（可选）
    try:
        NLTKDataDownloader.download_required_data()
    except Exception:
        pass

    max_files = args.max_files if args.max_files > 0 else None
    documents = ensure_documents(args.data_path, args.cache_dir, max_files, args.use_cache_only)
    if not documents:
        print('❌ 无法加载文档，退出')
        return 2

    vr = VectorRetrieval(documents, cache_dir=args.cache_dir)

    def do_query(q: str):
        results, cost = vr.search(q, top_k=args.top_k, use_cache=True, max_features=args.max_features)
        print(f"🔎 查询: {q}")
        print(f"⏱️  耗时: {cost:.4f}s  |  返回: {len(results)} 条")
        for i, (doc_id, score) in enumerate(results, 1):
            name = documents[doc_id]['name']
            if len(name) > 50:
                name = name[:50] + '...'
            print(f"  {i}. [相似度: {score:.4f}] [{doc_id}] {name}")

    if args.query:
        do_query(args.query)
        return 0

    print('=== 向量检索 (交互模式) ===')
    print("输入查询，或输入 'exit' 退出")
    while True:
        try:
            q = input('query> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n再见!')
            break
        if not q or q.lower() in {'exit', 'quit', ':q'}:
            break
        do_query(q)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
