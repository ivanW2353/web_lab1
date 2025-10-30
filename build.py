import os
import time
import argparse

from nltk_downloader import NLTKDataDownloader
from data_processor import DataProcessor
from text_normalizer import TextNormalizer
from inverted_index import InvertedIndex


def main():
    parser = argparse.ArgumentParser(description='构建阶段：解析->规范化->倒排索引->保存')
    parser.add_argument('--data_path', type=str, default='Meetup/All_Unpack', help='数据路径')
    parser.add_argument('--cache_dir', type=str, default='Meetup/cache', help='缓存目录')
    parser.add_argument('--max_files', type=int, default=10000, help='最大处理文件数量（0 表示全部）')
    parser.add_argument('--index_out', type=str, default='Meetup/inverted_index.json', help='索引输出文件')
    args = parser.parse_args()

    data_path = args.data_path
    cache_dir = args.cache_dir
    max_files = None if args.max_files is None else (args.max_files if args.max_files > 0 else None)

    print('=== 构建阶段开始 ===')
    print(f'数据路径: {data_path}')
    print(f'缓存目录: {cache_dir}')
    print(f'最大文件数: {args.max_files}')

    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        return 1

    # 准备 NLTK 数据（可选）
    try:
        print('🔍 检查 NLTK 数据...')
        NLTKDataDownloader.download_required_data()
    except Exception as e:
        print(f'⚠️  跳过 NLTK 数据检查: {e}')

    total_start = time.time()

    # 1) 解析
    print('\n📂 解析数据...')
    processor = DataProcessor(data_path=data_path, max_files=args.max_files, cache_dir=cache_dir)
    processor.parse_event_files(use_cache=True)
    documents = processor.get_documents()
    if not documents:
        print('❌ 没有解析到任何文档')
        return 2
    print(f'✅ 文档数: {len(documents)}')

    # 2) 规范化
    print('\n🔤 文本规范化...')
    normalizer = TextNormalizer(cache_dir=cache_dir)
    normalized_docs = normalizer.process_documents(documents, use_cache=True)

    # 3) 索引
    print('\n📊 构建倒排索引...')
    index = InvertedIndex(cache_dir=cache_dir)
    index.build_index(normalized_docs, use_cache=True)

    # 4) 保存索引
    print('\n💾 保存索引...')
    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)
    index.save_index(args.index_out)

    total_cost = time.time() - total_start
    print('\n=== 构建阶段完成 ===')
    print(f'总耗时: {total_cost:.2f}s')
    print(f'索引词项数: {len(index.index)}')
    print(f'索引文件: {args.index_out}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
