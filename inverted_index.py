from typing import Dict, List, Set, Tuple
from collections import defaultdict
import math
import json
import time
import os
import hashlib
from tqdm import tqdm

class InvertedIndex:
    def __init__(self, cache_dir: str = "Meetup/cache"):
        self.index = defaultdict(dict)  # term -> {doc_id: [positions]}
        self.doc_lengths = {}  # 文档长度（词项数量）
        self.doc_count = 0
        self.term_freq = defaultdict(dict)  # term -> {doc_id: frequency}
        self.processing_times = {}
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, normalized_docs: Dict[str, List[str]]) -> str:
        """生成缓存键，基于规范化文档内容"""
        # 使用文档ID和规范化词项生成哈希
        content = "".join([doc_id + "".join(terms) for doc_id, terms in normalized_docs.items()])
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_file(self, normalized_docs: Dict[str, List[str]]) -> str:
        """获取缓存文件路径"""
        cache_key = self._get_cache_key(normalized_docs)
        return os.path.join(self.cache_dir, f"index_{cache_key}.json")
    
    def load_index_from_cache(self, normalized_docs: Dict[str, List[str]]) -> bool:
        """从缓存加载索引"""
        cache_file = self._get_cache_file(normalized_docs)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                self.index = defaultdict(dict, index_data['index'])
                self.doc_lengths = index_data['doc_lengths']
                self.doc_count = index_data['doc_count']
                self.term_freq = defaultdict(dict, index_data['term_freq'])
                
                print(f"✅ 从缓存加载了包含 {len(self.index)} 个词项的索引")
                return True
            except Exception as e:
                print(f"❌ 加载索引缓存失败: {e}")
        
        return False
    
    def save_index_to_cache(self, normalized_docs: Dict[str, List[str]]):
        """保存索引到缓存"""
        cache_file = self._get_cache_file(normalized_docs)
        try:
            index_data = {
                'metadata': {
                    'document_count': self.doc_count,
                    'term_count': len(self.index),
                    'timestamp': time.time()
                },
                'index': dict(self.index),
                'doc_lengths': self.doc_lengths,
                'doc_count': self.doc_count,
                'term_freq': dict(self.term_freq)
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"❌ 保存索引缓存失败: {e}")
    
    def build_index(self, normalized_docs: Dict[str, List[str]], use_cache: bool = True):
        """构建倒排索引"""
        # 尝试从缓存加载
        if use_cache and self.load_index_from_cache(normalized_docs):
            return
            
        start_time = time.time()
        self.doc_count = len(normalized_docs)
        print(f"📊 构建倒排索引，共 {self.doc_count} 个文档")
        
        # 使用 tqdm 进度条
        progress_bar = tqdm(
            normalized_docs.items(),
            desc="📊 构建索引",
            total=self.doc_count,
            unit="文档",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for doc_id, terms in progress_bar:
            self.doc_lengths[doc_id] = len(terms)
            
            # 记录词项位置和频率
            term_freq_in_doc = defaultdict(int)
            for position, term in enumerate(terms):
                if doc_id not in self.index[term]:
                    self.index[term][doc_id] = []
                self.index[term][doc_id].append(position)
                term_freq_in_doc[term] += 1
            
            # 更新词项频率
            for term, freq in term_freq_in_doc.items():
                self.term_freq[term][doc_id] = freq
        
        # 关闭进度条
        progress_bar.close()
        
        end_time = time.time()
        self.processing_times['index_building'] = end_time - start_time
        
        print(f"✅ 倒排索引构建完成，共 {len(self.index)} 个词项，耗时 {end_time - start_time:.2f}秒")
        
        # 保存到缓存
        if use_cache:
            self.save_index_to_cache(normalized_docs)
    
    def get_term_frequency(self, term: str, doc_id: str) -> int:
        """获取词项在文档中的频率"""
        return self.term_freq.get(term, {}).get(doc_id, 0)
    
    def get_document_frequency(self, term: str) -> int:
        """获取包含词项的文档数量"""
        return len(self.index.get(term, {}))
    
    def get_inverse_document_frequency(self, term: str) -> float:
        """计算逆文档频率"""
        df = self.get_document_frequency(term)
        if df == 0:
            return 0
        return math.log(self.doc_count / df)
    
    def save_index(self, filepath: str):
        """保存索引到文件"""
        start_time = time.time()
        print("💾 保存索引到文件...", end=" ")
        
        index_data = {
            'index': dict(self.index),
            'doc_lengths': self.doc_lengths,
            'doc_count': self.doc_count,
            'term_freq': dict(self.term_freq)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        end_time = time.time()
        self.processing_times['index_saving'] = end_time - start_time
        
        print("✅")
    
    def load_index(self, filepath: str):
        """从文件加载索引"""
        start_time = time.time()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        self.index = defaultdict(dict, index_data['index'])
        self.doc_lengths = index_data['doc_lengths']
        self.doc_count = index_data['doc_count']
        self.term_freq = defaultdict(dict, index_data['term_freq'])
        
        end_time = time.time()
        self.processing_times['index_loading'] = end_time - start_time
    
    def get_processing_times(self) -> Dict:
        """获取处理时间统计"""
        return self.processing_times