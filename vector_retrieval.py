import time
from typing import List, Dict, Tuple
import os
import json
import hashlib
from tqdm import tqdm
import math
from collections import defaultdict

class OptimizedTFIDF:
    """优化的TF-IDF实现，使用稀疏向量减少内存使用"""
    
    def __init__(self, cache_dir: str = "Meetup/cache"):
        self.vocab = {}
        self.doc_freq = {}
        self.doc_count = 0
        self.doc_vectors = {}  # 使用稀疏表示: {doc_id: {term_idx: tfidf_value}}
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, documents: List[str]) -> str:
        """生成缓存键，基于文档内容"""
        # 使用文档数量和前几个文档的哈希来生成缓存键
        sample_content = "".join([doc[:100] for doc in documents[:10]]) + str(len(documents))
        return hashlib.md5(sample_content.encode()).hexdigest()
    
    def _get_cache_file(self, documents: List[str]) -> str:
        """获取缓存文件路径"""
        cache_key = self._get_cache_key(documents)
        return os.path.join(self.cache_dir, f"tfidf_optimized_{cache_key}.json")
    
    def load_from_cache(self, documents: List[str]) -> bool:
        """从缓存加载TF-IDF数据"""
        cache_file = self._get_cache_file(documents)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'vocab' in data and 'doc_freq' in data and 'doc_vectors' in data:
                    self.vocab = data['vocab']
                    self.doc_freq = data['doc_freq']
                    self.doc_vectors = data['doc_vectors']
                    self.doc_count = data['doc_count']
                    
                    print(f"✅ 从缓存加载TF-IDF数据成功: 词汇表大小 {len(self.vocab)}")
                    return True
            except Exception as e:
                print(f"❌ 加载TF-IDF缓存失败: {e}")
        
        return False
    
    def save_to_cache(self, documents: List[str]):
        """保存TF-IDF数据到缓存"""
        cache_file = self._get_cache_file(documents)
        try:
            data = {
                'metadata': {
                    'document_count': self.doc_count,
                    'vocab_size': len(self.vocab),
                    'timestamp': time.time()
                },
                'vocab': self.vocab,
                'doc_freq': self.doc_freq,
                'doc_vectors': self.doc_vectors,
                'doc_count': self.doc_count
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
            
        except Exception as e:
            print(f"❌ 保存TF-IDF缓存失败: {e}")
    
    def fit_transform(self, documents: List[str], use_cache: bool = True, max_features: int = 50000) -> Dict[int, Dict[int, float]]:
        """训练TF-IDF模型并转换文档，使用稀疏向量表示"""
        # 尝试从缓存加载
        if use_cache and self.load_from_cache(documents):
            return self.doc_vectors
            
        self.doc_count = len(documents)
        print(f"📈 构建 TF-IDF 向量 (限制特征数: {max_features})...")
        
        # 第一阶段：构建词汇表和计算文档频率（限制特征数）
        print("📝 第一阶段：构建词汇表和计算文档频率")
        term_doc_freq = defaultdict(int)
        
        # 使用进度条处理文档
        df_progress = tqdm(
            documents,
            desc="📊 计算词项频率",
            total=len(documents),
            unit="文档",
            ncols=100
        )
        
        for doc in df_progress:
            terms = self._tokenize(doc)
            for term in set(terms):  # 只计算每个文档中每个词项一次
                term_doc_freq[term] += 1
        
        df_progress.close()
        
        # 选择最常用的词项作为特征
        sorted_terms = sorted(term_doc_freq.items(), key=lambda x: x[1], reverse=True)
        selected_terms = [term for term, freq in sorted_terms[:max_features]]
        
        self.vocab = {term: idx for idx, term in enumerate(selected_terms)}
        self.doc_freq = {term: term_doc_freq[term] for term in selected_terms}
        
        print(f"✅ 词汇表构建完成，选择 {len(self.vocab)} 个特征")
        
        # 第二阶段：计算TF-IDF向量（稀疏表示）
        print("📈 第二阶段：计算TF-IDF稀疏向量")
        vectors = {}
        
        vector_progress = tqdm(
            enumerate(documents),
            desc="🔢 计算文档向量",
            total=len(documents),
            unit="文档",
            ncols=100
        )
        
        for doc_id, doc in vector_progress:
            terms = self._tokenize(doc)
            term_count = len(terms)
            
            if term_count == 0:
                vectors[doc_id] = {}
                continue
            
            # 计算词频
            tf_dict = defaultdict(int)
            for term in terms:
                if term in self.vocab:  # 只考虑在词汇表中的词项
                    tf_dict[term] += 1
            
            # 计算TF-IDF（稀疏表示）
            sparse_vector = {}
            for term, count in tf_dict.items():
                if term in self.vocab:
                    idx = self.vocab[term]
                    tf = count / term_count
                    idf = math.log(self.doc_count / (self.doc_freq[term] + 1))
                    sparse_vector[idx] = tf * idf
            
            vectors[doc_id] = sparse_vector
        
        vector_progress.close()
        
        self.doc_vectors = vectors
        
        # 保存到缓存
        if use_cache:
            self.save_to_cache(documents)
        
        print(f"✅ TF-IDF向量构建完成，使用稀疏表示")
        return vectors
    
    def transform(self, texts: List[str]) -> List[Dict[int, float]]:
        """转换新文本为TF-IDF稀疏向量"""
        vectors = []
        for text in texts:
            terms = self._tokenize(text)
            term_count = len(terms)
            
            if term_count == 0:
                vectors.append({})
                continue
            
            tf_dict = defaultdict(int)
            for term in terms:
                if term in self.vocab:
                    tf_dict[term] += 1
            
            sparse_vector = {}
            for term, count in tf_dict.items():
                if term in self.vocab:
                    idx = self.vocab[term]
                    tf = count / term_count
                    idf = math.log(self.doc_count / (self.doc_freq[term] + 1))
                    sparse_vector[idx] = tf * idf
            
            vectors.append(sparse_vector)
        
        return vectors
    
    def _tokenize(self, text: str) -> List[str]:
        """优化的分词函数"""
        if not text:
            return []
        
        # 转换为小写
        text = text.lower()
        
        # 使用简单的分词，移除非字母字符
        words = []
        current_word = []
        
        for char in text:
            if char.isalpha():
                current_word.append(char)
            elif current_word:
                word = ''.join(current_word)
                if len(word) > 2:  # 只保留长度大于2的词
                    words.append(word)
                current_word = []
        
        if current_word:
            word = ''.join(current_word)
            if len(word) > 2:
                words.append(word)
        
        return words

def sparse_cosine_similarity(vec1: Dict[int, float], vec2: Dict[int, float]) -> float:
    """计算两个稀疏向量的余弦相似度"""
    if not vec1 or not vec2:
        return 0.0
    
    # 计算点积
    dot_product = 0.0
    for idx, val1 in vec1.items():
        if idx in vec2:
            dot_product += val1 * vec2[idx]
    
    # 计算向量模长
    norm1 = math.sqrt(sum(val * val for val in vec1.values()))
    norm2 = math.sqrt(sum(val * val for val in vec2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

class VectorRetrieval:
    def __init__(self, documents: Dict[str, Dict], cache_dir: str = "Meetup/cache"):
        self.documents = documents
        self.tfidf = OptimizedTFIDF(cache_dir=cache_dir)
        self.processing_times: Dict[str, float] = {}
        self.cache_dir = cache_dir
        self.doc_ids = []  # 文档ID列表，索引对应doc_vectors中的索引
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
    
    def build_tfidf_vectors(self, use_cache: bool = True, max_features: int = 50000):
        """构建TF-IDF向量"""
        start_time = time.time()
        
        # 准备文档文本和ID映射
        doc_texts = []
        self.doc_ids = []
        
        # 使用稳定的文档顺序，确保缓存键与结果可复现
        for doc_id in sorted(self.documents.keys()):
            doc_texts.append(self.documents[doc_id]['content'])
            self.doc_ids.append(doc_id)
        
        # 使用优化的TF-IDF
        self.doc_vectors = self.tfidf.fit_transform(doc_texts, use_cache=use_cache, max_features=max_features)
        
        end_time = time.time()
        self.processing_times['tfidf_building'] = end_time - start_time
        
        print(f"✅ TF-IDF 向量构建完成，词汇表大小: {len(self.tfidf.vocab)}，耗时 {end_time - start_time:.2f}秒")
    
    def search(self, query: str, top_k: int = 10, use_cache: bool = True, max_features: int = 50000) -> Tuple[List[Tuple[str, float]], float]:
        """基于向量空间模型的检索（优化版本）"""
        start_time = time.time()
        
        if not hasattr(self, 'doc_vectors'):
            self.build_tfidf_vectors(use_cache=use_cache, max_features=max_features)
        
        # 转换查询为稀疏向量
        query_vectors = self.tfidf.transform([query])
        if not query_vectors:
            return [], 0.0
        
        query_vector = query_vectors[0]
        
        # 如果查询向量为空，返回空结果
        if not query_vector:
            return [], 0.0
        
        # 计算相似度
        similarities = []
        total_docs = len(self.doc_vectors)
        
        print(f"🔍 计算相似度，共 {total_docs} 个文档...")
        
        # 使用批量处理减少内存使用
        batch_size = 10000
        num_batches = (total_docs + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_docs)
            
            batch_progress = tqdm(
                range(start_idx, end_idx),
                desc=f"🔍 批次 {batch_idx + 1}/{num_batches}",
                total=end_idx - start_idx,
                unit="文档",
                ncols=100
            )
            
            for doc_id in batch_progress:
                doc_vector = self.doc_vectors.get(doc_id, {})
                similarity = sparse_cosine_similarity(query_vector, doc_vector)
                if similarity > 0:  # 只保留有相似度的文档
                    similarities.append((self.doc_ids[doc_id], similarity))
            
            batch_progress.close()
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_k]
        
        end_time = time.time()
        search_time = end_time - start_time
        
        # 记录搜索时间
        if 'search_times' not in self.processing_times:
            self.processing_times['search_times'] = []
        self.processing_times['search_times'].append(search_time)
        
        return results, search_time
    
    def get_processing_times(self) -> Dict[str, float]:
        return self.processing_times