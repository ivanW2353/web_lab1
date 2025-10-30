import time
from typing import List, Dict, Tuple

class SimpleTFIDF:
    """简化的TF-IDF实现"""
    def __init__(self):
        self.vocab = {}
        self.doc_freq = {}
        self.doc_count = 0
        self.doc_vectors = {}
        
    def fit_transform(self, documents: List[str]) -> Dict[int, List[float]]:
        """训练TF-IDF模型并转换文档"""
        import math
        from collections import defaultdict
        
        self.doc_count = len(documents)
        
        # 构建词汇表
        all_terms = set()
        doc_terms = []
        
        print("构建词汇表...")
        for i, doc in enumerate(documents):
            terms = self._tokenize(doc)
            doc_terms.append(terms)
            all_terms.update(terms)
            
            if (i + 1) % 100 == 0:
                print(f"处理文档: {i + 1}/{len(documents)}")
        
        self.vocab = {term: idx for idx, term in enumerate(sorted(all_terms))}
        
        # 计算文档频率
        print("计算文档频率...")
        for i, terms in enumerate(doc_terms):
            for term in set(terms):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        
        # 计算TF-IDF向量
        print("计算TF-IDF向量...")
        vectors = {}
        for doc_id, terms in enumerate(doc_terms):
            vector = [0.0] * len(self.vocab)
            term_count = len(terms)
            
            # 计算TF
            tf_dict = defaultdict(int)
            for term in terms:
                tf_dict[term] += 1
            
            # 计算TF-IDF
            for term, count in tf_dict.items():
                if term in self.vocab:
                    idx = self.vocab[term]
                    tf = count / term_count
                    idf = math.log(self.doc_count / (self.doc_freq[term] + 1))
                    vector[idx] = tf * idf
            
            vectors[doc_id] = vector
            
            if (doc_id + 1) % 100 == 0:
                print(f"计算向量: {doc_id + 1}/{len(doc_terms)}")
        
        return vectors
    
    def transform(self, texts: List[str]) -> List[List[float]]:
        """转换新文本为TF-IDF向量"""
        import math
        from collections import defaultdict
        
        vectors = []
        for text in texts:
            terms = self._tokenize(text)
            vector = [0.0] * len(self.vocab)
            term_count = len(terms)
            
            tf_dict = defaultdict(int)
            for term in terms:
                tf_dict[term] += 1
            
            for term, count in tf_dict.items():
                if term in self.vocab:
                    idx = self.vocab[term]
                    tf = count / term_count
                    idf = math.log(self.doc_count / (self.doc_freq[term] + 1))
                    vector[idx] = tf * idf
            
            vectors.append(vector)
        
        return vectors
    
    def _tokenize(self, text: str) -> List[str]:
        """简单的分词函数"""
        if not text:
            return []
        # 转换为小写并分割单词
        text = text.lower()
        # 移除非字母字符
        words = []
        current_word = []
        for char in text:
            if char.isalpha():
                current_word.append(char)
            elif current_word:
                words.append(''.join(current_word))
                current_word = []
        if current_word:
            words.append(''.join(current_word))
        return [w for w in words if len(w) > 2]

def simple_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    import math
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

class VectorRetrieval:
    def __init__(self, documents: Dict[str, Dict]):
        self.documents = documents
        self.simple_tfidf = SimpleTFIDF()
        self.processing_times: Dict[str, float] = {}
        
    def build_tfidf_vectors(self):
        """构建TF-IDF向量"""
        start_time = time.time()
        print("构建TF-IDF向量...")
        
        # 准备文档文本
        doc_texts = []
        doc_ids = []
        
        for doc_id, doc_info in self.documents.items():
            doc_texts.append(doc_info['content'])
            doc_ids.append(doc_id)
        
        # 使用简化TF-IDF
        self.doc_vectors = self.simple_tfidf.fit_transform(doc_texts)
        self.doc_ids = doc_ids
        
        end_time = time.time()
        self.processing_times['tfidf_building'] = end_time - start_time
        
        print(f"TF-IDF向量构建完成，词汇表大小: {len(self.simple_tfidf.vocab)}，耗时 {end_time - start_time:.2f}秒")
    
    def search(self, query: str, top_k: int = 10) -> Tuple[List[Tuple[str, float]], float]:
        """基于向量空间模型的检索"""
        start_time = time.time()
        
        if not hasattr(self, 'doc_vectors'):
            self.build_tfidf_vectors()
        
        # 转换查询为向量
        query_vector = self.simple_tfidf.transform([query])[0]
        
        # 计算相似度
        print("计算文档相似度...")
        similarities = []
        total_docs = len(self.doc_vectors)
        
        for i, (doc_id, doc_vector) in enumerate(self.doc_vectors.items()):
            similarity = simple_cosine_similarity(query_vector, doc_vector)
            similarities.append((self.doc_ids[doc_id], similarity))
            
            if (i + 1) % 100 == 0:
                print(f"计算进度: {i + 1}/{total_docs}")
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = [(doc_id, score) for doc_id, score in similarities[:top_k] if score > 0]
        
        end_time = time.time()
        search_time = end_time - start_time
        
        # 记录搜索时间
        if 'search_times' not in self.processing_times:
            self.processing_times['search_times'] = []
        self.processing_times['search_times'].append(search_time)
        
        return results, search_time
    
    def get_processing_times(self) -> Dict[str, float]:
        return self.processing_times