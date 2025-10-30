import re
import time
import sys
import os
from typing import List, Dict, Tuple
import json
import hashlib
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


# ============ 全局辅助函数（用于多进程） ============

def _process_single_document(doc_item: Tuple[str, Dict]) -> Tuple[str, List[str]]:
    """
    处理单个文档（用于多进程）
    必须是顶层函数才能被pickle序列化
    """
    doc_id, doc_info = doc_item
    content = doc_info['content']
    
    # 创建一个简化的规范化器实例
    normalizer = _SimpleNormalizer()
    return (doc_id, normalizer.normalize(content))


class _SimpleNormalizer:
    """简化的规范化器（用于多进程，避免NLTK序列化问题）"""
    
    def __init__(self):
        # 预编译正则表达式
        self.pattern_clean = re.compile(r'[^a-zA-Z\s\-]')
        self.pattern_tokenize = re.compile(r'\b[a-zA-Z][a-zA-Z\-]+\b')
        
        # 停用词集合
        self.stop_words = frozenset([
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'it', 'its',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'shall', 'ought', 'i', 'me', 'my', 'myself', 'we',
            'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'itself',
            'themselves'
        ])
        
        # 词干提取辅助集合
        self.double_consonants = frozenset('bdgmnprt')
        self.suffix2_endings = frozenset(['ch', 'sh', 'ss'])
        self.suffix4_set = frozenset(['ment', 'ness', 'tion'])
    
    def normalize(self, text: str) -> List[str]:
        """快速规范化（不依赖NLTK）"""
        if not text or not text.strip():
            return []
        
        # 转换为小写并清理
        text = text.lower().strip()
        text = self.pattern_clean.sub(' ', text)
        
        # 分词
        tokens = self.pattern_tokenize.findall(text)
        
        # 过滤并词干提取
        result = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                result.append(self._stem(token))
        
        return result
    
    def _stem(self, word: str) -> str:
        """快速词干提取"""
        word_len = len(word)
        
        if word_len <= 3:
            return word
        
        # 处理各种后缀
        if word_len > 3 and word[-3:] == 'ies':
            return word[:-3] + 'y'
        
        if word_len > 2 and word[-2:] == 'es':
            base = word[:-2]
            if len(base) >= 2 and base[-2:] in self.suffix2_endings:
                return base
            if len(base) >= 1 and base[-1] in 'sxz':
                return base
            return word[:-1]
        
        if word_len > 1 and word[-1] == 's' and word[-2:] != 'ss':
            return word[:-1]
        
        if word_len > 3 and word[-3:] == 'ing':
            base = word[:-3]
            if len(base) > 1 and base[-1] == base[-2] and base[-1] in self.double_consonants:
                return base[:-1]
            return base
        
        if word_len > 2 and word[-2:] == 'ed':
            base = word[:-2]
            if len(base) > 1 and base[-1] == base[-2] and base[-1] in self.double_consonants:
                return base[:-1]
            return base
        
        if word_len > 2 and word[-2:] == 'ly':
            return word[:-2]
        
        if word_len > 4 and word[-4:] in self.suffix4_set:
            return word[:-4]
        
        return word


# ============ 主要的TextNormalizer类 ============

class TextNormalizer:
    """统一的文本规范化器，自动处理 NLTK 可用性"""
    
    def __init__(self, cache_dir: str = "Meetup/cache"):
        self.processing_times = {}
        self.nltk_available = False
        self.stop_words = set()
        self.stemmer = None
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 预编译正则表达式（重大优化：避免每次调用都重新编译）
        self._compile_regex_patterns()
        
        # 初始化文本处理组件
        self._initialize_processor()
    
    def _compile_regex_patterns(self):
        """预编译所有正则表达式以提高性能"""
        # 主要清理模式
        self.pattern_clean = re.compile(r'[^a-zA-Z\s\-]')
        
        # 缩写替换模式（按优先级排序，避免冲突）
        self.pattern_contractions = [
            (re.compile(r"n't\b"), " not"),
            (re.compile(r"'re\b"), " are"),
            (re.compile(r"'ve\b"), " have"),
            (re.compile(r"'ll\b"), " will"),
            (re.compile(r"'d\b"), " would"),
            (re.compile(r"'m\b"), " am"),
            (re.compile(r"'s\b"), ""),  # 最后处理 's，避免与其他模式冲突
        ]
        
        # 分词模式
        self.pattern_tokenize = re.compile(r'\b[a-zA-Z][a-zA-Z\-]+\b')
        
        # 词干提取辅助集合（用于快速查找）
        self.double_consonants = frozenset('bdgmnprt')
        self.suffix2_endings = frozenset(['ch', 'sh', 'ss'])
        self.suffix4_set = frozenset(['ment', 'ness', 'tion'])
    
    def _initialize_processor(self):
        """初始化文本处理器"""
        # 设置停用词
        self._setup_comprehensive_stopwords()
        
        # 检查NLTK是否可用
        self._check_nltk_availability()
        
        if self.nltk_available:
            print("🔤 使用 NLTK 文本处理器")
        else:
            print("🔤 使用增强内置文本处理器")
    
    def _check_nltk_availability(self):
        """检查NLTK是否可用"""
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer
            
            # 检查数据是否可用
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            
            # 初始化NLTK组件
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            self.nltk_available = True
            
        except (LookupError, ImportError, OSError):
            self.nltk_available = False
    
    def _setup_comprehensive_stopwords(self):
        """设置全面的停用词列表"""
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'it', 'its', 
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'this', 'that', 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'been', 'being', 
            'have', 'has', 'had', 'do', 'does', 'did', 'shall', 'will', 'would', 
            'may', 'might', 'must', 'can', 'could', 'should', 'ought', 'i', 'me', 
            'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 
            'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 
            'themselves'
        }
    
    def _get_cache_key(self, documents: Dict[str, Dict]) -> str:
        """生成缓存键，基于文档内容"""
        content = "".join([doc_id + doc_info['content'] for doc_id, doc_info in documents.items()])
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_file(self, documents: Dict[str, Dict]) -> str:
        """获取缓存文件路径"""
        cache_key = self._get_cache_key(documents)
        return os.path.join(self.cache_dir, f"normalized_{cache_key}.json")
    
    def load_normalized_docs_from_cache(self, documents: Dict[str, Dict]) -> Dict[str, List[str]]:
        """从缓存加载规范化文档"""
        cache_file = self._get_cache_file(documents)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'normalized_docs' in data and 'metadata' in data:
                    print(f"✅ 从缓存加载了 {len(data['normalized_docs'])} 个规范化文档")
                    return data['normalized_docs']
            except Exception as e:
                print(f"❌ 加载规范化缓存失败: {e}")
        
        return None
    
    def save_normalized_docs_to_cache(self, normalized_docs: Dict[str, List[str]], documents: Dict[str, Dict]):
        """保存规范化文档到缓存"""
        cache_file = self._get_cache_file(documents)
        try:
            data = {
                'metadata': {
                    'document_count': len(normalized_docs),
                    'timestamp': time.time(),
                    'processor_type': 'NLTK' if self.nltk_available else '内置'
                },
                'normalized_docs': normalized_docs
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"❌ 保存规范化缓存失败: {e}")
    
    def normalize_text(self, text: str) -> List[str]:
        """文本规范化处理流程（优化版：使用预编译正则表达式）"""
        if not text or not text.strip():
            return []
        
        try:
            # 1. 转换为小写并清理文本
            text = text.lower().strip()
            
            # 2. 清理文本：使用预编译的正则表达式（性能提升）
            text = self.pattern_clean.sub(' ', text)
            
            # 3. 分词
            if self.nltk_available:
                tokens = self._nltk_tokenize(text)
            else:
                tokens = self._enhanced_tokenize(text)
            
            # 4. 过滤停用词和短词 + 词干提取（合并为一次遍历）
            result = []
            if self.nltk_available and self.stemmer:
                for token in tokens:
                    if token not in self.stop_words and len(token) > 2:
                        result.append(self.stemmer.stem(token))
            else:
                for token in tokens:
                    if token not in self.stop_words and len(token) > 2:
                        result.append(self._enhanced_stem(token))
            
            return result
            
        except Exception:
            # 如果任何步骤失败，使用最简化的分词
            return self._minimal_tokenize(text)
    
    def _nltk_tokenize(self, text: str) -> List[str]:
        """使用 NLTK 分词"""
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text)
        except Exception:
            return self._enhanced_tokenize(text)
    
    def _enhanced_tokenize(self, text: str) -> List[str]:
        """增强的分词器（优化版：使用预编译正则表达式）"""
        # 处理常见的缩写：使用预编译的模式
        for pattern, replacement in self.pattern_contractions:
            text = pattern.sub(replacement, text)
        
        # 使用预编译的正则表达式进行分词
        tokens = self.pattern_tokenize.findall(text)
        
        # 处理连字符单词
        processed_tokens = []
        for token in tokens:
            if '-' in token:
                # 分割连字符单词，但保留常见的复合词
                parts = token.split('-')
                if len(parts) == 2 and len(parts[0]) > 2 and len(parts[1]) > 2:
                    # 对于常见的复合词，同时保留整体和部分
                    processed_tokens.append(token)  # 整体
                    processed_tokens.extend([part for part in parts if len(part) > 2])  # 部分
                else:
                    processed_tokens.extend([part for part in parts if len(part) > 2])
            else:
                processed_tokens.append(token)
        
        return processed_tokens
    
    def _minimal_tokenize(self, text: str) -> List[str]:
        """最简化的分词方案（优化版：使用预编译正则）"""
        if not text:
            return []
        
        text = text.lower()
        text = self.pattern_clean.sub(' ', text)
        tokens = text.split()
        
        # 处理连字符
        processed_tokens = []
        for token in tokens:
            if '-' in token:
                parts = [part for part in token.split('-') if len(part) > 2]
                processed_tokens.extend(parts)
            else:
                processed_tokens.append(token)
        
        tokens = [token for token in processed_tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        tokens = [self._enhanced_stem(token) for token in tokens]
        
        return tokens
    
    def _enhanced_stem(self, word: str) -> str:
        """增强的词干提取（优化版：使用集合查找和缓存）"""
        word_len = len(word)
        
        if word_len <= 3:
            return word
        
        # 处理复数形式 -ies
        if word_len > 3 and word[-3:] == 'ies':
            return word[:-3] + 'y'
        
        # 处理 -es 结尾
        if word_len > 2 and word[-2:] == 'es':
            base = word[:-2]
            base_len = len(base)
            # 使用集合查找优化性能
            if base_len >= 2 and base[-2:] in self.suffix2_endings:
                return base
            if base_len >= 1 and base[-1] in 'sxz':
                return base
            return word[:-1]
        
        # 处理 -s 结尾（非 -ss）
        if word_len > 1 and word[-1] == 's' and word[-2:] != 'ss':
            return word[:-1]
        
        # 处理 -ing 结尾
        if word_len > 3 and word[-3:] == 'ing':
            base = word[:-3]
            base_len = len(base)
            # 双写辅音字母规则：使用集合查找
            if base_len > 1 and base[-1] == base[-2] and base[-1] in self.double_consonants:
                return base[:-1]
            return base
        
        # 处理 -ed 结尾
        if word_len > 2 and word[-2:] == 'ed':
            base = word[:-2]
            base_len = len(base)
            # 双写辅音字母规则
            if base_len > 1 and base[-1] == base[-2] and base[-1] in self.double_consonants:
                return base[:-1]
            return base
        
        # 处理 -ly 结尾
        if word_len > 2 and word[-2:] == 'ly':
            return word[:-2]
        
        # 处理名词后缀：使用集合查找
        if word_len > 4 and word[-4:] in self.suffix4_set:
            return word[:-4]
        
        return word
    
    def process_documents(self, documents: Dict[str, Dict], use_cache: bool = True, use_multiprocessing: bool = True) -> Dict[str, List[str]]:
        """处理所有文档（优化版：支持多进程并行）"""
        # 尝试从缓存加载
        if use_cache:
            cached_docs = self.load_normalized_docs_from_cache(documents)
            if cached_docs is not None:
                return cached_docs
        
        start_time = time.time()
        total_docs = len(documents)
        
        processor_type = "NLTK" if self.nltk_available else "增强内置"
        
        # 决定是否使用多进程
        use_parallel = use_multiprocessing and total_docs > 100 and cpu_count() > 1
        
        if use_parallel:
            num_workers = min(cpu_count(), 8)  # 最多使用8个进程
            print(f"🔤 文本规范化处理 (使用{processor_type}处理器，{num_workers}进程并行)")
            normalized_docs = self._process_parallel(documents, num_workers)
        else:
            print(f"🔤 文本规范化处理 (使用{processor_type}处理器，单进程)")
            normalized_docs = self._process_sequential(documents)
        
        end_time = time.time()
        self.processing_times['text_normalization'] = end_time - start_time
        
        print(f"✅ 文本规范化完成，耗时 {end_time - start_time:.2f}秒")
        
        # 保存到缓存（注意：参数顺序是 normalized_docs, documents）
        if use_cache:
            self.save_normalized_docs_to_cache(normalized_docs, documents)
        
        return normalized_docs
    
    def _process_sequential(self, documents: Dict[str, Dict]) -> Dict[str, List[str]]:
        """顺序处理文档"""
        normalized_docs = {}
        
        progress_bar = tqdm(
            documents.items(),
            desc="🔤 文本规范化",
            total=len(documents),
            unit="文档",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for doc_id, doc_info in progress_bar:
            content = doc_info['content']
            normalized_docs[doc_id] = self.normalize_text(content)
        
        progress_bar.close()
        return normalized_docs
    
    def _process_parallel(self, documents: Dict[str, Dict], num_workers: int) -> Dict[str, List[str]]:
        """并行处理文档（多进程）"""
        # 准备数据：转换为列表以便分块
        doc_items = list(documents.items())
        
        # 使用multiprocessing进行并行处理
        with Pool(processes=num_workers) as pool:
            # 使用imap_unordered以获得更好的进度反馈
            results = list(tqdm(
                pool.imap_unordered(
                    _process_single_document,
                    doc_items,
                    chunksize=max(1, len(doc_items) // (num_workers * 4))
                ),
                total=len(doc_items),
                desc="🔤 文本规范化",
                unit="文档",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ))
        
        # 转换回字典
        return dict(results)
    
    def get_processing_times(self) -> Dict:
        """获取处理时间统计"""
        return self.processing_times