import re
import time
import sys
import os
from typing import List, Dict
import json
import hashlib

class TextNormalizer:
    """统一的文本规范化器，自动处理 NLTK 可用性"""
    
    def __init__(self, cache_dir: str = "Meetup/cache"):  # 修改默认缓存路径
        self.processing_times = {}
        self.nltk_available = False
        self.stop_words = set()
        self.stemmer = None
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化文本处理组件
        self._initialize_processor()
    
    def _get_cache_key(self, documents: Dict[str, Dict]) -> str:
        """生成缓存键，基于文档内容"""
        # 使用文档ID和内容生成哈希
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
                print(f"从缓存加载规范化文档: {cache_file}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'normalized_docs' in data and 'metadata' in data:
                    print(f"从缓存加载了 {len(data['normalized_docs'])} 个规范化文档")
                    return data['normalized_docs']
            except Exception as e:
                print(f"加载规范化缓存失败: {e}")
        
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
            
            print(f"规范化文档已保存到缓存: {cache_file}")
        except Exception as e:
            print(f"保存规范化缓存失败: {e}")
    
    def _initialize_processor(self):
        """初始化文本处理器"""
        # 首先设置基本停用词
        self._setup_basic_stopwords()
        
        # 尝试初始化 NLTK
        if self._try_initialize_nltk():
            self.nltk_available = True
            print("[INFO] NLTK 文本处理器已初始化")
        else:
            print("[INFO] 使用内置文本处理器")
    
    def _setup_basic_stopwords(self):
        """设置基本停用词列表"""
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
    
    def _try_initialize_nltk(self):
        """尝试初始化 NLTK"""
        try:
            # 静默导入 NLTK
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer
            
            # 静默检查数据
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            
            # 检查必要的数据
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            
            # 恢复输出
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            # 初始化 NLTK 组件
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            
            # 测试 NLTK 功能
            test_tokens = word_tokenize("test sentence for nltk")
            if len(test_tokens) > 0:
                return True
            
        except Exception:
            # 恢复输出
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        
        return False
    
    def normalize_text(self, text: str) -> List[str]:
        """文本规范化处理流程"""
        if not text or not text.strip():
            return []
        
        try:
            # 1. 转换为小写
            text = text.lower().strip()
            
            # 2. 清理文本：移除标点、数字，保留连字符
            text = re.sub(r'[^a-zA-Z\s-]', ' ', text)
            
            # 3. 分词
            if self.nltk_available:
                tokens = self._nltk_tokenize(text)
            else:
                tokens = self._simple_tokenize(text)
            
            # 4. 过滤停用词和短词
            tokens = [token for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            
            # 5. 词干提取
            if self.nltk_available and self.stemmer:
                tokens = [self.stemmer.stem(token) for token in tokens]
            else:
                tokens = [self._simple_stem(token) for token in tokens]
            
            return tokens
            
        except Exception:
            # 如果任何步骤失败，使用最简化的分词
            return self._minimal_tokenize(text)
    
    def _nltk_tokenize(self, text: str) -> List[str]:
        """使用 NLTK 分词"""
        try:
            # 临时抑制 NLTK 错误输出
            import nltk
            from nltk.tokenize import word_tokenize
            
            original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            
            tokens = word_tokenize(text)
            
            # 恢复错误输出
            sys.stderr = original_stderr
            
            return tokens
            
        except Exception:
            # 恢复错误输出
            sys.stderr = sys.__stderr__
            # NLTK 分词失败，使用降级方案
            return self._simple_tokenize(text)
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """简单分词器"""
        # 简单的空格分词，处理连字符
        tokens = []
        current_token = []
        
        for char in text:
            if char.isalpha() or char == '-':
                current_token.append(char)
            elif current_token:
                token = ''.join(current_token)
                if '-' in token:
                    # 分割连字符单词
                    parts = [part for part in token.split('-') if len(part) > 2]
                    tokens.extend(parts)
                else:
                    tokens.append(token)
                current_token = []
        
        if current_token:
            token = ''.join(current_token)
            if '-' in token:
                parts = [part for part in token.split('-') if len(part) > 2]
                tokens.extend(parts)
            else:
                tokens.append(token)
        
        return tokens
    
    def _minimal_tokenize(self, text: str) -> List[str]:
        """最简化的分词方案"""
        if not text:
            return []
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s-]', ' ', text)
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
        
        tokens = [self._simple_stem(token) for token in tokens]
        
        return tokens
    
    def _simple_stem(self, word: str) -> str:
        """简单的词干提取"""
        if len(word) <= 3:
            return word
            
        if word.endswith('ies') and len(word) > 3:
            return word[:-3] + 'y'
        elif word.endswith('es') and len(word) > 2:
            return word[:-2]
        elif word.endswith('s') and len(word) > 1:
            return word[:-1]
        
        if word.endswith('ing') and len(word) > 3:
            return word[:-3]
        elif word.endswith('ed') and len(word) > 2:
            return word[:-2]
            
        return word
    
    def process_documents(self, documents: Dict[str, Dict], use_cache: bool = True) -> Dict[str, List[str]]:
        """处理所有文档"""
        # 尝试从缓存加载
        if use_cache:
            cached_docs = self.load_normalized_docs_from_cache(documents)
            if cached_docs is not None:
                return cached_docs
        
        start_time = time.time()
        normalized_docs = {}
        total_docs = len(documents)
        
        processor_type = "NLTK" if self.nltk_available else "内置"
        print(f"开始文本规范化处理 (使用{processor_type}处理器)...")
        
        # 优雅的进度显示
        processed_count = 0
        milestone = max(1, total_docs // 4)  # 分成4个里程碑
        
        for doc_id, doc_info in documents.items():
            content = doc_info['content']
            normalized_docs[doc_id] = self.normalize_text(content)
            
            processed_count += 1
            
            # 在里程碑显示进度
            if processed_count % milestone == 0 or processed_count == total_docs:
                percentage = int(processed_count / total_docs * 100)
                print(f"文本规范化进度: {percentage}% ({processed_count}/{total_docs})")
                
        end_time = time.time()
        self.processing_times['text_normalization'] = end_time - start_time
        
        print(f"文本规范化处理完成，耗时 {end_time - start_time:.2f}秒")
        
        # 保存到缓存
        if use_cache:
            self.save_normalized_docs_to_cache(normalized_docs, documents)
        
        return normalized_docs
    
    def get_processing_times(self) -> Dict:
        """获取处理时间统计"""
        return self.processing_times