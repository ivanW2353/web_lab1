import re
import time
import sys
import os
from typing import List, Dict
import json
import hashlib
from tqdm import tqdm

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
        
        # 初始化文本处理组件
        self._initialize_processor()
    
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
        """文本规范化处理流程"""
        if not text or not text.strip():
            return []
        
        try:
            # 1. 转换为小写
            text = text.lower().strip()
            
            # 2. 清理文本：移除标点、数字，保留连字符和基本符号
            text = re.sub(r'[^a-zA-Z\s\-]', ' ', text)
            
            # 3. 分词
            if self.nltk_available:
                tokens = self._nltk_tokenize(text)
            else:
                tokens = self._enhanced_tokenize(text)
            
            # 4. 过滤停用词和短词
            tokens = [token for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            
            # 5. 词干提取
            if self.nltk_available and self.stemmer:
                tokens = [self.stemmer.stem(token) for token in tokens]
            else:
                tokens = [self._enhanced_stem(token) for token in tokens]
            
            return tokens
            
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
        """增强的分词器"""
        # 处理常见的缩写和特殊字符
        text = re.sub(r"n't\b", " not", text)
        text = re.sub(r"'s\b", "", text)
        text = re.sub(r"'re\b", " are", text)
        text = re.sub(r"'ve\b", " have", text)
        text = re.sub(r"'ll\b", " will", text)
        text = re.sub(r"'d\b", " would", text)
        text = re.sub(r"'m\b", " am", text)
        
        # 使用正则表达式进行分词
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z\-]+\b', text)
        
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
        """最简化的分词方案"""
        if not text:
            return []
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s\-]', ' ', text)
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
        """增强的词干提取"""
        if len(word) <= 3:
            return word
        
        # 处理复数形式
        if word.endswith('ies') and len(word) > 3:
            return word[:-3] + 'y'
        elif word.endswith('es') and len(word) > 2:
            base = word[:-2]
            # 检查是否是特定结尾
            if base.endswith(('s', 'x', 'z', 'ch', 'sh')):
                return base
            else:
                return word[:-1]
        elif word.endswith('s') and len(word) > 1 and not word.endswith('ss'):
            return word[:-1]
        
        # 处理动词形式
        if word.endswith('ing') and len(word) > 3:
            base = word[:-3]
            # 双写辅音字母规则
            if len(base) > 1 and base[-1] == base[-2] and base[-1] in 'bdgmnprt':
                return base[:-1]
            else:
                return base
        elif word.endswith('ed') and len(word) > 2:
            base = word[:-2]
            # 双写辅音字母规则
            if len(base) > 1 and base[-1] == base[-2] and base[-1] in 'bdgmnprt':
                return base[:-1]
            else:
                return base
        
        # 处理副词
        if word.endswith('ly') and len(word) > 2:
            return word[:-2]
        
        # 处理名词后缀
        if word.endswith('ment') and len(word) > 4:
            return word[:-4]
        elif word.endswith('ness') and len(word) > 4:
            return word[:-4]
        elif word.endswith('tion') and len(word) > 4:
            return word[:-4]
        
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
        
        processor_type = "NLTK" if self.nltk_available else "增强内置"
        print(f"🔤 文本规范化处理 (使用{processor_type}处理器)")
        
        # 使用 tqdm 进度条
        progress_bar = tqdm(
            documents.items(),
            desc="🔤 文本规范化",
            total=total_docs,
            unit="文档",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for doc_id, doc_info in progress_bar:
            content = doc_info['content']
            normalized_docs[doc_id] = self.normalize_text(content)
        
        # 关闭进度条
        progress_bar.close()
        
        end_time = time.time()
        self.processing_times['text_normalization'] = end_time - start_time
        
        print(f"✅ 文本规范化完成，耗时 {end_time - start_time:.2f}秒")
        
        # 保存到缓存
        if use_cache:
            self.save_normalized_docs_to_cache(normalized_docs, documents)
        
        return normalized_docs
    
    def get_processing_times(self) -> Dict:
        """获取处理时间统计"""
        return self.processing_times