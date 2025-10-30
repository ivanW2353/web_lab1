import xml.etree.ElementTree as ET
import os
import glob
from typing import List, Dict
import time
import json
import hashlib
from tqdm import tqdm

class DataProcessor:
    def __init__(self, data_path: str, max_files: int = None, cache_dir: str = "Meetup/cache"):
        self.data_path = data_path
        self.documents = {}  # doc_id -> document_content
        self.max_files = max_files
        self.processing_times = {}
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self) -> str:
        """生成缓存键，基于数据路径和文件数量限制"""
        key_data = f"{self.data_path}_{self.max_files}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_file(self) -> str:
        """获取缓存文件路径"""
        cache_key = self._get_cache_key()
        return os.path.join(self.cache_dir, f"documents_{cache_key}.json")
    
    def load_documents_from_cache(self) -> bool:
        """从缓存加载文档"""
        cache_file = self._get_cache_file()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'documents' in data and 'metadata' in data:
                    self.documents = data['documents']
                    print(f"✅ 从缓存加载了 {len(self.documents)} 个文档")
                    return True
            except Exception as e:
                print(f"❌ 加载缓存失败: {e}")
        
        return False
    
    def save_documents_to_cache(self):
        """保存文档到缓存"""
        cache_file = self._get_cache_file()
        try:
            data = {
                'metadata': {
                    'data_path': self.data_path,
                    'max_files': self.max_files,
                    'document_count': len(self.documents),
                    'timestamp': time.time()
                },
                'documents': self.documents
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"❌ 保存缓存失败: {e}")
        
    def parse_event_files(self, use_cache: bool = True):
        """解析Event XML文件"""
        # 尝试从缓存加载
        if use_cache and self.load_documents_from_cache():
            return
            
        print(f"📂 解析目录: {self.data_path}")
        start_time = time.time()
        
        # 获取所有XML文件
        xml_files = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.xml'):
                    xml_files.append(os.path.join(root, file))
        
        print(f"📄 找到 {len(xml_files)} 个 XML 文件")
        
        if self.max_files and self.max_files > 0:
            xml_files = xml_files[:self.max_files]
            print(f"🔢 限制解析前 {self.max_files} 个文件")
        else:
            print("🔢 处理所有文件")
        
        processed_count = 0
        error_count = 0
        
        # 使用 tqdm 进度条
        progress_bar = tqdm(
            xml_files,
            desc="📂 解析XML文件",
            unit="文件",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for file_path in progress_bar:
            try:
                # 解析XML文件
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # 提取事件信息
                event_id = self._get_text(root, 'id')
                event_name = self._get_text(root, 'name')
                description = self._get_text(root, 'description')
                
                # 尝试获取group信息
                group_elem = root.find('group')
                group_name = self._get_text(group_elem, 'name') if group_elem is not None else ""
                
                # 合并为文档内容
                doc_content = f"{event_name} {description} {group_name}"
                
                if event_id and doc_content.strip():
                    self.documents[event_id] = {
                        'content': doc_content,
                        'name': event_name,
                        'group': group_name,
                        'file_path': file_path
                    }
                    processed_count += 1
                    
            except ET.ParseError as e:
                error_count += 1
            except Exception as e:
                error_count += 1
        
        # 关闭进度条
        progress_bar.close()
        
        end_time = time.time()
        self.processing_times['data_parsing'] = end_time - start_time
        
        print(f"✅ 数据解析完成: 成功 {processed_count}, 错误 {error_count}, 耗时 {end_time - start_time:.2f}秒")
        
        # 保存到缓存
        if use_cache:
            self.save_documents_to_cache()
                
    def _get_text(self, element, tag: str) -> str:
        """安全获取XML元素文本"""
        if element is None:
            return ""
        elem = element.find(tag)
        return elem.text if elem is not None else ""
    
    def get_documents(self) -> Dict[str, Dict]:
        return self.documents
    
    def get_document_count(self) -> int:
        return len(self.documents)
    
    def get_processing_times(self) -> Dict:
        """获取处理时间统计"""
        return self.processing_times