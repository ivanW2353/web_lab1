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
        """解析所有类型的 XML 文件（PastEvent, Member, Group, RSVPs）"""
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
        
        # 统计各类型文件的处理情况
        stats = {
            'PastEvent': {'processed': 0, 'skipped': 0},
            'Member': {'processed': 0, 'skipped': 0},
            'Group': {'processed': 0, 'skipped': 0},
            'RSVPs': {'processed': 0, 'skipped': 0},
            'Unknown': {'processed': 0, 'skipped': 0}
        }
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
                # 根据文件名判断类型
                filename = os.path.basename(file_path)
                file_type = self._get_file_type(filename)
                
                # 根据类型调用不同的解析方法
                if file_type == 'PastEvent':
                    count = self._parse_pastevent(file_path)
                elif file_type == 'Member':
                    count = self._parse_member(file_path)
                elif file_type == 'Group':
                    count = self._parse_group(file_path)
                elif file_type == 'RSVPs':
                    count = self._parse_rsvps(file_path)
                else:
                    count = 0
                
                if count > 0:
                    stats[file_type]['processed'] += count
                else:
                    stats[file_type]['skipped'] += 1
                    
            except ET.ParseError:
                error_count += 1
            except Exception as e:
                error_count += 1
        
        # 关闭进度条
        progress_bar.close()
        
        end_time = time.time()
        self.processing_times['data_parsing'] = end_time - start_time
        
        # 详细统计信息
        total_files = len(xml_files)
        total_processed = sum(s['processed'] for s in stats.values())
        total_skipped = sum(s['skipped'] for s in stats.values())
        
        print(f"✅ 数据解析完成: 耗时 {end_time - start_time:.2f}秒")
        print(f"   📊 总计: {total_files} 文件 -> {total_processed} 文档")
        print(f"\n   📁 各类型统计:")
        for ftype, counts in stats.items():
            if counts['processed'] > 0 or counts['skipped'] > 0:
                total = counts['processed'] + counts['skipped']
                print(f"      {ftype:12} {total:>6} 文件 -> {counts['processed']:>6} 文档 (跳过 {counts['skipped']})")
        
        if error_count > 0:
            print(f"\n   ❌ {error_count} 个文件解析失败（XML格式错误）")
        
        # 保存到缓存
        if use_cache:
            self.save_documents_to_cache()
    
    def _get_file_type(self, filename: str) -> str:
        """根据文件名判断文件类型"""
        if filename.startswith('PastEvent '):
            return 'PastEvent'
        elif filename.startswith('Memeber '):  # 注意拼写错误
            return 'Member'
        elif filename.startswith('Group '):
            return 'Group'
        elif filename.startswith('RSVPs '):
            return 'RSVPs'
        else:
            return 'Unknown'
    
    def _parse_pastevent(self, file_path: str) -> int:
        """解析 PastEvent 文件（历史事件）"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        event_id = self._get_text(root, 'id')
        event_name = self._get_text(root, 'name')
        description = self._get_text(root, 'description')
        
        # 获取group信息
        group_elem = root.find('group')
        group_name = self._get_text(group_elem, 'name') if group_elem is not None else ""
        
        # 合并为文档内容
        doc_content = f"{event_name} {description} {group_name}"
        
        if event_id and doc_content.strip():
            self.documents[event_id] = {
                'content': doc_content,
                'name': event_name,
                'group': group_name,
                'file_path': file_path,
                'type': 'PastEvent'
            }
            return 1
        return 0
    
    def _parse_member(self, file_path: str) -> int:
        """解析 Member 文件（成员信息）"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        member_id = self._get_text(root, 'id')
        member_name = self._get_text(root, 'name')
        bio = self._get_text(root, 'bio')
        hometown = self._get_text(root, 'hometown')
        
        # 获取 topics（兴趣话题）
        topics_text = ""
        topics_elem = root.find('topics')
        if topics_elem is not None:
            topic_names = [self._get_text(t, 'name') for t in topics_elem.findall('.//item')]
            topics_text = " ".join(topic_names)
        
        # 合并为文档内容
        doc_content = f"{member_name} {bio} {hometown} {topics_text}"
        
        if member_id and doc_content.strip():
            self.documents[member_id] = {
                'content': doc_content,
                'name': member_name,
                'group': hometown,  # 用 hometown 代替 group
                'file_path': file_path,
                'type': 'Member'
            }
            return 1
        return 0
    
    def _parse_group(self, file_path: str) -> int:
        """解析 Group 文件（群组信息）"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        group_id = self._get_text(root, 'id')
        group_name = self._get_text(root, 'name')
        description = self._get_text(root, 'description')
        who = self._get_text(root, 'who')  # 群组成员称呼
        
        # 获取 topics
        topics_text = ""
        topics_elem = root.find('topics')
        if topics_elem is not None:
            topic_names = [self._get_text(t, 'name') for t in topics_elem.findall('.//item')]
            topics_text = " ".join(topic_names)
        
        # 合并为文档内容
        doc_content = f"{group_name} {description} {who} {topics_text}"
        
        if group_id and doc_content.strip():
            self.documents[group_id] = {
                'content': doc_content,
                'name': group_name,
                'group': group_name,
                'file_path': file_path,
                'type': 'Group'
            }
            return 1
        return 0
    
    def _parse_rsvps(self, file_path: str) -> int:
        """解析 RSVPs 文件（活动报名列表）"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # RSVPs 文件结构: <results><items><item>...</item>...</items></results>
        items_elem = root.find('items')
        if items_elem is None:
            return 0
        
        count = 0
        for item in items_elem.findall('item'):
            rsvp_id = self._get_text(item, 'rsvp_id')
            response = self._get_text(item, 'response')  # yes/no/waitlist
            comments = self._get_text(item, 'comments')
            
            # 获取成员信息
            member_elem = item.find('member')
            member_name = self._get_text(member_elem, 'name') if member_elem is not None else ""
            
            # 获取事件信息
            event_elem = item.find('event')
            event_name = self._get_text(event_elem, 'name') if event_elem is not None else ""
            
            # 合并为文档内容
            doc_content = f"{member_name} {response} {event_name} {comments}"
            
            if rsvp_id and doc_content.strip():
                self.documents[rsvp_id] = {
                    'content': doc_content,
                    'name': f"{member_name} RSVP",
                    'group': event_name,
                    'file_path': file_path,
                    'type': 'RSVP'
                }
                count += 1
        
        return count
                
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