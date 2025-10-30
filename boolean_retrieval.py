from typing import List, Set, Dict
import time

class BooleanRetrieval:
    def __init__(self, inverted_index):
        self.index = inverted_index
        self.processing_times = {}
        
    def search(self, query: str) -> Set[str]:
        """执行布尔检索"""
        start_time = time.time()
        
        query = query.lower().strip()
        
        if " and " in query:
            parts = query.split(" and ")
            results = self._and_operation(
                self._process_term(parts[0].strip()),
                self._process_term(parts[1].strip())
            )
        elif " or " in query:
            parts = query.split(" or ")
            results = self._or_operation(
                self._process_term(parts[0].strip()),
                self._process_term(parts[1].strip())
            )
        elif query.startswith("not "):
            term = query[4:].strip()
            results = self._not_operation(self._process_term(term))
        else:
            results = self._process_term(query)
        
        end_time = time.time()
        search_time = end_time - start_time
        
        # 记录搜索时间
        if 'search_times' not in self.processing_times:
            self.processing_times['search_times'] = []
        self.processing_times['search_times'].append(search_time)
        
        return results
    
    def _process_term(self, term: str) -> Set[str]:
        """处理单个词项查询"""
        if term in self.index.index:
            return set(self.index.index[term].keys())
        return set()
    
    def _and_operation(self, set1: Set[str], set2: Set[str]) -> Set[str]:
        return set1.intersection(set2)
    
    def _or_operation(self, set1: Set[str], set2: Set[str]) -> Set[str]:
        return set1.union(set2)
    
    def _not_operation(self, term_set: Set[str]) -> Set[str]:
        all_docs = set(self.index.doc_lengths.keys())
        return all_docs - term_set
    
    def get_processing_times(self) -> Dict:
        """获取处理时间统计"""
        return self.processing_times