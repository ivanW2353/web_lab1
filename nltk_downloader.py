import nltk
import os
import sys

class NLTKDataDownloader:
    """静默下载和管理 NLTK 数据"""
    
    @staticmethod
    def download_required_data():
        """下载所有必需的 NLTK 数据"""
        required_packages = ['punkt', 'stopwords', 'punkt_tab']
        
        print("检查 NLTK 数据...")
        
        for package in required_packages:
            try:
                # 静默检查数据是否存在
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
                
                if package == 'punkt_tab':
                    # 特殊处理 punkt_tab
                    try:
                        nltk.data.find(f'tokenizers/punkt/PY3/english.pickle')
                        print(f"[OK] {package} 已存在")
                        continue
                    except LookupError:
                        pass
                else:
                    try:
                        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
                        print(f"[OK] {package} 已存在")
                        continue
                    except LookupError:
                        pass
                
                # 恢复输出以显示下载信息
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                print(f"下载 {package}...", end=" ", flush=True)
                
                # 静默下载
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
                
                nltk.download(package, quiet=True)
                
                # 恢复输出
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                print("[OK]")
                
            except Exception as e:
                # 恢复输出
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                print(f"[FAIL] ({str(e)[:50]}...)")
        
        print("NLTK 数据检查完成")

if __name__ == "__main__":
    NLTKDataDownloader.download_required_data()