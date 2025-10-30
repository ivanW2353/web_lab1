import nltk
import os
import sys

class NLTKDataDownloader:
    """静默下载和管理 NLTK 数据"""
    
    @staticmethod
    def download_required_data():
        """下载所有必需的 NLTK 数据"""
        
        required_packages = ['punkt', 'stopwords']
        
        # 设置NLTK数据下载路径到Meetup/nltk_data
        nltk_data_dir = os.path.join("Meetup", "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # 确保路径在NLTK数据路径列表中
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_dir)  # 插入到开头，优先使用
        
        success_count = 0
        total_count = len(required_packages)
        
        for package in required_packages:
            try:
                # 首先检查数据是否已经存在
                try:
                    if package == 'punkt':
                        nltk.data.find('tokenizers/punkt')
                    elif package == 'stopwords':
                        nltk.data.find('corpora/stopwords')
                    
                    success_count += 1
                    continue
                    
                except LookupError:
                    pass
                
                # 数据不存在，开始下载
                print(f"📥 下载 {package}...", end=" ")
                
                # 使用nltk下载
                nltk.download(package, download_dir=nltk_data_dir, quiet=True)
                
                # 验证下载
                try:
                    if package == 'punkt':
                        nltk.data.find('tokenizers/punkt')
                    elif package == 'stopwords':
                        nltk.data.find('corpora/stopwords')
                    
                    print("✅")
                    success_count += 1
                    
                except LookupError:
                    print("❌ (验证失败)")
                    
            except Exception as e:
                print(f"❌ ({str(e)[:50]}...)")
        
        # 最终验证
        final_success = 0
        for package in required_packages:
            try:
                if package == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif package == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                final_success += 1
            except LookupError:
                pass
        
        if final_success == total_count:
            print(f"📁 NLTK数据位置: {nltk_data_dir}")
        else:
            print(f"⚠️  NLTK数据不完整: {final_success}/{total_count}")
        
        return final_success == total_count

if __name__ == "__main__":
    NLTKDataDownloader.download_required_data()