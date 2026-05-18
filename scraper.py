
import os
import time
import random
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed


class AnimeFaceScraper:
    def __init__(self, save_dir="./data/anime_faces", max_workers=5):
        self.save_dir = save_dir
        self.max_workers = max_workers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def download_image(self, url, filename):
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(os.path.join(self.save_dir, filename))
            print(f"下载成功: {filename}")
            return True
        except Exception as e:
            print(f"下载失败 {url}: {e}")
            return False
    
    def scrape_from_thispersondoesnotexist(self, num_images=100):
        print(f"开始从 thispersondoesnotexist.art 风格的网站获取 {num_images} 张动漫头像...")
        
        base_url = "https://thisanimedoesnotexist.ai/results/psi-1.0/seed{}.png"
        
        downloaded = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(num_images):
                seed = random.randint(0, 1000000)
                url = base_url.format(seed)
                filename = f"anime_face_{int(time.time())}_{i}.png"
                futures.append(executor.submit(self.download_image, url, filename))
                time.sleep(0.5)
            
            for future in as_completed(futures):
                if future.result():
                    downloaded += 1
        
        print(f"完成！下载了 {downloaded}/{num_images} 张图片")
        return downloaded
    
    def scrape_from_placeholder(self, num_images=100):
        print(f"使用占位图像生成 {num_images} 张示例图像...")
        
        colors = [
            (255, 200, 200), (200, 255, 200), (200, 200, 255),
            (255, 255, 200), (255, 200, 255), (200, 255, 255),
            (255, 220, 180), (220, 180, 255), (180, 255, 220)
        ]
        
        downloaded = 0
        for i in range(num_images):
            try:
                size = 64
                color = random.choice(colors)
                image = Image.new('RGB', (size, size), color)
                
                for _ in range(50):
                    x = random.randint(0, size-1)
                    y = random.randint(0, size-1)
                    dx = random.randint(-3, 3)
                    dy = random.randint(-3, 3)
                    if 0 &lt;= x+dx &lt; size and 0 &lt;= y+dy &lt; size:
                        image.putpixel((x, y), image.getpixel((x+dx, y+dy)))
                
                filename = f"sample_face_{int(time.time())}_{i}.png"
                image.save(os.path.join(self.save_dir, filename))
                print(f"生成: {filename}")
                downloaded += 1
                time.sleep(0.1)
            except Exception as e:
                print(f"生成失败: {e}")
        
        print(f"完成！生成了 {downloaded} 张示例图片")
        return downloaded
    
    def scrape(self, num_images=100, source="placeholder"):
        print(f"开始抓取 {num_images} 张动漫头像到 {self.save_dir}")
        
        if source == "thisanimedoesnotexist":
            return self.scrape_from_thispersondoesnotexist(num_images)
        else:
            return self.scrape_from_placeholder(num_images)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="动漫头像爬虫")
    parser.add_argument("--num_images", type=int, default=100, help="要抓取的图像数量")
    parser.add_argument("--save_dir", type=str, default="./data/anime_faces", help="保存目录")
    parser.add_argument("--source", type=str, default="placeholder", 
                        choices=["placeholder", "thisanimedoesnotexist"], help="图像源")
    
    args = parser.parse_args()
    
    scraper = AnimeFaceScraper(save_dir=args.save_dir)
    scraper.scrape(num_images=args.num_images, source=args.source)


if __name__ == "__main__":
    main()

