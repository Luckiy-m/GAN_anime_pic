
import os
import sys
import argparse
import subprocess


def run_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e.stderr}")
        return False


def scrape_data(num_images=500, save_dir="./data/anime_faces", source="placeholder"):
    print("\n" + "="*50)
    print("步骤 1: 抓取动漫头像数据")
    print("="*50)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cmd = f"python scraper.py --num_images {num_images} --save_dir {save_dir} --source {source}"
    return run_command(cmd)


def train_model(num_epochs=100, data_dir="./data"):
    print("\n" + "="*50)
    print("步骤 2: 训练 GAN 模型")
    print("="*50)
    
    cmd = f"python train.py"
    return run_command(cmd)


def generate_images(checkpoint=None, num_images=64, output_dir="./results"):
    print("\n" + "="*50)
    print("步骤 3: 生成动漫头像")
    print("="*50)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if checkpoint is None:
        checkpoint = "./checkpoints/generator_epoch_100.pth"
        if not os.path.exists(checkpoint):
            checkpoints = [f for f in os.listdir("./checkpoints") if f.startswith("generator_epoch")]
            if checkpoints:
                checkpoints.sort(reverse=True)
                checkpoint = os.path.join("./checkpoints", checkpoints[0])
                print(f"使用最新的检查点: {checkpoint}")
    
    cmd = f"python generate.py --checkpoint {checkpoint} --num_images {num_images} --save_path {output_dir}/final_generated.png"
    return run_command(cmd)


def full_pipeline(num_images=500, num_epochs=100, skip_scrape=False, skip_train=False, skip_generate=False):
    print("开始完整的动漫头像生成流程！")
    print("="*50)
    
    success = True
    
    if not skip_scrape:
        if not scrape_data(num_images=num_images):
            print("数据抓取失败，终止流程")
            success = False
    
    if success and not skip_train:
        if not train_model(num_epochs=num_epochs):
            print("模型训练失败，终止流程")
            success = False
    
    if success and not skip_generate:
        if not generate_images():
            print("图像生成失败")
            success = False
    
    if success:
        print("\n" + "="*50)
        print("🎉 完整流程执行成功！")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ 流程执行失败")
        print("="*50)
    
    return success


def main():
    parser = argparse.ArgumentParser(description="动漫头像生成完整流程")
    
    subparsers = parser.add_subparsers(title="子命令", dest="command")
    
    full_parser = subparsers.add_parser("full", help="执行完整流程（抓取→训练→生成）")
    full_parser.add_argument("--num_images", type=int, default=500, help="抓取的图像数量")
    full_parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    full_parser.add_argument("--skip_scrape", action="store_true", help="跳过数据抓取步骤")
    full_parser.add_argument("--skip_train", action="store_true", help="跳过训练步骤")
    full_parser.add_argument("--skip_generate", action="store_true", help="跳过生成步骤")
    
    scrape_parser = subparsers.add_parser("scrape", help="仅执行数据抓取")
    scrape_parser.add_argument("--num_images", type=int, default=500, help="抓取的图像数量")
    scrape_parser.add_argument("--source", type=str, default="placeholder", 
                              choices=["placeholder", "thisanimedoesnotexist"], help="图像源")
    
    train_parser = subparsers.add_parser("train", help="仅执行模型训练")
    train_parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    
    generate_parser = subparsers.add_parser("generate", help="仅执行图像生成")
    generate_parser.add_argument("--checkpoint", type=str, help="模型检查点路径")
    generate_parser.add_argument("--num_images", type=int, default=64, help="生成的图像数量")
    
    args = parser.parse_args()
    
    if args.command == "full":
        full_pipeline(
            num_images=args.num_images,
            num_epochs=args.num_epochs,
            skip_scrape=args.skip_scrape,
            skip_train=args.skip_train,
            skip_generate=args.skip_generate
        )
    elif args.command == "scrape":
        scrape_data(num_images=args.num_images, source=args.source)
    elif args.command == "train":
        train_model(num_epochs=args.num_epochs)
    elif args.command == "generate":
        generate_images(checkpoint=args.checkpoint, num_images=args.num_images)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

