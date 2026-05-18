
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
from model import Generator


def generate_images(checkpoint_path, num_images=64, latent_dim=100, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    
    netG = Generator(latent_dim).to(device)
    netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
    netG.eval()
    
    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    
    with torch.no_grad():
        fake = netG(noise).detach().cpu()
    
    return fake


def plot_images(images, save_path=None, grid_size=8):
    grid = vutils.make_grid(images, nrow=grid_size, padding=2, normalize=True)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"图像已保存到: {save_path}")
    
    plt.show()


def interpolate_images(checkpoint_path, num_interpolations=10, latent_dim=100, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    netG = Generator(latent_dim).to(device)
    netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
    netG.eval()
    
    z1 = torch.randn(1, latent_dim, 1, 1, device=device)
    z2 = torch.randn(1, latent_dim, 1, 1, device=device)
    
    interpolated_z = []
    for i in range(num_interpolations):
        alpha = i / (num_interpolations - 1)
        z = z1 * (1 - alpha) + z2 * alpha
        interpolated_z.append(z)
    
    interpolated_z = torch.cat(interpolated_z, dim=0)
    
    with torch.no_grad():
        fake = netG(interpolated_z).detach().cpu()
    
    return fake


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="动漫头像生成器")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/generator_epoch_100.pth", help="生成器模型检查点路径")
    parser.add_argument("--num_images", type=int, default=64, help="生成图像数量")
    parser.add_argument("--mode", type=str, default="generate", choices=["generate", "interpolate"], help="模式: generate(生成) 或 interpolate(插值)")
    parser.add_argument("--save_path", type=str, default="./results/generated.png", help="保存路径")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        images = generate_images(args.checkpoint, args.num_images)
        plot_images(images, args.save_path)
    elif args.mode == "interpolate":
        images = interpolate_images(args.checkpoint, args.num_images)
        plot_images(images, args.save_path, grid_size=args.num_images)

