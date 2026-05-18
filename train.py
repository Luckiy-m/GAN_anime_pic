
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
from model import Generator, Discriminator, weights_init
from dataset import get_data_loader, denormalize


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    data_dir = "./data"
    image_size = 64
    batch_size = 128
    num_workers = 4
    latent_dim = 100
    num_epochs = 100
    lr = 0.0002
    beta1 = 0.5
    
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not os.path.exists("./results"):
        os.makedirs("./results")
    
    print("加载数据...")
    dataloader = get_data_loader(data_dir, image_size, batch_size, num_workers)
    
    print("初始化网络...")
    netG = Generator(latent_dim).to(device)
    netD = Discriminator().to(device)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    criterion = nn.BCELoss()
    
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    
    real_label = 1.0
    fake_label = 0.0
    
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
    print("开始训练...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            netG.zero_grad()
            label.fill_(real_label)
            
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            optimizerG.step()
            
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
            iters += 1
        
        torch.save(netG.state_dict(), f'./checkpoints/generator_epoch_{epoch+1}.pth')
        torch.save(netD.state_dict(), f'./checkpoints/discriminator_epoch_{epoch+1}.pth')
        
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        grid = vutils.make_grid(fake, padding=2, normalize=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(f'./results/epoch_{epoch+1}.png')
        plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./results/loss_curve.png')
    plt.close()
    
    print("训练完成！")


if __name__ == "__main__":
    train()

