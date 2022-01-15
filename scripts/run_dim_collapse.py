import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from scripts.load_ckpt import return_system_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.491, 0.482, 0.446], [0.247, 0.243, 0.261])])
test_set = datasets.CIFAR10(root='scripts/data/cifar10/', train=False,
                                       download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


def compute_embed_matrix(model):          # model = expert_madel / viewmaker_model
  model = model.eval().to(device)
  embed_matrix = torch.Tensor().to(device);
  with torch.no_grad():
      for images, _ in test_loader:
          images = images.to(device)
          mini_batch_embed_matrix = model(images) # B x K
          # print(mini_batch_embed_matrix.shape)
          embed_matrix = torch.cat((embed_matrix, mini_batch_embed_matrix),0)
          # print(embed_matrix.shape)  
  return embed_matrix # N x K

enc_expert_resnet = return_system_encoder('expert', 198)
enc_viewmaker_resnet = return_system_encoder('viewmaker', 198)

# for p1, p2 in zip(enc_expert_resnet.parameters(), enc_viewmaker_resnet.parameters()):
#     if p1.data.ne(p2.data).sum() > 0:
#         print('False')
#     else:
#         print('True')

# import pdb; pdb.set_trace()
output_embed_matrix_expt = compute_embed_matrix(enc_expert_resnet)
output_embed_matrix_vwmkr = compute_embed_matrix(enc_viewmaker_resnet)


############################################################################
#               Rank & Singular Values of Embedding Matrices
############################################################################

print('Shape of output_embed_matrix = ',output_embed_matrix_expt.shape)
print('Shape of output_embed_matrix = ',output_embed_matrix_vwmkr.shape)

Rank = torch.matrix_rank(output_embed_matrix_expt)
print('Rank of expert embedding matrix = ',Rank.item())

Rank = torch.matrix_rank(output_embed_matrix_vwmkr)
print('Rank of viewmaker embedding matrix = ',Rank.item())

_,s_expert,_ = torch.svd(output_embed_matrix_expt, some=False)
_,s_viewmaker,_ = torch.svd(output_embed_matrix_vwmkr, some=False)


plt.figure(dpi=1200)
plt.plot(torch.log(s_expert.cpu()), "r-")
plt.plot(torch.log(s_viewmaker.cpu()), "b-")
plt.legend(["expert", "viewmaker"], loc ="upper right")
# plt.axis([-2, 128, -12, 2])
plt.xlabel('Singular Value Rank Index')
plt.ylabel('Log of Singular Values')

plt.savefig('log.png')
plt.show()


############################################################################
# Rank & Singular Values of Cross-Correlation Matrices of Embedding Matrices
############################################################################

N_expt = output_embed_matrix_expt.size(0)
Z_expt = (output_embed_matrix_expt - torch.mean(output_embed_matrix_expt,0)).t()
correlation_matrix_expt = (1/N_expt) * (Z_expt @ Z_expt.t())

N_vwmkr = output_embed_matrix_vwmkr.size(0)
Z_vwmkr = (output_embed_matrix_vwmkr - torch.mean(output_embed_matrix_vwmkr,0)).t()
correlation_matrix_vwmkr = (1/N_vwmkr) * (Z_vwmkr @ Z_vwmkr.t())


print('Shape of correlation_matrix = ',correlation_matrix_expt.shape)
print('Shape of correlation_matrix = ',correlation_matrix_vwmkr.shape)

Rank = torch.matrix_rank(correlation_matrix_expt)
print('Rank of expert correlation matrix = ',Rank.item())

Rank = torch.matrix_rank(correlation_matrix_vwmkr)
print('Rank of viewmaker correlation matrix = ',Rank.item())

_,s_corr_expert,_ = torch.svd(correlation_matrix_expt, some=False)
_,s_corr_viewmaker,_ = torch.svd(correlation_matrix_vwmkr, some=False)


plt.figure(dpi=1200)
plt.plot(torch.log(s_corr_expert.cpu()), "r-")
plt.plot(torch.log(s_corr_viewmaker.cpu()), "b-")
plt.legend(["expert", "viewmaker"], loc ="upper right")
# plt.axis([-2, 128, -12, 2])
plt.xlabel('Singular Value Rank Index')
plt.ylabel('Log of Singular Values')

plt.savefig('log.png')
plt.show()

