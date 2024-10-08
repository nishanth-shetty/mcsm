import torch
import torch.nn as nn
import time


def mcsm_loss_old(scorenet, samples, n_particles=1, m=20, eps=1e-3):

    N = samples.shape[0]
    scores = scorenet(samples)

    first_term = torch.norm(scores, p=2) ** 2
    sums = 0.0

    for i in range(m):
        b = torch.randn_like(samples)
        forw = scorenet(samples + eps * b)
        back = scorenet(samples - eps * b)
        sums += torch.einsum('bi,bi->b', (b.view(N, -1),
                             (forw - back).view(N, -1)))

    second_term = torch.mean((1 / (m * eps)) * sums)
    loss = 1 / 2. * (torch.mean(first_term) + second_term)

    return loss, torch.mean(first_term), second_term


def mcsm_forward_loss(scorenet, samples, n_particles=1, m=20, eps=1e-3):

    list_indices = [x for x in range(1, len(list(samples.shape)))]
    N = samples.shape[0]
    scores = scorenet(samples)

    loss1 = torch.sum(scores * scores, dim=list_indices) / 2.0
    sums = 0.0

    for i in range(m):
        b = torch.randn_like(samples)
        forw = scorenet(samples + eps * b)
        back = scorenet(samples)
        sums += torch.einsum('bi,bi->b', (b.view(N, -1),
                             (forw - back).view(N, -1)))

    loss2 = sums.mean() / (m * eps)
    
    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2

    return loss.mean(), loss1.mean(), loss2.mean()


def mcsm_backward_loss(scorenet, samples, n_particles=1, m=20, eps=1e-3):

    list_indices = [x for x in range(1, len(list(samples.shape)))]
    N = samples.shape[0]
    scores = scorenet(samples)

    loss1 = torch.sum(scores * scores, dim=list_indices) / 2.0
    sums = 0.0

    for i in range(m):
        b = torch.randn_like(samples)
        forw = scorenet(samples)
        back = scorenet(samples - eps * b)
        sums += torch.einsum('bi,bi->b', (b.view(N, -1),
                             (forw - back).view(N, -1)))

    loss2 = sums.mean() / (m * eps)
    
    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2

    return loss.mean(), loss1.mean(), loss2.mean()


def mcsm_loss_optimized(scorenet, samples, n_particles=1, m=20, eps=1e-3):
    
    N = samples.shape[0]
    scores = scorenet(samples)

    # Compute the first term of the loss
    loss1 = torch.sum(scores * scores, dim=tuple(range(1, scores.dim()))) / 2.0

    # Generate all m random vectors at once
    b = torch.randn(m, N, *samples.shape[1:]).to(samples.device)

    # Compute forward and backward scores at once
    samples_eps_b = samples.unsqueeze(0).expand(m, N, *samples.shape[1:])
    forw_samples = samples_eps_b + eps * b
    back_samples = samples_eps_b - eps * b

    # Concatenate along the batch dimension
    forw_back_samples = torch.cat([forw_samples, back_samples], dim=1)
    forw_back_samples = forw_back_samples.view(-1, *samples.shape[1:])
    print(f"{forw_back_samples.shape}")
    forw_back_scores = scorenet(forw_back_samples)

    # Reshape and compute sums
    b_reshaped = b.view(m * N, -1)
    forw_back_scores_reshaped = forw_back_scores.view(2 * m * N, -1)
    sums = torch.einsum('bi,bi->b', b_reshaped, forw_back_scores_reshaped[:m * N] - forw_back_scores_reshaped[m * N:])

    # Compute the second term of the loss
    sums_reshaped = sums.view(m, N).mean(dim=0)  # Average over m samples
    loss2 = sums_reshaped.mean() / (2 * eps)

    # Reshape and compute the mean for n_particles
    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0) * m

    # Compute the total loss
    loss = loss1 + loss2 

    return loss.mean(), loss1.mean(), loss2.mean()

'''
def mcsm_loss_optimized(scorenet, samples, n_particles=1, m=20, eps=1e-3):
    N = samples.shape[0]
    scores = scorenet(samples)

    # Compute the first term of the loss
    loss1 = torch.sum(scores * scores, dim=tuple(range(1, scores.dim()))) / 2.0

    # Generate all m random vectors at once
    b = torch.randn(m, N, *samples.shape[1:]).to(samples.device)

    # Compute forward and backward scores at once
    samples_eps_b = samples.unsqueeze(0).expand(m, N, *samples.shape[1:])
    forw_back_samples = torch.cat([samples_eps_b + eps * b, samples_eps_b - eps * b], dim=0)
    forw_back_scores = scorenet(forw_back_samples)

    # Reshape and compute sums
    b_reshaped = b.view(m * N, -1)
    forw_back_scores_reshaped = forw_back_scores.view(2 * m * N, -1)
    sums = torch.einsum('bi,bi->b', b_reshaped, forw_back_scores_reshaped[:m * N] - forw_back_scores_reshaped[m * N:])

    # Compute the second term of the loss
    sums_reshaped = sums.view(m, N).mean(dim=0)  # Average over m samples
    loss2 = sums_reshaped.mean() / (2 * eps)

    # Reshape and compute the mean for n_particles
    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    # Compute the total loss
    loss = loss1 + loss2

    return loss.mean(), loss1.mean(), loss2.mean()

'''
def mcsm_forward_loss_optimized(scorenet, samples, n_particles=1, m=20, eps=1e-3):
    
    N = samples.shape[0]
    scores = scorenet(samples)

    # Compute the first term of the loss
    loss1 = torch.sum(scores * scores, dim=tuple(range(1, scores.dim()))) / 2.0

    # Generate all m random vectors at once
    b = torch.randn(m, N, *samples.shape[1:]).to(samples.device)

    # Compute forward and backward scores at once
    samples_eps_b = samples.unsqueeze(0).expand(m, N, *samples.shape[1:])
    forw_samples = samples_eps_b + eps * b
    back_samples = samples_eps_b

    # Concatenate along the batch dimension
    forw_back_samples = torch.cat([forw_samples, back_samples], dim=1)
    forw_back_samples = forw_back_samples.view(-1, *samples.shape[1:])
    print(f"{forw_back_samples.shape}")
    forw_back_scores = scorenet(forw_back_samples)

    # Reshape and compute sums
    b_reshaped = b.view(m * N, -1)
    forw_back_scores_reshaped = forw_back_scores.view(2 * m * N, -1)
    sums = torch.einsum('bi,bi->b', b_reshaped, forw_back_scores_reshaped[:m * N] - forw_back_scores_reshaped[m * N:])

    # Compute the second term of the loss
    sums_reshaped = sums.view(m, N).mean(dim=0)  # Average over m samples
    loss2 = sums_reshaped.mean() / eps

    # Reshape and compute the mean for n_particles
    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0) * m

    # Compute the total loss
    loss = loss1 + loss2 

    return loss.mean(), loss1.mean(), loss2.mean()



def mcsm_backward_loss_optimized(scorenet, samples, n_particles=1, m=20, eps=1e-3):
    
    N = samples.shape[0]
    scores = scorenet(samples)

    # Compute the first term of the loss
    loss1 = torch.sum(scores * scores, dim=tuple(range(1, scores.dim()))) / 2.0

    # Generate all m random vectors at once
    b = torch.randn(m, N, *samples.shape[1:]).to(samples.device)

    # Compute forward and backward scores at once
    samples_eps_b = samples.unsqueeze(0).expand(m, N, *samples.shape[1:])
    forw_samples = samples_eps_b
    back_samples = samples_eps_b - eps * b

    # Concatenate along the batch dimension
    forw_back_samples = torch.cat([forw_samples, back_samples], dim=1)
    forw_back_samples = forw_back_samples.view(-1, *samples.shape[1:])
    print(f"{forw_back_samples.shape}")
    forw_back_scores = scorenet(forw_back_samples)

    # Reshape and compute sums
    b_reshaped = b.view(m * N, -1)
    forw_back_scores_reshaped = forw_back_scores.view(2 * m * N, -1)
    sums = torch.einsum('bi,bi->b', b_reshaped, forw_back_scores_reshaped[:m * N] - forw_back_scores_reshaped[m * N:])

    # Compute the second term of the loss
    sums_reshaped = sums.view(m, N).mean(dim=0)  # Average over m samples
    loss2 = sums_reshaped.mean() / eps

    # Reshape and compute the mean for n_particles
    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0) * m

    # Compute the total loss
    loss = loss1 + loss2 

    return loss.mean(), loss1.mean(), loss2.mean()



def mcsm_loss(scorenet, samples, n_particles=1, m=20, eps=1e-3):

    list_indices = [x for x in range(1, len(list(samples.shape)))]
    N = samples.shape[0]
    scores = scorenet(samples)

    loss1 = torch.sum(scores * scores, dim=list_indices) / 2.0
    sums = 0.0

    for i in range(m):
        b = torch.randn_like(samples)
        forw = scorenet(samples + eps * b)
        back = scorenet(samples - eps * b)
        sums += torch.einsum('bi,bi->b', (b.view(N, -1),
                             (forw - back).view(N, -1)))

    loss2 = sums.mean() / (2 * m * eps)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2

    return loss.mean(), loss1.mean(), loss2.mean()


'''
class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()

    def forward(self, x):
        return x

inps = torch.randn((16, 3, 32, 32))
model = NN()
outs = model(inps)

start_time = time.perf_counter()  # or time.time()
a1, b1, c1 = mcsm_loss(model, outs)
end_time = time.perf_counter()  # or time.time()

elapsed_time = end_time - start_time
print(f"Time taken by original loss: {elapsed_time} seconds")


start_time = time.perf_counter()  # or time.time()
a2, b2, c2 = mcsm_loss_optimized(model, outs)
end_time = time.perf_counter()  # or time.time()

elapsed_time = end_time - start_time
print(f"Time taken by optimized loss (C): {elapsed_time} seconds")

start_time = time.perf_counter()  # or time.time()
a3, b3, c3 = mcsm_forward_loss_optimized(model, outs)
end_time = time.perf_counter()  # or time.time()

elapsed_time = end_time - start_time
print(f"Time taken by optimized loss (F): {elapsed_time} seconds")

start_time = time.perf_counter()  # or time.time()
a4, b4, c4 = mcsm_backward_loss_optimized(model, outs)
end_time = time.perf_counter()  # or time.time()

elapsed_time = end_time - start_time
print(f"Time taken by optimized loss (B): {elapsed_time} seconds")

print(f"{a1.detach().cpu().numpy(), a2.detach().cpu().numpy(), a3.detach().cpu().numpy(), a4.detach().cpu().numpy()}")
print(f"{b1.detach().cpu().numpy(), b2.detach().cpu().numpy(), b3.detach().cpu().numpy(), b4.detach().cpu().numpy()}")
print(f"{c1.detach().cpu().numpy(), c2.detach().cpu().numpy(), c3.detach().cpu().numpy(), c4.detach().cpu().numpy()}")
'''