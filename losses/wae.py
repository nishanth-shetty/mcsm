import torch
from losses.mcsm import mcsm_loss, mcsm_backward_loss, mcsm_forward_loss, mcsm_loss_optimized, mcsm_forward_loss_optimized, mcsm_backward_loss_optimized
from losses.sliced_sm import sliced_score_matching, sliced_score_estimation, sliced_score_estimation_vr
import numpy as np


def wae_ssm(encoder, decoder, score, score_opt, X, training=True, n_energy_opt=1, n_particles=1, lam=10):
    z = encoder(X)
    ssm_loss, * \
        _ = sliced_score_estimation_vr(score, z, n_particles=n_particles)
    if training:
        score_opt.zero_grad()
        ssm_loss.backward()
        score_opt.step()
        for i in range(n_energy_opt - 1):
            z = encoder(X)
            ssm_loss, * \
                _ = sliced_score_estimation_vr(
                    score, z, n_particles=n_particles)
            score_opt.zero_grad()
            ssm_loss.backward()
            score_opt.step()

    z = encoder(X)
    decoded_X = decoder(z)
    recon = (X - decoded_X) ** 2
    recon = recon.sum(dim=(1, 2, 3))

    nlogpz = z ** 2 / 2. + np.log(2. * np.pi) / 2.
    nlogpz = nlogpz.sum(dim=-1)

    scores = score(z)
    entropy_surrogate = (scores.detach() * z).sum(dim=-1)

    loss = recon + lam * (nlogpz + entropy_surrogate)

    loss = loss.mean()

    return loss, ssm_loss, recon


def wae_mcsm(encoder, decoder, score, score_opt, X, training=True, n_energy_opt=1, n_particles=1, lam=10, eps=1e-3):
    z = encoder(X)
    mcsm_losses, *_ = mcsm_loss_optimized(score, z, n_particles=n_particles, eps=eps)
    if training:
        score_opt.zero_grad()
        mcsm_losses.backward()
        score_opt.step()
        for i in range(n_energy_opt - 1):
            z = encoder(X)
            mcsm_losses, *_ = mcsm_loss_optimized(score, z, n_particles=n_particles, eps=eps)
            score_opt.zero_grad()
            mcsm_losses.backward()
            score_opt.step()

    z = encoder(X)
    decoded_X = decoder(z)
    recon = (X - decoded_X) ** 2
    recon = recon.sum(dim=(1, 2, 3))

    nlogpz = z ** 2 / 2. + np.log(2. * np.pi) / 2.
    nlogpz = nlogpz.sum(dim=-1)

    scores = score(z)
    entropy_surrogate = (scores.detach() * z).sum(dim=-1)

    loss = recon + lam * (nlogpz + entropy_surrogate)

    loss = loss.mean()

    return loss, mcsm_losses, recon


def wae_mcsm_forward(encoder, decoder, score, score_opt, X, training=True, n_energy_opt=1, n_particles=1, lam=10, eps=1e-3):
    z = encoder(X)
    mcsm_loss, *_ = mcsm_forward_loss_optimized(score, z, n_particles=n_particles, eps=eps)
    if training:
        score_opt.zero_grad()
        mcsm_loss.backward()
        score_opt.step()
        for i in range(n_energy_opt - 1):
            z = encoder(X)
            mcsm_loss, *_ = mcsm_forward_loss_optimized(score, z, n_particles=n_particles, eps=eps)
            score_opt.zero_grad()
            mcsm_loss.backward()
            score_opt.step()

    z = encoder(X)
    decoded_X = decoder(z)
    recon = (X - decoded_X) ** 2
    recon = recon.sum(dim=(1, 2, 3))

    nlogpz = z ** 2 / 2. + np.log(2. * np.pi) / 2.
    nlogpz = nlogpz.sum(dim=-1)

    scores = score(z)
    entropy_surrogate = (scores.detach() * z).sum(dim=-1)

    loss = recon + lam * (nlogpz + entropy_surrogate)

    loss = loss.mean()

    return loss, mcsm_loss, recon


def wae_mcsm_backward(encoder, decoder, score, score_opt, X, training=True, n_energy_opt=1, n_particles=1, lam=10, eps=1e-3):
    z = encoder(X)
    mcsm_loss, *_ = mcsm_backward_loss_optimized(score, z, n_particles=n_particles, eps=eps)
    if training:
        score_opt.zero_grad()
        mcsm_loss.backward()
        score_opt.step()
        for i in range(n_energy_opt - 1):
            z = encoder(X)
            mcsm_loss, *_ = mcsm_backward_loss_optimized(score, z, eps=eps, n_particles=n_particles)
            score_opt.zero_grad()
            mcsm_loss.backward()
            score_opt.step()

    z = encoder(X)
    decoded_X = decoder(z)
    recon = (X - decoded_X) ** 2
    recon = recon.sum(dim=(1, 2, 3))

    nlogpz = z ** 2 / 2. + np.log(2. * np.pi) / 2.
    nlogpz = nlogpz.sum(dim=-1)

    scores = score(z)
    entropy_surrogate = (scores.detach() * z).sum(dim=-1)

    loss = recon + lam * (nlogpz + entropy_surrogate)

    loss = loss.mean()

    return loss, mcsm_loss, recon



def wae_kernel(encoder, decoder, estimator, X, lam=10):
    z = encoder(X)
    decoded_X = decoder(z)
    recon = (X - decoded_X) ** 2
    recon = recon.sum(dim=(1, 2, 3))

    nlogpz = z ** 2 / 2. + np.log(2. * np.pi) / 2.
    nlogpz = nlogpz.sum(dim=-1)

    with torch.no_grad():
        scores = estimator.compute_gradients(z.unsqueeze(0))

    entropy_surrogate = (scores.squeeze(0).detach() * z).sum(dim=-1)

    loss = recon + lam * (nlogpz + entropy_surrogate)

    loss = loss.mean()
    return loss
