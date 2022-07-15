import hydra
from torch.nn.parallel import DistributedDataParallel as DDP


# Scripts to initialize different agents - used in training scripts
def init_pli(cfg, device, rank):
    # Initialize the model
    model = hydra.utils.instantiate(cfg.model,
                                    input_dim=cfg.pos_dim*2, # For dog and box
                                    action_dim=cfg.action_dim,
                                    hidden_dim=cfg.hidden_dim).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)


    # Initialize the optimizer
    # parameters = list(encoder.parameters()) + list(trans.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = model.parameters(),
                                        lr = cfg.lr,
                                        weight_decay = cfg.weight_decay)

    # Initialize the total agent
    agent = hydra.utils.instantiate(cfg.agent,
                                    model=model,
                                    optimizer=optimizer)

    agent.to(device)

    return agent

def init_cpn(cfg, device, rank):
    # Initialize the encoder and the trans
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    trans = hydra.utils.instantiate(cfg.trans,
                                    z_dim=cfg.z_dim,
                                    action_dim=cfg.action_dim).to(device)
    encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False) # To fix the inplace error https://github.com/pytorch/pytorch/issues/22095 
    trans = DDP(trans, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # Initialize the optimizer
    parameters = list(encoder.parameters()) + list(trans.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = parameters,
                                        lr = cfg.lr,
                                        weight_decay = cfg.weight_decay)

    # Initialize the total agent
    agent = hydra.utils.instantiate(cfg.agent,
                                    encoder=encoder,
                                    trans=trans,
                                    optimizer=optimizer)
    agent.to(device)

    return agent 