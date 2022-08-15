import hydra
from torch.nn.parallel import DistributedDataParallel as DDP

def init_agent(cfg, device, rank, dataset=None):
    if cfg.agent_type == 'cpn': # Contrastive Predictive Network 
        agent = init_cpn(cfg, device, rank)
    elif cfg.agent_type == 'sbfd': # State Based Forward Dynamics
        agent = init_sbfd(cfg, device, rank) # For SBFD the encoder is different
    elif cfg.agent_type == 'pli':
        agent = init_pli(cfg, device, rank)
    elif cfg.agent_type == 'bc':
        agent = init_bc(cfg, device, rank)
    elif cfg.agent_type == 'diffusion':
        agent = init_diffusion(cfg, device, rank, dataset)

    return agent

# Initialize the diffusion agent - TODO
def init_diffusion(cfg, device, rank, dataset):
    # Initialize the model
    eps_model = hydra.utils.instantiate(cfg.eps_model,
                                        input_dim=(cfg.pos_dim*2)*2+cfg.action_dim+1, # Two positions, 1 action and diffusion time
                                        hidden_dim=cfg.hidden_dim,
                                        output_dim=cfg.pos_dim*2) .to(device) # This will give an error - it should
    eps_model = DDP(eps_model, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # Initialize the optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = eps_model.parameters(),
                                        lr = cfg.lr,
                                        weight_decay = cfg.weight_decay)

    # Initialize the total agent
    agent = hydra.utils.instantiate(cfg.agent,
                                    eps_model=eps_model,
                                    optimizer=optimizer,
                                    dataset=dataset,
                                    checkpoint_dir=cfg.checkpoint_dir)
    agent.to(device)

    return agent

# Scripts to initialize different agents - used in training scripts
def init_pli(cfg, device, rank):
    # Initialize the model
    model = hydra.utils.instantiate(cfg.pli_model,
                                    input_dim=cfg.pos_dim*2, # For dog and box
                                    action_dim=cfg.action_dim,
                                    hidden_dim=cfg.hidden_dim).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)


    # Initialize the optimizer
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

def init_bc(cfg, device, rank):
    # Initialize the model
    if cfg.agent.predict_dist:
        model = hydra.utils.instantiate(cfg.bc_dist_model,
                                        state_dim=cfg.pos_dim*2,
                                        hidden_dim=cfg.hidden_dim).to(device)
    else:
        model = hydra.utils.instantiate(cfg.bc_reg_model,
                                        state_dim=cfg.pos_dim*2,
                                        action_dim=cfg.action_dim,
                                        hidden_dim=cfg.hidden_dim).to(device)
    
    model = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params=model.parameters(),
                                        lr=cfg.lr,
                                        weight_decay=cfg.weight_decay)

    # Initialize the total agent 
    agent = hydra.utils.instantiate(cfg.agent,
                                    model=model,
                                    optimizer=optimizer)
    agent.to(device) # Agent has its own to method

    return agent

def init_cpn(cfg, device, rank): # This can be used for sbfd agents as well
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

def init_sbfd(cfg, device, rank): # This can be used for sbfd agents as well
    # Initialize the encoder and the trans
    if cfg.agent.use_encoder == False:
        cfg.z_dim = cfg.pos_dim*2
        print('z_dim: {}'.format(cfg.z_dim))
    pos_encoder = hydra.utils.instantiate(cfg.pos_encoder,
                                         input_dim=cfg.pos_dim*2,
                                         hidden_dim=cfg.hidden_dim,
                                         out_dim=cfg.z_dim).to(device)
    trans = hydra.utils.instantiate(cfg.trans,
                                    z_dim=cfg.z_dim,
                                    action_dim=cfg.action_dim).to(device)
    pos_encoder = DDP(pos_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False) # To fix the inplace error https://github.com/pytorch/pytorch/issues/22095 
    trans = DDP(trans, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # Initialize the optimizer
    parameters = list(pos_encoder.parameters()) + list(trans.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = parameters,
                                        lr = cfg.lr,
                                        weight_decay = cfg.weight_decay)

    # Initialize the total agent
    agent = hydra.utils.instantiate(cfg.agent,
                                    pos_encoder=pos_encoder,
                                    trans=trans,
                                    optimizer=optimizer)
    agent.to(device)

    return agent     