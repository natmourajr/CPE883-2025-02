if __name__ == '__main__':
    import sys
    import torch
    sys.path.append('/3W')
    sys.path.append('/')

    from loader import Loader3W
    from .ts_diffusion2 import TSDiffusion

    ld = Loader3W()
    ld.load_stats()

    ts_diffusion = TSDiffusion(
        in_channels=17,
        latent_dim=256,
        model_dim=256,
        static_dim=7,
        hidden_dim=1024,
        num_steps=1000
        )
    try:
        ts_diffusion = ts_diffusion.load(
            'state.pt',
            in_channels=17,
            latent_dim=256,
            model_dim=256,
            static_dim=7,
            hidden_dim=1024,
            num_steps=1000            
        )
    except:
        print('Sem arquivo de estados do Torch. Será criado um novo arquivo.')
    ts_diffusion = ts_diffusion.to(device=torch.device('cuda' if torch.cuda.is_available else 'cpu'))
    ts_diffusion.train3W(
        window_size=15,
        batch_size=2000,
        epochs=10,
    )
    ts_diffusion.save('state.pt')
