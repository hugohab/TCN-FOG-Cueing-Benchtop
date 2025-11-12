# stream_fog.py
import torch
from pytorch_tcn.buffer import BufferIO
from pytorch_tcn import TCN

ckpt = torch.load('tcn_fog_best.pt', map_location='cpu')
mean, std = ckpt['mean'], ckpt['std']

def normalize_block(x_block):         # x_block: [1,6,Tb]
    return (x_block - mean.view(1,-1,1)) / std.view(1,-1,1)

# same architecture as training
model = TCN(
    num_inputs=6,
    num_channels=[32,32,64,64],
    kernel_size=5,
    dilations=[1,2,4,8],
    dropout=0.1,
    causal=True,
    use_norm='weight_norm',
    activation='relu',
    use_skip_connections=True,
    input_shape='NCL',
    output_projection=1,
    output_activation=None,
)
model.load_state_dict(ckpt['state_dict'])
model.eval()

buffer_io = BufferIO()  # internal buffers get used for the first block
thr = 0.5               # initial threshold; tune later

with torch.no_grad():
    for x_block_np in daq_blocks():               # yields np.array shape (6, Tb) at 60 Hz
        x = torch.from_numpy(x_block_np).float().unsqueeze(0)  # [1,6,Tb]
        x = normalize_block(x)
        logits = model(x, inference=True, buffer_io=buffer_io)  # [1,1,Tb]
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0)     # [Tb]
        fog_mask = (probs >= thr)

        # Post-processing (debounce)
        # example: require at least 6 consecutive frames (~100 ms) to declare FOG
        # ... your FSM / threshold logic here ...

        buffer_io.step()   # rotate buffers after each block
