import torch
import numpy as np
from pathlib import Path

from bimanual_imitation.algorithms.imitate_diffusion_state import ImitateDiffusionState
from bimanual_imitation.algorithms.configs import DiffusionStateParamConfig

task_name = "snackbox_push"

# Path to your checkpoint
CHECKPOINT_PATH = f"/juno/u/oliviayl/repos/cross_embodiment/state-imitation/test_results/diffusion_state/{task_name}/checkpoints/iteration_030.ckpt"

device = "cuda"

obs_tensor = torch.tensor(
    [[-1.49789786,  0.79693413,  0.22912885, -1.70729935,  1.24776483,
         1.91593158,  0.37197036, -0.54166758,  0.47487646,  0.61238086,
         0.32505634, -0.1889299 ,  0.38526362,  0.54325885,  0.29367885,
         0.28225803,  0.31798214,  0.74458116,  0.43271869,  1.20436382,
         0.0683949 ,  0.97561121,  0.04778602,  0.29934135, -0.4346239 ,
         0.24744351,  0.92640716,  0.32220504,  0.16284432,  0.10693634,
        -0.89427841, -0.44464043, -0.05060767,  0.49824476, -0.4474366 ,
         0.89045817,  0.08297466, -0.26869756,  0.00817013,  0.09684616,
        -0.99526584,  0.19295141,  0.        ,  0.        ,  0.        ,
         1.        ],
       [-1.50004172,  0.79095072,  0.23799801, -1.71438479,  1.25088644,
         1.90641153,  0.37678933, -0.54798466,  0.47288358,  0.60022521,
         0.31773087, -0.19343498,  0.38374218,  0.53041071,  0.28587559,
         0.28595641,  0.31475127,  0.73143935,  0.42474851,  1.20242834,
         0.06693083,  0.9869132 ,  0.04692607,  0.30127066, -0.43209994,
         0.24921528,  0.92705154,  0.32126844,  0.16180733,  0.10573758,
        -0.89427841, -0.44464043, -0.05060767,  0.49824476, -0.4474366 ,
         0.89045817,  0.08297466, -0.26869756,  0.00817013,  0.09684616,
        -0.99526584,  0.19295141,  0.        ,  0.        ,  0.        ,
         1.        ],
       [-1.50218546,  0.7849673 ,  0.24686718, -1.72147024,  1.25400805,
         1.89689136,  0.38160831, -0.55430168,  0.4708907 ,  0.5880695 ,
         0.31040543, -0.19794008,  0.38222075,  0.51756251,  0.2780723 ,
         0.28965476,  0.31152037,  0.71829748,  0.41677836,  1.20049286,
         0.06546676,  0.9982152 ,  0.04606611,  0.30317622, -0.42953795,
         0.25097701,  0.92769033,  0.32036176,  0.16073738,  0.10451054,
        -0.89427841, -0.44464043, -0.05060767,  0.49824476, -0.4474366 ,
         0.89045817,  0.08297466, -0.26869756,  0.00817013,  0.09684616,
        -0.99526584,  0.19295141,  0.        ,  0.        ,  0.        ,
         1.        ],
       [-1.50432932,  0.77898389,  0.25573635, -1.72855568,  1.25712967,
         1.88737118,  0.38642728, -0.56061876,  0.46889782,  0.57591385,
         0.30307996, -0.20244516,  0.38069934,  0.50471437,  0.27026904,
         0.29335311,  0.30828947,  0.70515567,  0.4088082 ,  1.19855738,
         0.06400269,  1.00951719,  0.04520616,  0.30505747, -0.42693806,
         0.2527279 ,  0.92832363,  0.31948486,  0.15963379,  0.10325588,
        -0.89427841, -0.44464043, -0.05060767,  0.49824476, -0.4474366 ,
         0.89045817,  0.08297466, -0.26869756,  0.00817013,  0.09684616,
        -0.99526584,  0.19295141,  0.        ,  0.        ,  0.        ,
         1.        ]],
    dtype=torch.float32,
).float().unsqueeze(0).to(device)
obs_tensor = obs_tensor.flatten(start_dim=1)
assert obs_tensor.shape == (1, 4 * 46), f"Expected (1, {4 * 46}), got {obs_tensor.shape}"

# Optional: Ground truth action for comparison (if available)
ground_truth_action = np.array([[-0.02358188, -0.06581737,  0.09756082, -0.07793972,  0.03433799,
        -0.10472135,  0.05300867, -0.06948754, -0.02192169, -0.13371238,
        -0.08058006, -0.04955597, -0.01673572, -0.14132991, -0.08583602,
         0.04068197, -0.03553984, -0.14456026, -0.08767176, -0.02129043,
        -0.01610478,  0.12432187, -0.00945951],
       [-0.02143807, -0.05983397,  0.08869164, -0.07085429,  0.03121635,
        -0.09520122,  0.0481897 , -0.06317049, -0.01992881, -0.12155671,
        -0.0732546 , -0.04505089, -0.01521429, -0.12848173, -0.07803275,
         0.03698361, -0.03230894, -0.13141842, -0.0797016 , -0.01935494,
        -0.01464071,  0.11301988, -0.00859955],
       [-0.01929426, -0.05385058,  0.07982248, -0.06376886,  0.02809472,
        -0.0856811 ,  0.04337073, -0.05685344, -0.01793593, -0.10940104,
        -0.06592914, -0.0405458 , -0.01369286, -0.11563356, -0.07022947,
         0.03328525, -0.02907805, -0.11827658, -0.07173145, -0.01741944,
        -0.01317664,  0.1017179 , -0.0077396 ],
       [-0.01715046, -0.04786718,  0.07095332, -0.05668343,  0.02497308,
        -0.07616097,  0.03855176, -0.05053639, -0.01594304, -0.09724537,
        -0.05860368, -0.03604071, -0.01217143, -0.10278539, -0.06242619,
         0.02958689, -0.02584716, -0.10513474, -0.06376129, -0.01548395,
        -0.01171257,  0.09041591, -0.00687964],
       [-0.01500665, -0.04188378,  0.06208415, -0.049598  ,  0.02185145,
        -0.06664085,  0.03373279, -0.04421934, -0.01395016, -0.0850897 ,
        -0.05127822, -0.03153562, -0.01065001, -0.08993722, -0.05462292,
         0.02588853, -0.02261626, -0.09199289, -0.05579112, -0.01354846,
        -0.01024849,  0.07911392, -0.00601969],
       [-0.01286284, -0.03590038,  0.05321499, -0.04251257,  0.01872981,
        -0.05712073,  0.02891382, -0.0379023 , -0.01195728, -0.07293402,
        -0.04395276, -0.02703053, -0.00912858, -0.07708904, -0.04681965,
         0.02219017, -0.01938537, -0.07885105, -0.04782096, -0.01161296,
        -0.00878442,  0.06781193, -0.00515973],
       [-0.01071903, -0.02991699,  0.04434582, -0.03542715,  0.01560818,
        -0.04760061,  0.02409485, -0.03158525, -0.0099644 , -0.06077836,
        -0.0366273 , -0.02252544, -0.00760715, -0.06424087, -0.03901637,
         0.01849181, -0.01615447, -0.06570921, -0.0398508 , -0.00967747,
        -0.00732035,  0.05650994, -0.00429978],
       [-0.00857523, -0.02393359,  0.03547666, -0.02834172,  0.01248654,
        -0.03808049,  0.01927588, -0.0252682 , -0.00797152, -0.04862269,
        -0.02930184, -0.01802035, -0.00608572, -0.05139269, -0.0312131 ,
         0.01479344, -0.01292358, -0.05256737, -0.03188064, -0.00774198,
        -0.00585628,  0.04520795, -0.00343982]], dtype=np.float32)  # Uncomment and fill if available

# Config parameters that match your training configuration
PRED_HORIZON = 8
OBS_HORIZON = 4
ACTION_HORIZON = 4

# Initialize policy
config = DiffusionStateParamConfig(
    pred_horizon=PRED_HORIZON,
    obs_horizon=OBS_HORIZON,
    action_horizon=ACTION_HORIZON,
    batch_size=128,
    num_diffusion_iters=50,
    opt_learning_rate=1e-4,
    opt_weight_decay=1e-6,
    lr_warmup_steps=500,
    lr_scheduler="constant"
)

# Create policy and initialize
policy = ImitateDiffusionState(task_name=task_name)
policy.run = lambda: None  # Override run method
policy.env_name = "quad_insert_a0o0"
policy.init_params(config)

# Load checkpoint
checkpoint_state = torch.load(CHECKPOINT_PATH, map_location=policy.device)

# Handle different checkpoint formats
if 'model_state_dict' in checkpoint_state:
    policy._noise_pred_net.load_state_dict(checkpoint_state['model_state_dict'])
elif isinstance(checkpoint_state, dict) and not 'model_state_dict' in checkpoint_state:
    policy._noise_pred_net.load_state_dict(checkpoint_state)
else:
    policy._noise_pred_net.load_state_dict(checkpoint_state['model_state'])

# Set to evaluation mode
policy._noise_pred_net.eval()
policy._ema_noise_pred_net = policy._noise_pred_net

# Run inference
print("Running inference...")
with torch.no_grad():
    action = policy.bc_policy_fn(obs_tensor)

# Convert to numpy
sample_obs = obs_tensor.cpu().numpy()
predicted_action = action.cpu().numpy()

# Print results
print(f"Observation shape: {sample_obs.shape}")
print(f"Predicted action shape: {predicted_action.shape}")
print(f"Predicted action: {predicted_action.squeeze()}")

# Compare with ground truth if available
if ground_truth_action is not None:
    mse = np.mean((ground_truth_action - predicted_action.squeeze())**2)
    print(f"\nGround truth action: {ground_truth_action}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.6f}")