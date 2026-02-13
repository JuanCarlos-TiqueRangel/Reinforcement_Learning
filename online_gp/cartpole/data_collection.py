# ============================
# Cell 2 — Random data collection + PIL render (NO teleport training)
#   ✅ Uses info["terminal_obs"] when respawned=True (keeps real physics transition)
#   ✅ Never trains on reset teleport
#   ✅ Collects dataset in float64 (GP training friendly)
# ============================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
from PIL import Image, ImageDraw

NP_DTYPE = np.float64  # dataset dtype (GP side)

# ------------------------------------------------------------
# Pure-PIL renderer: (x, theta) -> RGB frame
# ------------------------------------------------------------
def render_cartpole_frame_from_state(
    x, theta,
    x_threshold=2.4,
    W=720, H=450,
    cart_width=70,
    cart_height=35,
    pole_length_px=180,
    wheel_radius=12,
):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    track_y = int(H * 0.70)
    draw.line((int(W * 0.10), track_y, int(W * 0.90), track_y), fill=(40, 40, 40), width=4)

    x = float(x)
    x_clipped = max(-x_threshold, min(x_threshold, x))
    t = (x_clipped + x_threshold) / (2.0 * x_threshold)
    cart_x = int(W * 0.10 + t * (W * 0.80))
    cart_y = track_y - cart_height // 2

    x0 = cart_x - cart_width // 2
    y0 = cart_y - cart_height // 2
    x1 = cart_x + cart_width // 2
    y1 = cart_y + cart_height // 2
    draw.rectangle((x0, y0, x1, y1), fill=(120, 160, 230), outline=(0, 0, 0))

    w1x = cart_x - int(cart_width * 0.25)
    w2x = cart_x + int(cart_width * 0.25)
    wy = track_y
    for wx in [w1x, w2x]:
        draw.ellipse((wx - wheel_radius, wy - wheel_radius, wx + wheel_radius, wy + wheel_radius),
                     fill=(60, 60, 60), outline=(0, 0, 0))

    theta = float(theta)
    pivot_x = cart_x
    pivot_y = y0  # cart top
    tip_x = pivot_x + int(pole_length_px * np.sin(theta))
    tip_y = pivot_y - int(pole_length_px * np.cos(theta))
    draw.line((pivot_x, pivot_y, tip_x, tip_y), fill=(200, 60, 60), width=8)
    draw.ellipse((pivot_x - 6, pivot_y - 6, pivot_x + 6, pivot_y + 6),
                 fill=(0, 0, 0), outline=(0, 0, 0))

    return np.array(img, dtype=np.uint8)

def show_gif(frames_uint8, fps=20):
    fig = plt.figure(figsize=(7.2, 4.5))
    plt.axis("off")
    im = plt.imshow(frames_uint8[0])

    def animate(i):
        im.set_data(frames_uint8[i])
        return (im,)

    anim = animation.FuncAnimation(fig, animate, frames=len(frames_uint8), interval=1000 / fps)
    plt.close(fig)
    display(HTML(anim.to_jshtml()))

# ------------------------------------------------------------
# Random collection (keeps REAL terminal transition when respawned)
# ------------------------------------------------------------
def collect_random_dataset_with_render(
    n_steps=1500,
    seed=0,
    resize=(720, 450),
    fps=20,
    edge_respawn=True,
    respawn_penalty=-2.0,
    verbose=False,
):
    env = make_env(
        render_mode=None,
        seed=seed,
        start_down=True,
        edge_respawn=edge_respawn,
        respawn_penalty=respawn_penalty,
    )

    obs, _ = env.reset(seed=seed)
    x, xdot, th, thdot = obs_to_state(obs)

    frames = []
    traj = {
        "x": [], "xdot": [], "theta": [], "thetadot": [],
        "u": [], "reward": [], "respawned": [],
    }

    X_list, ydx_list, ydxdot_list, ydth_list, ydthdot_list = [], [], [], [], []

    for t in range(n_steps):
        u = float(env.action_space.sample()[0])

        # frame from current state BEFORE stepping
        frames.append(render_cartpole_frame_from_state(x, th, W=resize[0], H=resize[1]))

        # step
        obs2, r, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))

        respawned = bool(info.get("respawned", False))

        # use TRUE physics next_obs if respawned
        if respawned and ("terminal_obs" in info):
            obs2_for_training = info["terminal_obs"]
        else:
            obs2_for_training = obs2

        x2, xdot2, th2, thdot2 = obs_to_state(obs2_for_training)

        # logs
        traj["x"].append(x); traj["xdot"].append(xdot); traj["theta"].append(th); traj["thetadot"].append(thdot)
        traj["u"].append(u); traj["reward"].append(float(r)); traj["respawned"].append(respawned)

        # ALWAYS add a valid transition (no teleports)
        X = state_to_features(x, xdot, th, thdot, u, dtype=NP_DTYPE).astype(NP_DTYPE)
        dx = float(x2 - x)
        dxdot = float(xdot2 - xdot)
        dth = float(wrap_pi(th2 - th))
        dthdot = float(thdot2 - thdot)

        X_list.append(X)
        ydx_list.append([dx])
        ydxdot_list.append([dxdot])
        ydth_list.append([dth])
        ydthdot_list.append([dthdot])

        if verbose and (t % 200 == 0):
            print(f"[t={t}] respawned={respawned}  x={x:.2f} th={th:.2f} u={u:+.2f}")

        # advance CONTROL state using returned obs2 (which is reset_obs if respawned)
        x, xdot, th, thdot = obs_to_state(obs2)

    env.close()

    X0 = np.asarray(X_list, dtype=NP_DTYPE)
    Ydx0 = np.asarray(ydx_list, dtype=NP_DTYPE)
    Ydxdot0 = np.asarray(ydxdot_list, dtype=NP_DTYPE)
    Ydth0 = np.asarray(ydth_list, dtype=NP_DTYPE)
    Ydthdot0 = np.asarray(ydthdot_list, dtype=NP_DTYPE)

    return X0, Ydx0, Ydxdot0, Ydth0, Ydthdot0, frames, traj

# ------------------------------------------------------------
# Run data collection + visualize
# ------------------------------------------------------------
N_STEPS_RANDOM = 300
SEED_RANDOM = 0
FPS = 20
RECORD_RGB = True

X0, Ydx0, Ydxdot0, Ydth0, Ydthdot0, frames0, traj0 = collect_random_dataset_with_render(
    n_steps=N_STEPS_RANDOM,
    seed=SEED_RANDOM,
    resize=(720, 450),
    fps=FPS,
    edge_respawn=True,
    respawn_penalty=-2.0,
    verbose=False,
)

print("✅ Random dataset collected (NO teleport training)")
print("X0:", X0.shape)
print("Ydx0:", Ydx0.shape, " Ydxdot0:", Ydxdot0.shape, " Ydth0:", Ydth0.shape, " Ydthdot0:", Ydthdot0.shape)

if RECORD_RGB:
    show_gif(frames0, fps=FPS)

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
x_arr = np.array(traj0["x"])
xdot_arr = np.array(traj0["xdot"])
th_arr = np.array(traj0["theta"])
thdot_arr = np.array(traj0["thetadot"])
u_arr = np.array(traj0["u"])
rew_arr = np.array(traj0["reward"])
resp_arr = np.array(traj0["respawned"])
t = np.arange(len(x_arr)) * 0.02

fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
axs[0].plot(t, x_arr); axs[0].set_ylabel("x")
axs[1].plot(t, xdot_arr); axs[1].set_ylabel("xdot")
axs[2].plot(t, th_arr); axs[2].set_ylabel("theta")
axs[3].plot(t, thdot_arr); axs[3].set_ylabel("thetadot")
axs[4].plot(t, u_arr); axs[4].set_ylabel("u"); axs[4].set_xlabel("time (s)")
for ax in axs: ax.grid(True, alpha=0.25)
fig.suptitle("Random rollout trajectories (terminal transitions kept, teleports excluded)")
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(x_arr, xdot_arr, linewidth=1.0)
plt.scatter(x_arr[resp_arr], xdot_arr[resp_arr], s=18, label="respawned step", alpha=0.8)
plt.xlabel("x"); plt.ylabel("xdot")
plt.title("Phase plot: x vs xdot (random rollout)")
plt.grid(True, alpha=0.25)
plt.legend()
plt.show()
