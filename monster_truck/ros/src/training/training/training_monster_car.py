import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import mujoco as mj
import mujoco.viewer as viewer
import time, math, numpy as np

from dataclasses import dataclass

#from rclpy.node import Node

@dataclass
class trainingConfig:
    # ------------------ config ------------------
    CTRL_DT   = 0.4
    DURATION  = 60.0
    U_MIN, U_MAX = -1.0, 1.0
    RTF      = 1.0
    REFRESH_HZ = 5
    MODEL_XML = "monstertruck.xml"


#class training_car(Node):
class training_car():
    def __init__(self):
        super().__init__('lqr_node')

        # ------------------ model + sensors ------------------
        model = mj.MjModel.from_xml_path(trainingConfig.MODEL_XML)
        data = mj.MjData(model)
        mj.mj_resetData(model, data); mj.mj_forward(model, data)
        data.ctrl[:] = 0.0

        sim_dt        = float(model.opt.timestep)
        steps_per_cmd = max(1, int(round(trainingConfig.CTRL_DT / sim_dt)))
        free_j = next(j for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE)
        qadr   = model.jnt_qposadr[free_j] + 3  # qw,qx,qy,qz start

        gyro_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if gyro_id < 0:
            raise RuntimeError("imu_gyro sensor not found in model XML")
        gyro_adr = model.sensor_adr[gyro_id]

        acc_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "imu_acc")
        if acc_id < 0:
            raise RuntimeError("imu_acc sensor not found in model XML")
        acc_adr = model.sensor_adr[acc_id]

        # ------------------ logs ------------------
        t_log         = []
        pitch_log     = []
        flip_rel_log  = []
        u_log         = []
        rate_log      = []
        acc_log       = []
        vz_log        = []
        vx_log        = []

        last_refresh_wall = time.perf_counter()

        t0_sim   = data.time
        t0_wall  = time.perf_counter()
        next_cmd = t0_sim
        u = 0.0

        # for angle unwrapping & reference
        prev_theta = None
        prev_theta_unwrapped = 0.0
        theta0 = None  # reference angle at t=0 (so flip_rel starts near 0)



    def plotting(self):
        # ------------------ plotting ------------------
        plt.ioff()
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 8), sharex=True)

        (line_pitch,)   = ax1.plot([], [], lw=1.4)
        (line_flip,)    = ax2.plot([], [], lw=1.4)
        (line_u,)       = ax3.plot([], [], lw=1.4)
        (line_rate,)    = ax4.plot([], [], lw=1.4)
        (line_acc,)     = ax5.plot([], [], lw=1.0)

        ax1.set_ylabel("Euler pitch [rad]")
        ax1.set_ylim(-np.pi, np.pi); ax1.grid(True, linewidth=0.3)

        ax2.set_ylabel("flip angle rel [rad]")
        ax2.set_ylim(-3.5, 3.5); ax2.grid(True, linewidth=0.3)

        ax3.set_ylabel("u")
        ax3.set_ylim(trainingConfig.U_MIN-0.1, trainingConfig.U_MAX+0.1); ax3.grid(True, linewidth=0.3)

        ax4.set_ylabel("pitch rate [rad/s]")
        ax4.set_ylim(-10, 10); ax4.grid(True, linewidth=0.3)

        ax5.set_ylabel("acc imu")
        ax5.set_ylim(-50, 50); ax5.set_xlabel("time [s]"); ax5.grid(True, linewidth=0.3)

        # small inset axis for the up-vector (v_z, v_x)
        ax_up = fig.add_axes([0.45, 0.48, 0.18, 0.18])  # [left, bottom, width, height]
        theta_circle = np.linspace(-np.pi, np.pi, 200)
        ax_up.plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', linewidth=0.8)
        (line_upvec,) = ax_up.plot([], [], 'b-', lw=1.4)
        ax_up.set_aspect('equal', adjustable='box')
        ax_up.set_xlim(-1.1, 1.1)
        ax_up.set_ylim(-1.1, 1.1)
        ax_up.set_xlabel("v_z (up/down)")
        ax_up.set_ylabel("v_x (fwd/back)")
        ax_up.set_title("Up-vector on circle", fontsize=9)

        fig.tight_layout()
        display(fig)


def main():
    #with viewer.launch_passive(m, data) as v:
    while data.time - t0_sim < DURATION:
        # --- control update ---
        if data.time >= next_cmd:
            t_rel = data.time - t0_sim
            #u = float(np.clip(flip_policy(t_rel, U_MIN, U_MAX), U_MIN, U_MAX))
            u = float(np.random.uniform(U_MIN, U_MAX))
            data.ctrl[:] = u
            next_cmd += CTRL_DT

        # --- physics ---
        mj.mj_step(m, data)

        # --- state & orientation ---
        qw, qx, qy, qz = data.qpos[qadr:qadr+4]
        R, euler_pitch = quat_to_R_and_pitch(qw, qx, qy, qz)

        # body "up" vector in world coordinates = 3rd column of R
        up_x, up_y, up_z = R[0, 2], R[1, 2], R[2, 2]

        # angle of up vector in (z,x) plane: atan2(v_x, v_z) in [-pi, pi]
        theta = math.atan2(up_x, up_z)

        # unwrap over time to avoid jump at ±pi
        prev_theta, theta_unwrapped = unwrap_angle(prev_theta, prev_theta_unwrapped, theta)
        prev_theta_unwrapped = theta_unwrapped

        # define reference at the first step so car starts at ~0
        if theta0 is None:
            theta0 = theta_unwrapped
        flip_rel = theta_unwrapped - theta0  # this is your "progress" angle

        # IMU signals
        gyro = data.sensordata[gyro_adr:gyro_adr+3]
        pitch_rate_imu = float(gyro[1])

        acc = data.sensordata[acc_adr:acc_adr+3]
        acc_imu = float(acc[0])

        t_rel = data.time - t0_sim

        # --- log ---
        t_log.append(t_rel)
        pitch_log.append(euler_pitch)
        flip_rel_log.append(flip_rel)
        u_log.append(u)
        rate_log.append(pitch_rate_imu)
        acc_log.append(acc_imu)
        vz_log.append(up_z)
        vx_log.append(up_x)

        # --- plotting ---
        now = time.perf_counter()
        if now - last_refresh_wall >= 1.0 / REFRESH_HZ:
            line_pitch.set_data(t_log, pitch_log)
            line_flip.set_data(t_log, flip_rel_log)
            line_u.set_data(t_log, u_log)
            line_rate.set_data(t_log, rate_log)
            line_acc.set_data(t_log, acc_log)

            line_upvec.set_data(vz_log, vx_log)

            ax1.set_xlim(0.0, max(2.0, t_rel))
            clear_output(wait=True)
            display(fig)
            last_refresh_wall = now

        # --- real-time pacing ---
        sim_elapsed  = data.time - t0_sim
        target_wall  = t0_wall + sim_elapsed / max(1e-9, RTF)
        sleep_needed = target_wall - time.perf_counter()
        if sleep_needed > 0:
            time.sleep(min(sleep_needed, 0.01))

            #v.sync()

    # final draw
    line_pitch.set_data(t_log, pitch_log)
    line_flip.set_data(t_log, flip_rel_log)
    line_u.set_data(t_log, u_log)
    line_rate.set_data(t_log, rate_log)
    line_acc.set_data(t_log, acc_log)
    line_upvec.set_data(vz_log, vx_log)

    ax1.set_xlim(0.0, max(2.0, t_log[-1] if t_log else 2.0))
    clear_output(wait=True)
    display(fig)

    print(f"Done. Samples: {len(t_log)}  Sim time: {t_log[-1]:.3f}s")


if __name__ == "__main__":
    main()