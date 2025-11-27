import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import mujoco as mj
import mujoco.viewer as viewer
import time, math, numpy as np

from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from sensor_msgs.msgs import Imu

class mujoco_model(Node):
    def __init__(self):
        super().__init__('mujoco_model_node')

        MODEL_XML = "monstertruck.xml"

        # =========================================================================================
        #                                      model + sensors 
        # =========================================================================================
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

        # =========================================================================================
        #                          Publishers
        # =========================================================================================
        self.pub_car_imu = self.create_publisher(Imu, "car_imu", 1)


    def car_model(self):
        x=1
