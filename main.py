import numpy as np
import matplotlib.pyplot as plt
import time
import pybullet as pb
import pybullet_data
from control.matlab import place


USE_VISUALIZATION = False
TIME_STEP = 1 / 240.0
SIM_DURATION = 5.0
INIT_ANGLE = 10.0
TARGET_ANGLE = np.pi
GRAVITY = 9.81
ROD_LENGTH = 0.8
MASS = 1.0


def configure_controller():
    angular_acceleration = GRAVITY / ROD_LENGTH
    inertia_coeff = 1.0 / (MASS * ROD_LENGTH ** 2)

    state_matrix = np.vstack([
        np.array([0.0, 1.0]),
        np.array([angular_acceleration, 0.0])
    ])

    input_matrix = np.vstack([
        np.zeros(1),
        np.array([inertia_coeff])
    ])

    damping_ratio = 0.707
    natural_freq = 2.828

    pole_real = -damping_ratio * natural_freq
    pole_imag = natural_freq * np.sqrt(1 - damping_ratio ** 2)

    desired_poles = np.array([
        pole_real + pole_imag * 1j,
        pole_real - pole_imag * 1j
    ])

    feedback_gains = -place(state_matrix, input_matrix, desired_poles)

    return feedback_gains


def setup_simulation():
    physics_client = pb.connect(pb.GUI if USE_VISUALIZATION else pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -GRAVITY)

    ground_id = pb.loadURDF("plane.urdf")
    pendulum_id = pb.loadURDF("./pendulum.urdf.xml", useFixedBase=True)

    for link in [1, 2]:
        pb.changeDynamics(pendulum_id, link, linearDamping=0, angularDamping=0)

    pb.resetJointState(pendulum_id, 1, INIT_ANGLE, 0)
    pb.setJointMotorControl2(
        bodyIndex=pendulum_id,
        jointIndex=1,
        controlMode=pb.VELOCITY_CONTROL,
        force=0,
        targetVelocity=0
    )

    return pendulum_id


def run_simulation(pendulum, controller_gains):
    time_points = np.arange(0, SIM_DURATION, TIME_STEP)
    n_steps = len(time_points)

    angle_history = np.zeros(n_steps)
    velocity_history = np.zeros(n_steps)
    torque_history = np.zeros(n_steps)

    for step, current_time in enumerate(time_points):

        joint_state = pb.getJointState(pendulum, 1)
        current_angle, current_velocity = joint_state[0], joint_state[1]
        angle_history[step] = current_angle

        angle_error = current_angle - TARGET_ANGLE
        control_torque = controller_gains[0, 0] * angle_error + controller_gains[0, 1] * current_velocity
        torque_history[step] = control_torque

        pb.setJointMotorControl2(
            bodyIndex=pendulum,
            jointIndex=1,
            controlMode=pb.TORQUE_CONTROL,
            force=control_torque
        )

        pb.stepSimulation()

        updated_velocity = pb.getJointState(pendulum, 1)[1]
        velocity_history[step] = updated_velocity

        if USE_VISUALIZATION:
            time.sleep(TIME_STEP)

    return time_points, angle_history, velocity_history, torque_history


def plot_results(time, angles, velocities, torques):
    _, axes = plt.subplots(3, 1, figsize=(10, 8))

    axes[0].plot(time, angles, 'navy')
    axes[0].plot([time[0], time[-1]], [TARGET_ANGLE, TARGET_ANGLE], 'r--')
    axes[0].grid(True)

    axes[1].plot(time, velocities, 'darkgreen')
    axes[1].grid(True)

    axes[2].plot(time, torques, 'purple')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    gain_matrix = configure_controller()
    pendulum_model = setup_simulation()

    sim_time, angles, velocities, torques = run_simulation(
        pendulum_model,
        gain_matrix
    )

    pb.disconnect()

    plot_results(sim_time, angles, velocities, torques)
