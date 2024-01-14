from typing import List
import numpy as np
from utils import Particle
import scipy.stats as stats
import scipy as spy
### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000
W = 0.086255
a = 1.03003
sigma_pos = 0.09586
sigma_angle = 0.036542
k = 4
### 可以在这里写下一些你需要的变量和函数 ###




def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    
    # Find the maximum width and length from the wall information
    max_width = np.max(walls[:, 0])
    min_width = np.min(walls[:, 0])
    max_length = np.max(walls[:, 1])
    min_length = np.min(walls[:, 1])    
    
    # Generate uniformly distributed random values for width and length
    # Create particles with the generated width, length, and equal weights
    for i in range(N):
        width_values = np.random.uniform(min_width, max_width)
        length_values = np.random.uniform(min_length, max_length)
        theta_values = np.random.uniform(-np.pi,np.pi)
        all_particles.append(Particle(width_values,length_values,theta_values,float(1.0/N)))    
            
                
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = np.exp(- W * np.sqrt(np.abs(np.sum(np.square(estimated - gt)))))
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    N = len(particles)
    
    for i in range(N):
        num_particle = particles[i].weight*N*a
        for j in range(int(num_particle)):
            if(len(resampled_particles)>=N):
                break
            resampled_particles.append(Particle(particles[i].position[0],particles[i].position[1],particles[i].theta,1.0/N))
    
    if(N>len(resampled_particles)):
        # Find the maximum width and length from the wall information
        max_width = np.max(walls[:, 0])
        min_width = np.min(walls[:, 0])
        max_length = np.max(walls[:, 1])
        min_length = np.min(walls[:, 1])    
        
        # Generate uniformly distributed random values for width and length
        # Create particles with the generated width, length, and equal weights
        for i in range(N-len(resampled_particles)):
                width_values = np.random.uniform(min_width, max_width)
                length_values = np.random.uniform(min_length, max_length)
                theta_values = np.random.uniform(-np.pi,np.pi)
                resampled_particles.append(Particle(width_values,length_values,theta_values,1.0/N))
            
        for i in range(N):
            resampled_particles[i].position[0] += np.random.normal(0,sigma_pos)
            resampled_particles[i].position[1] += np.random.normal(0,sigma_pos)
            resampled_particles[i].theta += np.random.normal(0,sigma_angle)
            if(resampled_particles[i].theta > np.pi):
                resampled_particles[i].theta -= 2 * np.pi
            elif (resampled_particles[i].theta < -np.pi):
                resampled_particles[i].theta += 2 * np.pi
            

    return resampled_particles   

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    # Update position
    p.theta += dtheta
    if(p.theta > np.pi):
        p.theta -= 2 * np.pi
    elif (p.theta < -np.pi):
        p.theta += 2 * np.pi
    p.position[0] += traveled_distance * np.cos(p.theta)
    p.position[1] += traveled_distance * np.sin(p.theta)
    
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    particles.sort(key=lambda x:x.weight, reverse=True)
    
    num_particles = len(particles)
    
    if num_particles > 0:
        # Initialize sum of x, y, and theta
        sum_x = 0.0
        sum_y = 0.0
        sum_theta = 0.0
        # Sum up x, y, and theta of all particles
        for i in range(k):
            sum_x += particles[i].position[0]/k
            sum_y += particles[i].position[1]/k
            sum_theta += particles[i].theta/k
            

    final_result.position[0]=sum_x
    final_result.position[1]=sum_y
    final_result.theta=sum_theta
    return final_result