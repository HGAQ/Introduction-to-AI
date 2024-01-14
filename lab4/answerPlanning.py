import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap


### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 1
STEP_DISTANCE_ = 0.2
TARGET_THREHOLD = 0.25
TRY_TIME = 20
### 定义一些你需要的变量和函数 ###


class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置 
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.map = PlanningMap(walls)
        self.walls = walls
        
        # 其他需要的变量
        ### 你的代码 ###      
        self.max_height = np.max(walls[:, 0])
        self.min_height = np.min(walls[:, 0])
        self.max_length = np.max(walls[:, 1])
        self.min_length = np.min(walls[:, 1])  
        
        ### 你的代码 ###
        
        # 如有必要，此行可删除
        self.path = None
        self.next_target = None
        
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
        
        ### 你的代码 ###      
        self.path = self.build_tree(current_position, next_food)
        self.next_target = next_food
        self.visit_time = 0
        ### 你的代码 ###
        # 如有必要，此行可删除
        
        
        
        
        
        
        
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动
        current_position: pacman 当前的仿真位置
        current_velocity: pacman 当前的仿真速度
        一种可能的实现策略是，仅作为参考：
        （1）记录该函数的调用次数
        （2）假设当前 path 中每个节点需要作为目标 n 次
        （3）记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点
        
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target
        
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。你可能需要考虑在什么情况下重新规划 path
        """
        target_position = current_position
        self.visit_time += 1
        
        if (self.visit_time>TRY_TIME) :
            self.find_path(current_position,self.next_target)
        
        ### 你的代码 ###
        if(self.path):
            target_position=self.path[0]
            if self.distance(current_position,target_position) < TARGET_THREHOLD:
                if len(self.path) > 1:
                    self.path.pop(0)
            target_position=self.path[0]
        
        ### 你的代码 ###
        return target_position - current_velocity * 0.0375
        
    ### 以下是RRT中一些可能用到的函数框架，全部可以修改，当然你也可以自己实现 ###
    def build_tree(self, start, goal):
        """
        实现你的快速探索搜索树，输入为当前目标食物的编号，规划从 start 位置食物到 goal 位置的路径
        返回一个包含坐标的列表，为这条路径上的pd targets
        你可以调用find_nearest_point和connect_a_to_b两个函数
        另外self.map的checkoccupy和checkline也可能会需要，可以参考simuScene.py中的PlanningMap类查看它们的用法
        """
        
        
        
        path=[goal]
        graph: List[TreeNode] = []
        graph.append(TreeNode(-1, start[0], start[1]))
        if(self.distance(start, goal) < 0.65):
            path=[goal]
            return path
        ### 你的代码 ###
        foodmap = np.zeros((self.max_height+1,self.max_length+1))
        visit = np.zeros((self.max_height+1,self.max_length+1))
        minlength = np.zeros((self.max_height+1,self.max_length+1))
        fatherx = np.zeros((self.max_height+1,self.max_length+1))
        fathery = np.zeros((self.max_height+1,self.max_length+1))
        
        for i in range(self.max_height+1):
            for j in range(self.max_length+1):
                if(self.map.checkoccupy([i,j])):
                    foodmap[i][j] = -1
                visit[i][j] = -1
                fatherx[i][j] = -1
                fathery[i][j] = -1
                minlength[i][j] = -1
        
        start_=[int(start[0]+0.5),int(start[1]+0.5)]
        goal_=[int(goal[0]),int(goal[1])]
        
        if(start_[0]==goal_[0] and start_[1]==goal_[1]):
            path=[goal]
            return path

        foodmap[goal_[0]][goal_[1]] = 1

        queue=[]
        visit[start_[0]][start_[1]] = 0
        minlength[start_[0]][start_[1]] = 0
        queue.append(start_)
        while(len(queue)):
            curr_pos=queue[0]
            queue.pop(0)
            for i in range(0,self.max_height):
                for j in range(0,self.max_length):
                    next_pos=[i , j]
                    if (self.map.checkoccupy(next_pos)):
                        continue
                    if(next_pos[0]==curr_pos[0] and next_pos[1]==curr_pos[1]):
                        continue
                    if(foodmap[next_pos[0]][next_pos[1]] == -1):
                        continue
                    if(visit[next_pos[0]][next_pos[1]]!=-1 and visit[next_pos[0]][next_pos[1]] < visit[curr_pos[0]][curr_pos[1]]+1):
                        continue
                    if(minlength[next_pos[0]][next_pos[1]]!=-1 and 
                       minlength[next_pos[0]][next_pos[1]] < minlength[curr_pos[0]][curr_pos[1]] + self.distance(curr_pos,next_pos)): 
                        continue
                    is_collapse,coll_pos=self.map.checkline(curr_pos,next_pos)
                    if(is_collapse):
                        continue
                    visit[next_pos[0]][next_pos[1]]=visit[curr_pos[0]][curr_pos[1]]+1
                    minlength[next_pos[0]][next_pos[1]] = minlength[curr_pos[0]][curr_pos[1]] + self.distance(curr_pos,next_pos)
                    fatherx[next_pos[0]][next_pos[1]] = curr_pos[0]
                    fathery[next_pos[0]][next_pos[1]] = curr_pos[1]
                    if next_pos not in queue:
                        queue.append(next_pos)
                        
        curr_goal=[goal_[0], goal_[1]]
        while curr_goal[0] != int(start[0]+0.5) or curr_goal[1] != int(start[1]+0.5):
            path.append([curr_goal[0], curr_goal[1]])
            curr_goal = [int(fatherx[curr_goal[0]][curr_goal[1]]), int(fathery[curr_goal[0]][curr_goal[1]])]
        path.reverse()  
        ### 你的代码 ###
        return path

    @staticmethod
    def find_nearest_point(point, graph):
        """
        找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离、
        输入：
        point：维度为(2,)的np.array, 目标位置
        graph: List[TreeNode]节点列表
        输出：
        nearest_idx, nearest_distance 离目标位置距离最近节点的编号和距离
        """
        nearest_idx = -1
        nearest_distance = 10000000.
        ### 你的代码 ###
        for i in range(len(graph)):
            curr_point=graph[i]
            distance=np.sqrt((point[0]-curr_point.pos[0])**2+(point[1]-curr_point.pos[1])**2)
            if nearest_distance>distance:
                nearest_distance=distance
                nearest_idx=i
        ### 你的代码 ###
        return nearest_idx, nearest_distance

    def distance(self, point_a, point_b):
        distance=np.sqrt((point_a[0]-point_b[0])**2+(point_a[1]-point_b[1])**2)
        return distance

    def connect_a_to_b(self, point_a, point_b):
        """
        以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，并用self.map.checkline函数检查这段路程是否可以通过
        输入：
        point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进
        输出：
        is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物
        newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，如果is_empty为真，之后的代码需要把这个新位置添加到图中
        """
        
        
        is_empty = False
        newpoint = np.zeros(2)
        ### 你的代码 ###
        deltax=point_b[0]-point_a[0]
        deltay=point_b[1]-point_a[1]
        deltadistance=np.sqrt(deltax**2+deltay**2)
        
        
        step=STEP_DISTANCE
        
        dx = step*deltax/deltadistance
        dy = step*deltay/deltadistance
        
        newpoint=[point_a[0]+dx,point_a[1]+dy]
        
        if (self.map.checkline(np.ndarray.tolist(point_a),newpoint)):
            is_empty=1
        else:
            is_empty=0
        
        
        ### 你的代码 ###
        return is_empty, newpoint



                
'''
        time=0
        while True:
            width_values = np.random.uniform(self.min_height, self.max_height)
            length_values = np.random.uniform(self.min_length, self.max_length)
            new_pos = np.array([width_values, length_values])
            if not self.map.checkoccupy(np.ndarray.tolist(new_pos)):
                time+=1
                if(time>3000):
                    nearest_idx, nearest_distance=self.find_nearest_point(goal,graph)
                    print(nearest_distance)
                    curr_node = graph[nearest_idx]
                    curr_idx = nearest_idx
                    while True:
                        path.append(curr_node.pos)
                        if(graph[curr_idx].parent_idx==-1):
                            break
                        curr_node=graph[curr_idx]
                        curr_idx=curr_node.parent_idx
                    path.reverse()
                    return path
                nearest_idx, nearest_distance=self.find_nearest_point(new_pos,graph)
                
                close_pos = graph[nearest_idx].pos
                
                is_empty, newpoint= self.connect_a_to_b(close_pos, new_pos)
                if(is_empty):
                    new_node=TreeNode(nearest_idx, newpoint[0], newpoint[1])
                    graph.append(new_node)
                    distance=np.sqrt((goal[0]-newpoint[0])**2+(goal[1]-newpoint[1])**2)
                    if(distance<TARGET_THREHOLD):
                        curr_node=new_node
                        curr_idx=nearest_idx
                        while True:
                            path.append(curr_node.pos)
                            if(graph[curr_idx].parent_idx==-1):
                                break
                            curr_node=graph[curr_idx]
                            curr_idx=curr_node.parent_idx
                        path.reverse()
                        return path
            else:
                continue
        '''