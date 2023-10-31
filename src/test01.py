# tensorboard --logdir "/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/scalars_logs" --port 6006
# tensorboard --logdir "/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/scalars_logs/run_14h_xsinx" --port 6006

# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# writer = SummaryWriter("/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/scalars_logs")
# r = 5
# for i in range(100):
#     writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),'xcosx':i*np.cos(i/r),'tanx': np.tan(i/r)}, i)
# writer.close()

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("/home/he/catkin_nav/src/turtlebot3_ddpg_collision_avoidance/scalars_logs")
x = range(100)
for i in x:
    writer.add_scalar('Main/A', i, i+10)
    # writer.add_scalar('Main/y=x+10',i+10,i)
for i in x:
    # writer.add_scalar('Main/y=2x+1', i, 2*i+1)
    writer.add_scalar('Main/B',2*i+1, i )

writer.close()


