import mini_behavior.envs
import gym

env = gym.make('MiniGrid-TwoRoomNavigation-8x8-N2-v0')

obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)

    if done:
        obs = env.reset()
    env.render()

env.close()

# ####################################################################################################

# class base:
#     def __init__(self):
#         super(child).__init__()
#         print("I'm SUPER")
#
#
# class child(base):
#     def __init__(self):
#         super(child, self).__init__()
#         print("Child")
#
#
# class GChild(child):
#     def __init__(self):
#         super(GChild, self).__init__()
#         print("Grand Child")
#
#
# Base = base()
# print("== " * 10)
# Child = child()
# print("== " * 10)
# g = GChild()
