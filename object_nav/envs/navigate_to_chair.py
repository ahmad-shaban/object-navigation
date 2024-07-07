from mini_behavior.roomgrid import *
from mini_behavior.register import register
from mini_behavior.objects import *

import os, json
from object_nav.rewarder.rewarder import distance_rw, composite_rw, steps_rw
import copy
import random

# todo:what we want to achieve
GOAL_OBJECTS_orig = ['bed', 'chair', 'table', 'printer', 'electric_refrigerator', 'shower']
# temp list to test the environment
GOAL_OBJECTS = ['chair']

# write mapping from goal object to a one-hot econding {bed: [1, 0, 0, 0, 0, 0], chair: [0, 1, 0, 0, 0, 0], ...}
GOAL_OBJECTS_ONEHOT = {obj: [int(obj == goal_obj) for goal_obj in GOAL_OBJECTS] for obj in GOAL_OBJECTS}


class NavigateToObjEnv(RoomGrid):
    """
    Environment simulate an agent navigate to a specific obj
    """

    # override the Actions enum
    class Actions(IntEnum):
        left = 0
        right = 1
        forward = 2
        open = 3

    def __init__(self,
                 mode='primitive',
                 initial_dict=None,
                 dense_reward=True,
                 room_size=16,
                 max_steps=1e5):
        # todo: make the mission random from this list
        self.goal_obj = random.sample(GOAL_OBJECTS, k=1)[0]   # random choose from Goal objects
        self.goal_hot = GOAL_OBJECTS_ONEHOT[self.goal_obj]

        self.mission = initial_dict["Grid"]["mission"]

        # configurations for agent generation
        self.initial_dict = initial_dict
        self.state_dict = copy.deepcopy(initial_dict)

        # for auto nums
        self.auto_room_split_dirs = self.initial_dict["Grid"]["auto"]["room_split_dirs"]
        self.auto_min_room_dim = self.initial_dict["Grid"]["auto"]["min_room_dim"]
        self.auto_max_num_room = self.initial_dict["Grid"]["auto"]["max_num_room"]
        # agents
        self.agents = []

        # if no goal object is available, reset the environment
        self.max_reset_steps = 10

        self.rewarder = composite_rw([distance_rw(), steps_rw()])
        # generate floorplans(i.e rooms), objects, agents
        super().__init__(mode=mode,
                         init_dict=initial_dict,
                         max_steps=max_steps,
                         dense_reward=dense_reward
                         )
        self.actions = NavigateToObjEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))

    def overlap_disable(self):
        for obj in self.obj_instances.values():
            obj.can_overlap = False

    def _gen_env_objs(self):
        """
        in case of no init_dict is given,
        it will be called by RoomGrid._gen_objs() ...
        to create env objs.
        :return: None
        """

        # todo: adjust the implementation to be suitable to our needs
        electric_refrigerator = self.objs['electric_refrigerator'][0]
        lettuce = self.objs['lettuce']
        countertop = self.objs['countertop'][0]
        apple = self.objs['apple']
        tomato = self.objs['tomato']
        lights = self.objs['light']
        carving_knife = self.objs['carving_knife'][0]
        plate = self.objs['plate']
        cabinet = self.objs['cabinet'][0]
        sink = self.objs['sink'][0]
        beds = self.objs['bed']
        chairs = self.objs['chair']

        self.objs['electric_refrigerator'][0].width = 1
        self.objs['electric_refrigerator'][0].height = 2

        self.objs['countertop'][0].width = 2
        self.objs['countertop'][0].height = 3

        self.objs['sink'][0].width = 2
        self.objs['sink'][0].height = 2

        self.objs['cabinet'][0].width = 2
        self.objs['cabinet'][0].height = 2

        for bed in self.objs['bed']:
            bed.width = 2
            bed.height = 3

        for chair in self.objs['chair']:
            chair.width = 1
            chair.height = 1

        room_tops = [(1, 1), (9, 1), (1, 23)]
        room_sizes = [(4, 10), (5, 10), (8, 6)]
        kitchen_top = (9, 12)
        kitchen_size = (5, 7)
        bathroom_top = (10, 23)
        bathroom_size = (4, 6)

        self.place_obj(countertop, kitchen_top, kitchen_size)
        self.place_obj(electric_refrigerator, kitchen_top, kitchen_size)
        self.place_obj(cabinet, kitchen_top, kitchen_size)
        self.place_obj(sink, bathroom_top, bathroom_size)
        for i, bed in enumerate(beds):
            self.place_obj(bed, room_tops[i], room_sizes[i])
            self.place_obj(chairs[i], room_tops[i], room_sizes[i])
            self.place_obj(lights[i], room_tops[i], room_sizes[i])

        self.place_obj(lights[-1], kitchen_top, kitchen_size)

        countertop_pos = random.sample(countertop.all_pos, 6)
        self.put_obj(lettuce[0], *countertop_pos[0])
        self.put_obj(lettuce[1], *countertop_pos[1])
        self.put_obj(apple[0], *countertop_pos[2])
        self.put_obj(apple[1], *countertop_pos[3])

        fridge_pos = random.sample(electric_refrigerator.all_pos, 2)
        self.put_obj(tomato[0], *fridge_pos[0])
        self.put_obj(tomato[1], *fridge_pos[1])

        cabinet_pos = random.sample(cabinet.all_pos, 3)
        self.put_obj(plate[0], *cabinet_pos[0])
        plate[0].states['dustyable'].set_value(False)
        self.put_obj(plate[1], *cabinet_pos[1])
        plate[1].states['dustyable'].set_value(False)
        self.put_obj(carving_knife, *cabinet_pos[2])

    def _gen_random_floorplan(self, room_num):
        x_min, y_min, x_max, y_max = 1, 1, self.width-2, self.height-2
        tops = []
        sizes = []
        for room_id in range(room_num-1):
            cur_dir = self._rand_subset(self.auto_room_split_dirs, 1)[0]
            if cur_dir == 'vert':
                # Create a vertical splitting wall
                splitIdx = self._rand_int(
                    x_min + self.auto_min_room_dim, max(x_min + self.auto_min_room_dim + 1, min(3*(x_min + x_max)/2, x_max - (room_num - room_id - 1) * self.auto_min_room_dim)))
                self.floor_plan_walls.append(('vert', (splitIdx, y_min), y_max - y_min + 1))
                tops.append((x_min, y_min))
                sizes.append((splitIdx - x_min, y_max - y_min + 1))
                x_min = splitIdx + 1
            else:
                # Create a horizontal splitting wall
                splitIdx = self._rand_int(
                    y_min + self.auto_min_room_dim, max(y_min + self.auto_min_room_dim + 1, min(3*(y_min + y_max)/2, y_max - (room_num - room_id - 1) * self.auto_min_room_dim)))
                self.floor_plan_walls.append(('horz', (x_min, splitIdx), x_max - x_min + 1))
                tops.append((x_min, y_min))
                sizes.append((x_max - x_min + 1, splitIdx - y_min))
                # horiz generate room with top and size", splitIdx, top, size)
                y_min = splitIdx + 1
        tops.append((x_min, y_min))
        sizes.append((x_max - x_min + 1, y_max - y_min + 1))

        return tops, sizes

    def _reward(self):
        # print(self.gen_obs())
        if self._end_conditions():
            return 1000
        else:
            if self.dense_reward:
                reward = self.rewarder.get_reward(self)
                self.previous_progress = reward
                return reward
            else:
                return 0

    def get_progress(self) -> float:
        pass

    def reset(self):
        super().reset()
        self.overlap_disable()

        if self._init_conditions():
            self.max_reset_steps = 10
            self.goal_obj = random.sample(self.available_goals, k=1)[0]

            # check no goal is in reach of robot at beginning
            for goal in self.objs[self.goal_obj]:
                if goal.check_abs_state(self, 'inreachofrobot'):
                    return self.reset()

            self.goal_hot = GOAL_OBJECTS_ONEHOT[self.goal_obj]
            self.rewarder.reset(self)
            return self.gen_obs()
        else:
            self.max_reset_steps -= 1
            if self.max_reset_steps > 0:
                return self.reset()
            else:
                raise Exception("No goal object is available in the environment")

    def _init_conditions(self):
        """
        check if a goal object is available in the environment
        :return: true if goal object is available, false otherwise
        """
        self.available_goals = [obj for obj in self.objs.keys() if obj in GOAL_OBJECTS]
        if self.available_goals:
            return True
        return False

    def _end_conditions(self):
        goals: list[WorldObj] = self.objs[self.goal_obj]
        for goal in goals:
            if goal.check_abs_state(self, 'inreachofrobot'):
                return True
        return False


script_dir = os.path.dirname(__file__)
env_name = "MiniGrid-NavigateToObj-16x16-N2-v0"
kwargs = {"max_steps": 1000}
abs_file_path = os.path.join(script_dir, './floorplans/one_room.json')
with open(abs_file_path, 'r') as f:
    initial_dict = json.load(f)
    kwargs["initial_dict"] = initial_dict
register(
    id=env_name,
    entry_point=f'object_nav.envs:NavigateToObjEnv',
    kwargs=kwargs
)
