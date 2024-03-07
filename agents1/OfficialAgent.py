import sys, random, enum, ast, time, csv
import numpy as np
from matrx import grid_world
from brains1.ArtificialBrain import ArtificialBrain
from actions1.CustomActions import *
from matrx import utils
from matrx.grid_world import GridWorld
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject, RemoveObject
from matrx.actions.move_actions import MoveNorth
from matrx.messages.message import Message
from matrx.messages.message_manager import MessageManager
from actions1.CustomActions import RemoveObjectTogether, CarryObjectTogether, DropObjectTogether, CarryObject, Drop

# TODO:
"""     韩少确定的新逻辑是这样的：每次一遍遍历完所有房间，更新一次trust，我们需要一个数据结构来存储在
        这个过程中发生的人所有的对话，例如“我去三号房救人了”，在一次遍历后如果人没救齐那就是人说谎，这
        个时候才发生减分的情况。同理如果齐了就加分。当然这个里面还有其他的情况也要考虑，例如人说了别的
        真话。"""



class Phase(enum.Enum):
    INTRO = 1,
    FIND_NEXT_GOAL = 2,
    PICK_UNSEARCHED_ROOM = 3,
    PLAN_PATH_TO_ROOM = 4,
    FOLLOW_PATH_TO_ROOM = 5,
    PLAN_ROOM_SEARCH_PATH = 6,
    FOLLOW_ROOM_SEARCH_PATH = 7,
    PLAN_PATH_TO_VICTIM = 8,
    FOLLOW_PATH_TO_VICTIM = 9,
    TAKE_VICTIM = 10,
    PLAN_PATH_TO_DROPPOINT = 11,
    FOLLOW_PATH_TO_DROPPOINT = 12,
    DROP_VICTIM = 13,
    WAIT_FOR_HUMAN = 14,
    WAIT_AT_ZONE = 15,
    FIX_ORDER_GRAB = 16,
    FIX_ORDER_DROP = 17,
    REMOVE_OBSTACLE_IF_NEEDED = 18,
    ENTER_ROOM = 19


class BaselineAgent(ArtificialBrain):
    def __init__(self, slowdown, condition, name, folder):
        super().__init__(slowdown, condition, name, folder)
        # Initialization of some relevant variables
        self._tick = None
        self._slowdown = slowdown
        self._condition = condition
        self._human_name = name
        self._folder = folder
        self._phase = Phase.INTRO
        self._room_vics = []
        self._searched_rooms = []
        self._found_victims = []
        self._collected_victims = []
        self._found_victim_logs = {}
        self._send_messages = []
        self._current_door = None
        self._team_members = []
        self._carrying_together = False
        self._remove = False
        self._goal_vic = None
        self._goal_loc = None
        self._human_loc = None
        self._distance_human = None
        self._distance_drop = None
        self._agent_loc = None
        self._todo = []
        self._answered = False
        self._to_search = []
        self._carrying = False
        self._waiting = False
        self._rescue = None
        self._recent_vic = None
        self._received_messages = []
        self._moving = False


        # TODO:
        """
            这里写的东西是两个个list，一个用来存人说过的话（string类型），如果在遍历结束过后人没有救齐，
            那么就需要去检查第一个list里人说的话是否实现了，结果会存在第二个list里（这个list就是bool类型了）
            根据第二个结果来扣分/加分
        """

        self.declared_actions = []
        self.actual_actions = []

        self.human_found_victims = []
        self.human_searched_rooms = []
        self.human_found_victim_logs = {}

        self.human_collected_victims = []
        self.human_collected_victim_logs = {}
        self.test_human = False

        self.initial_competence = 0
        self.initial_willingness = 0


    def initialize(self):
        # Initialization of the state tracker and navigation algorithm
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set,
                                    algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_observations(self, state):
        # Filtering of the world state before deciding on an action
        return state

    def decide_on_actions(self, state):
        # Identify team members
        agent_name = state[self.agent_id]['obj_id']
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._team_members:
                self._team_members.append(member)
        # Create a list of received messages from the human team member
        for mssg in self.received_messages:
            for member in self._team_members:
                if mssg.from_id == member and mssg.content not in self._received_messages:
                    self._received_messages.append(mssg.content)
                    self.declared_actions.append(mssg.content)
        # Process messages from team members
        # self._process_messages(state, self._team_members, self._condition)
        # Initialize and update trust beliefs for team members
        trustBeliefs = self._loadBelief(self._team_members, self._folder)
        # print(trustBeliefs[self._human_name]['competence'], trustBeliefs[self._human_name]['willingness'])
        self._trustBelief(self._team_members, trustBeliefs, self._folder, 0)
        # Process messages from team members
        self._process_messages(state, self._team_members, self._condition, trustBeliefs, self._human_name)

        # Check whether human is close in distance
        if state[{'is_human_agent': True}]:
            self._distance_human = 'close'
        if not state[{'is_human_agent': True}]:
            # Define distance between human and agent based on last known area locations
            if self._agent_loc in [1, 2, 3, 4, 5, 6, 7] and self._human_loc in [8, 9, 10, 11, 12, 13, 14]:
                self._distance_human = 'far'
            if self._agent_loc in [1, 2, 3, 4, 5, 6, 7] and self._human_loc in [1, 2, 3, 4, 5, 6, 7]:
                self._distance_human = 'close'
            if self._agent_loc in [8, 9, 10, 11, 12, 13, 14] and self._human_loc in [1, 2, 3, 4, 5, 6, 7]:
                self._distance_human = 'far'
            if self._agent_loc in [8, 9, 10, 11, 12, 13, 14] and self._human_loc in [8, 9, 10, 11, 12, 13, 14]:
                self._distance_human = 'close'

        # Define distance to drop zone based on last known area location
        if self._agent_loc in [1, 2, 5, 6, 8, 9, 11, 12]:
            self._distance_drop = 'far'
        if self._agent_loc in [3, 4, 7, 10, 13, 14]:
            self._distance_drop = 'close'

        # Check whether victims are currently being carried together by human and agent
        for info in state.values():
            # if 'is_goal_block' in info and :

            # if 'img_name' in info and 'injured' in info['img_name']:
            #     print(info)
            # 如果机器人看到人carry一个人 那么这个行为就是正向的
            # 给人加一次分
            # 给人加一次分
            # if 'carried_by' in info and len(info['carried_by']) > 0:
            #     print(info)
            # if ('is_carrying' in info and info['name'] == self._human_name
            #         and len(info['is_carrying']) > 0):
            #     print(info)
            #     self.trust_mechanism_draft('+')
            if 'is_human_agent' in info and self._human_name in info['name'] and len(
                    info['is_carrying']) > 0 and 'critical' in info['is_carrying'][0]['obj_id'] or \
                    'is_human_agent' in info and self._human_name in info['name'] and len(
                info['is_carrying']) > 0 and 'mild' in info['is_carrying'][0][
                'obj_id'] and self._rescue == 'together' and not self._moving:
                # If victim is being carried, add to collected victims memory
                if info['is_carrying'][0]['img_name'][8:-4] not in self._collected_victims:
                    self._collected_victims.append(info['is_carrying'][0]['img_name'][8:-4])

                    #Diff: 一起搬人说明人没有撒谎，积极行为，将这个victim从human_found_victim种取消，trust也应该增加
                    if info['is_carrying'][0]['img_name'][8:-4] in self.human_collected_victims:
                        self.human_collected_victim_logs.pop(info['is_carrying'][0]['img_name'][8:-4], None)
                        # self.human_found_victims.remove(info['is_carrying'][0]['img_name'][8:-4])
                        self.human_collected_victims.remove(info['is_carrying'][0]['img_name'][8:-4])

                        self.human_found_victim_logs.pop(info['is_carrying'][0]['img_name'][8:-4], None)
                        # self.human_found_victims.remove(info['is_carrying'][0]['img_name'][8:-4])
                        self.human_found_victims.remove(info['is_carrying'][0]['img_name'][8:-4])
                        #DiffToDo:
                        #increase trust here
                        self._trustBelief(self._team_members, trustBeliefs, self._folder, 1)
                        print("走到了 --------------  1")
                        print("一起搬人说明人没有撒谎，积极行为，将这个victim从human_found_victim种取消，trust也应该增加")

                self._carrying_together = True
            if 'is_human_agent' in info and self._human_name in info['name'] and len(info['is_carrying']) == 0:
                self._carrying_together = False
        # If carrying a victim together, let agent be idle (because joint actions are essentially carried out by the human)
        if self._carrying_together == True:
            return None, {}

        # Send the hidden score message for displaying and logging the score during the task, DO NOT REMOVE THIS
        self._send_message('Our score is ' + str(state['rescuebot']['score']) + '.', 'RescueBot')

        # Ongoing loop until the task is terminated, using different phases for defining the agent's behavior
        while True:
            if Phase.INTRO == self._phase:
                # Send introduction message
                self._send_message('Hello! My name is RescueBot. Together we will collaborate and try to search and rescue the 8 victims on our right as quickly as possible. \
                Each critical victim (critically injured girl/critically injured elderly woman/critically injured man/critically injured dog) adds 6 points to our score, \
                each mild victim (mildly injured boy/mildly injured elderly man/mildly injured woman/mildly injured cat) 3 points. \
                If you are ready to begin our mission, you can simply start moving.', 'RescueBot')
                # Wait untill the human starts moving before going to the next phase, otherwise remain idle
                if not state[{'is_human_agent': True}]:
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    return None, {}

            if Phase.FIND_NEXT_GOAL == self._phase:
                # Definition of some relevant variables
                self._answered = False
                self._goal_vic = None
                self._goal_loc = None
                self._rescue = None
                self._moving = True
                remaining_zones = []
                remaining_vics = []
                remaining = {}
                # Identification of the location of the drop zones
                zones = self._get_drop_zones(state)
                print('agent_collected:'+str(self._collected_victims))
                # Identification of which victims still need to be rescued and on which location they should be dropped
                for info in zones:
                    #print(info)
                    if str(info['img_name'])[8:-4] not in self._collected_victims and len(self._searched_rooms) != 14:
                        remaining_zones.append(info)
                        remaining_vics.append(str(info['img_name'])[8:-4])
                        remaining[str(info['img_name'])[8:-4]] = info['location']
                if remaining_zones:
                    self._remainingZones = remaining_zones
                    self._remaining = remaining
                # Remain idle if there are no victims left to rescue
                #Diff: 如果房间搜完但分数不够，则查询human found的vicitms
                elif not remaining_zones and state['rescuebot']['score'] == 36:
                    return None, {}
                else:
                    for info in zones:
                        if str(info['img_name'])[8:-4] in self.human_found_victims or str(info['img_name'])[8:-4] in self.human_collected_victims:
                            remaining_zones.append(info)
                            remaining_vics.append(str(info['img_name'])[8:-4])
                            remaining[str(info['img_name'])[8:-4]] = info['location']
                    if remaining_zones:
                        self._remainingZones = remaining_zones
                        self._remaining = remaining
                        # DiffToDo: decrease trust here
                        if not self.test_human:
                            self._trustBelief(self._team_members, trustBeliefs, self._folder, 2)
                            print("走到了 --------------  2")
                            print("如果房间搜完但分数不够，则查询human found的vicitms")
                        self.test_human = True


                # Check which victims can be rescued next because human or agent already found them
                for vic in remaining_vics:
                    # Define a previously found victim as target victim because all areas have been searched
                    if vic in self._found_victims and vic in self._todo and len(self._searched_rooms) == 0:
                        self._goal_vic = vic
                        self._goal_loc = remaining[vic]
                        # Move to target victim
                        self._rescue = 'together'
                        self._send_message('Moving to ' + self._found_victim_logs[vic][
                            'room'] + ' to pick up ' + self._goal_vic + '. Please come there as well to help me carry ' + self._goal_vic + ' to the drop zone.',
                                          'RescueBot')
                        # Plan path to victim because the exact location is known (i.e., the agent found this victim)
                        if 'location' in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        # Plan path to area because the exact victim location is not known, only the area (i.e., human found this  victim)
                        if 'location' not in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}
                    # Define a previously found victim as target victim
                    if vic in self._found_victims and vic not in self._todo:
                        self._goal_vic = vic
                        self._goal_loc = remaining[vic]
                        # Rescue together when victim is critical or when the human is weak and the victim is mildly injured
                        if 'critical' in vic or 'mild' in vic and self._condition == 'weak':
                            self._rescue = 'together'
                        # Rescue alone if the victim is mildly injured and the human not weak
                        if 'mild' in vic and self._condition != 'weak':
                            self._rescue = 'alone'
                        # Plan path to victim because the exact location is known (i.e., the agent found this victim)
                        if 'location' in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        # Plan path to area because the exact victim location is not known, only the area (i.e., human found this  victim)
                        if 'location' not in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}
                    # If there are no target victims found, visit an unsearched area to search for victims
                    if vic not in self._found_victims or vic in self._found_victims and vic in self._todo and len(
                            self._searched_rooms) > 0:
                        self._phase = Phase.PICK_UNSEARCHED_ROOM

            if Phase.PICK_UNSEARCHED_ROOM == self._phase:
                agent_location = state[self.agent_id]['location']
                # Identify which areas are not explored yet
                unsearched_rooms = [room['room_name'] for room in state.values()
                                   if 'class_inheritance' in room
                                   and 'Door' in room['class_inheritance']
                                   and room['room_name'] not in self._searched_rooms
                                   and room['room_name'] not in self._to_search]
                # If all areas have been searched but the task is not finished, start searching areas again
                #diff: 优先搜索human searched rooms
                # print(self.test_human)
                # print(len(unsearched_rooms))
                if (self.test_human and len(unsearched_rooms) == 0) and len(self.human_searched_rooms) != 0:

                    for room in self.human_searched_rooms:
                        unsearched_rooms.append(room)
                    #self.human_searched_rooms = []
                    print('human_searched_rooms:' + str(unsearched_rooms))
                    print(self._remainingZones)

                if self._remainingZones and len(unsearched_rooms) == 0:
                    print("走了if")
                    self._to_search = []
                    self._searched_rooms = []
                    self._send_messages = []
                    self.received_messages = []
                    self.received_messages_content = []
                    self._send_message('Going to re-search all areas.', 'RescueBot')
                    self._phase = Phase.FIND_NEXT_GOAL
                # If there are still areas to search, define which one to search next
                else:
                    print("走了else")
                    self.received_messages = []
                    self.received_messages_content = []
                    # Identify the closest door when the agent did not search any areas yet
                    if self._current_door == None:
                        print("走了第一个if")
                        # Find all area entrance locations
                        self._door = state.get_room_doors(self._getClosestRoom(state, unsearched_rooms, agent_location))[
                            0]
                        self._doormat = \
                            state.get_room(self._getClosestRoom(state, unsearched_rooms, agent_location))[-1]['doormat']
                        # Workaround for one area because of some bug
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        # Plan path to area
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    # Identify the closest door when the agent just searched another area
                    if self._current_door != None:
                        print("走了第二个if")
                        self._door = \
                            state.get_room_doors(self._getClosestRoom(state, unsearched_rooms, self._current_door))[0]
                        self._doormat = \
                            state.get_room(self._getClosestRoom(state, unsearched_rooms, self._current_door))[-1][
                                'doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                # Reset the navigator for a new path planning
                self._navigator.reset_full()

                # Check if there is a goal victim, and it has been found, but its location is not known
                if self._goal_vic \
                        and self._goal_vic in self._found_victims \
                        and 'location' not in self._found_victim_logs[self._goal_vic].keys():
                    # Retrieve the victim's room location and related information
                    victim_location = self._found_victim_logs[self._goal_vic]['room']
                    self._door = state.get_room_doors(victim_location)[0]
                    self._doormat = state.get_room(victim_location)[-1]['doormat']

                    # Handle special case for 'area 1'
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)

                    # Set the door location based on the doormat
                    doorLoc = self._doormat

                # If the goal victim's location is known, plan the route to the identified area
                else:
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)
                    doorLoc = self._doormat

                # Add the door location as a waypoint for navigation
                self._navigator.add_waypoints([doorLoc])
                # Follow the route to the next area to search
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                # Check if the previously identified target victim was rescued by the human
                if not self.test_human and self._goal_vic and self._goal_vic in self._collected_victims:
                    print("进了这个if")
                    # Reset current door and switch to finding the next goal
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Check if the human found the previously identified target victim in a different room
                if not self.test_human and self._goal_vic \
                        and self._goal_vic in self._found_victims \
                        and self._door['room_name'] != self._found_victim_logs[self._goal_vic]['room']:
                    print("进了第二个if")
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Check if the human already searched the previously identified area without finding the target victim
                if not self.test_human and self._door['room_name'] in self._searched_rooms and self._goal_vic not in self._found_victims:
                    print("进了第三个if")
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Move to the next area to search
                else:
                    # Update the state tracker with the current state
                    self._state_tracker.update(state)

                    # Explain why the agent is moving to the specific area, either:
                    # [-] it contains the current target victim
                    # [-] it is the closest un-searched area
                    if self._goal_vic in self._found_victims \
                            and str(self._door['room_name']) == self._found_victim_logs[self._goal_vic]['room'] \
                            and not self._remove:
                        if self._condition == 'weak':
                            self._send_message('Moving to ' + str(
                                self._door['room_name']) + ' to pick up ' + self._goal_vic + ' together with you.',
                                              'RescueBot')
                        else:
                            self._send_message(
                                'Moving to ' + str(self._door['room_name']) + ' to pick up ' + self._goal_vic + '.',
                                'RescueBot')

                    if self._goal_vic not in self._found_victims and not self._remove or not self._goal_vic and not self._remove:
                        self._send_message(
                            'Moving to ' + str(self._door['room_name']) + ' because it is the closest unsearched area.',
                            'RescueBot')

                    # Set the current door based on the current location
                    self._current_door = self._door['location']

                    # Retrieve move actions to execute
                    action = self._navigator.get_move_action(self._state_tracker)
                    # Check for obstacles blocking the path to the area and handle them if needed
                    if action is not None:
                        # Remove obstacles blocking the path to the area
                        for info in state.values():
                            if 'class_inheritance' in info and 'ObstacleObject' in info[
                                'class_inheritance'] and 'stone' in info['obj_id'] and info['location'] not in [(9, 4),
                                                                                                                (9, 7),
                                                                                                                (9, 19),
                                                                                                                (21,
                                                                                                                 19)]:
                                self._send_message('Reaching ' + str(self._door['room_name'])
                                                   + ' will take a bit longer because I found stones blocking my path.',
                                                   'RescueBot')
                                return RemoveObject.__name__, {'object_id': info['obj_id']}
                        return action, {}
                    print("进了最后一个else")
                    # Identify and remove obstacles if they are blocking the entrance of the area
                    self._phase = Phase.REMOVE_OBSTACLE_IF_NEEDED

            if Phase.REMOVE_OBSTACLE_IF_NEEDED == self._phase:
                objects = []
                agent_location = state[self.agent_id]['location']
                # Identify which obstacle is blocking the entrance
                for info in state.values():
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'rock' in info[
                        'obj_id']:
                        objects.append(info)
                        # diff decrease trust
                        if (self.test_human and not self._waiting and not self._answered and self._door['room_name'] in self.human_searched_rooms):
                            self._trustBelief(self._team_members, trustBeliefs, self._folder, 2)
                            # self.human_searched_rooms.remove(self._door['room_name'])
                            print("如果有障碍物但人说搜过了")
                            print(self.human_searched_rooms)

                        # Communicate which obstacle is blocking the entrance
                        if self._answered == False and not self._remove and not self._waiting:


                            self._send_message('Found rock blocking ' + str(self._door['room_name']) + '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(
                                self._collected_victims) + ' \n explore - areas searched: area ' + str(
                                self._searched_rooms).replace('area ', '') + ' \
                                \n clock - removal time: 5 seconds \n afstand - distance between us: ' + self._distance_human,
                                              'RescueBot')
                            self._waiting = True
                            # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            #有continue就是懒逼
                            self._trustBelief(self._team_members, trustBeliefs, self._folder, 3)
                            # Add area to the to do list
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Wait for the human to help removing the obstacle and remove the obstacle together
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove' or self._remove:
                            if not self._remove:
                                self._answered = True
                            # Tell the human to come over and be idle untill human arrives
                            if not state[{'is_human_agent': True}]:
                                self._send_message('Please come to ' + str(self._door['room_name']) + ' to remove rock.',
                                                  'RescueBot')
                                return None, {}
                            # Tell the human to remove the obstacle when he/she arrives
                            if state[{'is_human_agent': True}]:
                                self._send_message('Lets remove rock blocking ' + str(self._door['room_name']) + '!',
                                                  'RescueBot')
                                return None, {}
                        # Remain idle untill the human communicates what to do with the identified obstacle
                        else:
                            return None, {}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'tree' in info[
                        'obj_id']:
                        objects.append(info)

                        # diff decrease trust
                        if (self.test_human and not self._waiting and not self._answered and self._door['room_name'] in self.human_searched_rooms):
                            self._trustBelief(self._team_members, trustBeliefs, self._folder, 2)
                            # self.human_searched_rooms.remove(self._door['room_name'])
                            print("如果有障碍物但人说搜过了")
                            print(self.human_searched_rooms)

                        #Diff:信任的话
                        if (trustBeliefs[self._human_name]['competence'] > 0.0 and trustBeliefs[self._human_name][
                            'willingness'] > 0.0):
                        # Communicate which obstacle is blocking the entrance
                            if self._answered == False and not self._remove and not self._waiting:
                                self._send_message('Found tree blocking  ' + str(self._door['room_name']) + '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                    Important features to consider are: \n safe - victims rescued: ' + str(
                                    self._collected_victims) + '\n explore - areas searched: area ' + str(
                                    self._searched_rooms).replace('area ', '') + ' \
                                    \n clock - removal time: 10 seconds', 'RescueBot')
                                self._waiting = True
                            # Determine the next area to explore if the human tells the agent not to remove the obstacle
                            if self.received_messages_content and self.received_messages_content[
                                -1] == 'Continue' and not self._remove:
                                self._answered = True
                                self._waiting = False
                                self._trustBelief(self._team_members, trustBeliefs, self._folder, 3)
                                # Add area to the to do list
                                self._to_search.append(self._door['room_name'])
                                self._phase = Phase.FIND_NEXT_GOAL
                            # Remove the obstacle if the human tells the agent to do so
                            if self.received_messages_content and self.received_messages_content[
                                -1] == 'Remove' or self._remove:
                                if not self._remove:
                                    self._answered = True
                                    self._waiting = False
                                    self._send_message('Removing tree blocking ' + str(self._door['room_name']) + '.',
                                                      'RescueBot')
                                if self._remove:
                                    self._send_message('Removing tree blocking ' + str(
                                        self._door['room_name']) + ' because you asked me to.', 'RescueBot')
                                self._phase = Phase.ENTER_ROOM
                                self._remove = False
                                return RemoveObject.__name__, {'object_id': info['obj_id']}
                            # Remain idle untill the human communicates what to do with the identified obstacle
                            else:
                                return None, {}
                        else:
                            #Diff: move alone
                            self._send_message('Removing tree blocking ' + str(
                                self._door['room_name']) + ' because you are laying.', 'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in \
                            info['obj_id']:
                        objects.append(info)

                        # diff decrease trust
                        if (self.test_human and not self._waiting and not self._answered and self._door['room_name'] in self.human_searched_rooms):
                            self._trustBelief(self._team_members, trustBeliefs, self._folder, 2)
                            #self.human_searched_rooms.remove(self._door['room_name'])
                            print("如果有障碍物但人说搜过了")
                            print(self.human_searched_rooms)

                        # Diff:信任的话
                        #这里是小石头的，小石头就是比较慢所以机器人更倾向于俩人一起，能力大于0.0就行
                        #然后在是不是懒逼方面，如果人近的话大于0.0，如果人远的话大于0.5
                        if (trustBeliefs[self._human_name]['competence'] > 0.0 and trustBeliefs[self._human_name][
                                'willingness'] > 0.5 and self._distance_human == 'far') or (trustBeliefs[self._human_name]['competence'] > 0.0 and trustBeliefs[self._human_name][
                                'willingness'] > 0.0 and self._distance_human == 'close'):
                            # Communicate which obstacle is blocking the entrance
                            if self._answered == False and not self._remove and not self._waiting:
                                #if willingness>0.5 else do it alone
                                self._send_message('Found stones blocking  ' + str(self._door['room_name']) + '. Please decide whether to "Remove together", "Remove alone", or "Continue" searching. \n \n \
                                    Important features to consider are: \n safe - victims rescued: ' + str(
                                    self._collected_victims) + ' \n explore - areas searched: area ' + str(
                                    self._searched_rooms).replace('area', '') + ' \
                                    \n clock - removal time together: 3 seconds \n afstand - distance between us: ' + self._distance_human + '\n clock - removal time alone: 20 seconds',
                                                  'RescueBot')
                                self._waiting = True
                            # Determine the next area to explore if the human tells the agent not to remove the obstacle
                            if self.received_messages_content and self.received_messages_content[
                                -1] == 'Continue' and not self._remove:

                                self._answered = True
                                self._waiting = False
                                # Add area to the to do list
                                self._trustBelief(self._team_members, trustBeliefs, self._folder, 3)
                                self._to_search.append(self._door['room_name'])
                                self._phase = Phase.FIND_NEXT_GOAL
                            # Remove the obstacle alone if the human decides so
                            if (trustBeliefs[self._human_name]['willingness'] < 0.2) or (self.received_messages_content and self.received_messages_content[
                                -1] == 'Remove alone' and not self._remove):
                                self._answered = True
                                self._waiting = False
                                self._send_message('Removing stones blocking ' + str(self._door['room_name']) + '.',
                                                  'RescueBot')
                                self._phase = Phase.ENTER_ROOM
                                self._remove = False
                                return RemoveObject.__name__, {'object_id': info['obj_id']}
                            # Remove the obstacle together if the human decides so
                            if self.received_messages_content and self.received_messages_content[
                                -1] == 'Remove together' or self._remove:
                                if not self._remove:
                                    self._answered = True
                                # Tell the human to come over and be idle untill human arrives
                                if not state[{'is_human_agent': True}]:
                                    self._send_message(
                                        'Please come to ' + str(self._door['room_name']) + ' to remove stones together.',
                                        'RescueBot')
                                    return None, {}
                                # Tell the human to remove the obstacle when he/she arrives
                                if state[{'is_human_agent': True}]:
                                    self._send_message('Lets remove stones blocking ' + str(self._door['room_name']) + '!',
                                                      'RescueBot')
                                    return None, {}
                            # Remain idle until the human communicates what to do with the identified obstacle
                            else:
                                return None, {}
                        else:
                            #Diff: move it alone
                            self._answered = True
                            self._waiting = False
                            self._send_message('Removing stones blocking ' + str(self._door['room_name']) + 'because you are not trust.',
                                               'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}

                # If no obstacles are blocking the entrance, enter the area
                if len(objects) == 0:
                    self._answered = False
                    self._remove = False
                    self._waiting = False
                    self._phase = Phase.ENTER_ROOM

            if Phase.ENTER_ROOM == self._phase:
                self._answered = False

                # Check if the target victim has been rescued by the human, and switch to finding the next goal
                if self._goal_vic in self._collected_victims:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Check if the target victim is found in a different area, and start moving there
                if self._goal_vic in self._found_victims \
                        and self._door['room_name'] != self._found_victim_logs[self._goal_vic]['room']:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Check if area already searched without finding the target victim, and plan to search another area
                if self._door['room_name'] in self._searched_rooms and self._goal_vic not in self._found_victims:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Enter the area and plan to search it
                else:
                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)
                    # If there is a valid action, return it; otherwise, plan to search the room
                    if action is not None:
                        return action, {}
                    self._phase = Phase.PLAN_ROOM_SEARCH_PATH

                #diff 搜索人声明搜索过的房间如果当前状态为testhuman
                if self.test_human and self._door['room_name'] in self.human_searched_rooms:
                    self.human_searched_rooms.remove(self._door['room_name'])

                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)
                    # If there is a valid action, return it; otherwise, plan to search the room
                    if action is not None:
                        return action, {}
                    self._phase = Phase.PLAN_ROOM_SEARCH_PATH
                    print("搜索人声明搜索过的房间如果当前状态为testhuman")
                    print(self.human_searched_rooms)

            if Phase.PLAN_ROOM_SEARCH_PATH == self._phase:
                # Extract the numeric location from the room name and set it as the agent's location
                self._agent_loc = int(self._door['room_name'].split()[-1])

                # Store the locations of all area tiles in the current room
                room_tiles = [info['location'] for info in state.values()
                             if 'class_inheritance' in info
                             and 'AreaTile' in info['class_inheritance']
                             and 'room_name' in info
                             and info['room_name'] == self._door['room_name']]
                self._roomtiles = room_tiles

                # Make the plan for searching the area
                self._navigator.reset_full()
                self._navigator.add_waypoints(self._efficientSearch(room_tiles))

                # Initialize variables for storing room victims and switch to following the room search path
                self._room_vics = []
                self._phase = Phase.FOLLOW_ROOM_SEARCH_PATH

            if Phase.FOLLOW_ROOM_SEARCH_PATH == self._phase:
                # Search the area
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    # Identify victims present in the area
                    for info in state.values():
                        if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                            vic = str(info['img_name'][8:-4])
                            # Remember which victim the agent found in this area
                            if vic not in self._room_vics:
                                self._room_vics.append(vic)

                            # DiffToDo: 检查发现地点是否与人报告相同，检查发现人员是否已被人类声明已经救了
                            if (self.test_human and ((vic in self.human_found_victims and self._door['room_name'] != self.human_found_victim_logs[vic]['room']) or \
                                    vic in self.human_collected_victims)):
                                if(vic in self.human_collected_victims):
                                    self.human_collected_victim_logs.pop(vic, None)
                                    self.human_collected_victims.remove(vic)
                                    # self.human_found_victim_logs.pop(vic, None)
                                    # self.human_found_victims.remove(vic)
                                    self._collected_victims.remove(vic)

                                # decrease trust
                                if self._door['room_name'] in self.human_searched_rooms:
                                    self.human_searched_rooms.remove(self._door['room_name'] )
                                trustBeliefs = self._trustBelief(self._team_members, trustBeliefs, self._folder, 2)
                                print("走到了 --------------  3")
                                print("检查发现地点是否与人报告相同，检查发现人员是否已被人类声明已经救了")
                                self._todo.append(vic)

                            # Identify the exact location of the victim that was found by the human earlier
                            if vic in self._found_victims and 'location' not in self._found_victim_logs[vic].keys():
                                self._recent_vic = vic
                                # Add the exact victim location to the corresponding dictionary
                                self._found_victim_logs[vic] = {'location': info['location'],
                                                                'room': self._door['room_name'],
                                                                'obj_id': info['obj_id']}
                                if vic == self._goal_vic:
                                    # Communicate which victim was found
                                    self._send_message('Found ' + vic + ' in ' + self._door[
                                        'room_name'] + ' because you told me ' + vic + ' was located here.',
                                                      'RescueBot')
                                    # Add the area to the list with searched areas
                                    if self._door['room_name'] not in self._searched_rooms:
                                        self._searched_rooms.append(self._door['room_name'])
                                    # Do not continue searching the rest of the area but start planning to rescue the victim
                                    self._phase = Phase.FIND_NEXT_GOAL

                            # Identify injured victim in the area
                            if 'healthy' not in vic and vic not in self._found_victims:
                                self._recent_vic = vic
                                # Add the victim and the location to the corresponding dictionary
                                self._found_victims.append(vic)
                                self._found_victim_logs[vic] = {'location': info['location'],
                                                                'room': self._door['room_name'],
                                                                'obj_id': info['obj_id']}
                                # Communicate which victim the agent found and ask the human whether to rescue the victim now or at a later stage
                                if 'mild' in vic and self._answered == False and not self._waiting:
                                    #Diff: 如果相信
                                    #救轻伤这件事对于能力要求低，所以只要大于0.0就行
                                    #但是是不是懒逼这个受到距离影响
                                    if (trustBeliefs[self._human_name]['competence'] > 0.0 and trustBeliefs[self._human_name][
                                        'willingness'] > 0.5 and self._distance_human == 'far') or (trustBeliefs[self._human_name]['competence'] > 0.0 and trustBeliefs[self._human_name][
                                        'willingness'] > 0.0 and self._distance_human == 'close'):

                                        self._send_message('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue together", "Rescue alone", or "Continue" searching. \n \n \
                                            Important features to consider are: \n safe - victims rescued: ' + str(
                                            self._collected_victims) + '\n explore - areas searched: area ' + str(
                                            self._searched_rooms).replace('area ', '') + '\n \
                                            clock - extra time when rescuing alone: 15 seconds \n afstand - distance between us: ' + self._distance_human,
                                                          'RescueBot')
                                        self._waiting = True
                                    else:
                                        #Diff: 不相信则自己救mild
                                        if self._door['room_name'] not in self._searched_rooms:
                                            self._searched_rooms.append(self._door['room_name'])

                                        self._send_message(
                                            'Picking up ' + self._recent_vic + ' in ' + self._door['room_name'] + 'because you are not trust.',
                                            'RescueBot')
                                        self._rescue = 'alone'
                                        self._answered = True
                                        self._waiting = False
                                        self._goal_vic = self._recent_vic
                                        self._goal_loc = self._remaining[self._goal_vic]
                                        self._recent_vic = None
                                        self._phase = Phase.PLAN_PATH_TO_VICTIM


                                if 'critical' in vic and self._answered == False and not self._waiting:
                                    self._send_message('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue" or "Continue" searching. \n\n \
                                        Important features to consider are: \n explore - areas searched: area ' + str(
                                        self._searched_rooms).replace('area',
                                                                      '') + ' \n safe - victims rescued: ' + str(
                                        self._collected_victims) + '\n \
                                        afstand - distance between us: ' + self._distance_human, 'RescueBot')
                                    self._waiting = True
                                    # Execute move actions to explore the area
                    return action, {}

                # Communicate that the agent did not find the target victim in the area while the human previously communicated the victim was located here
                if self._goal_vic in self._found_victims and self._goal_vic not in self._room_vics and \
                        self._found_victim_logs[self._goal_vic]['room'] == self._door['room_name']:
                    self._send_message(self._goal_vic + ' not present in ' + str(self._door[
                                                                                    'room_name']) + ' because I searched the whole area without finding ' + self._goal_vic + '.',
                                      'RescueBot')
                    #DiffToDo: 人撒谎，从human__found移除，decrease trust
                    self.human_found_victim_logs.pop(self._goal_vic, None)
                    self.human_found_victims.remove(self._goal_vic)
                    #decrease trust
                    self._trustBelief(self._team_members, trustBeliefs, self._folder, 2)
                    print("走到了 -------------- 4")
                    print("人撒谎，从human__found移除，decrease trust")

                    # Remove the victim location from memory
                    self._found_victim_logs.pop(self._goal_vic, None)
                    self._found_victims.remove(self._goal_vic)
                    self._room_vics = []
                    # Reset received messages (bug fix)
                    self.received_messages = []
                    self.received_messages_content = []
                # Add the area to the list of searched areas
                if self._door['room_name'] not in self._searched_rooms:
                    self._searched_rooms.append(self._door['room_name'])
                # Make a plan to rescue a found critically injured victim if the human decides so
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue' and 'critical' in self._recent_vic:
                    self._rescue = 'together'
                    self._answered = True
                    self._waiting = False
                    # Tell the human to come over and help carry the critically injured victim
                    if not state[{'is_human_agent': True}]:
                        self._send_message('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(
                            self._recent_vic) + ' together.', 'RescueBot')
                    # Tell the human to carry the critically injured victim together
                    if state[{'is_human_agent': True}]:
                        self._send_message('Lets carry ' + str(
                            self._recent_vic) + ' together! Please wait until I moved on top of ' + str(
                            self._recent_vic) + '.', 'RescueBot')
                    self._goal_vic = self._recent_vic
                    self._recent_vic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                # Make a plan to rescue a found mildly injured victim together if the human decides so
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue together' and 'mild' in self._recent_vic:
                    self._rescue = 'together'
                    self._answered = True
                    self._waiting = False
                    # Tell the human to come over and help carry the mildly injured victim
                    if not state[{'is_human_agent': True}]:
                        self._send_message('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(
                            self._recent_vic) + ' together.', 'RescueBot')
                    # Tell the human to carry the mildly injured victim together
                    if state[{'is_human_agent': True}]:
                        self._send_message('Lets carry ' + str(
                            self._recent_vic) + ' together! Please wait until I moved on top of ' + str(
                            self._recent_vic) + '.', 'RescueBot')
                    self._goal_vic = self._recent_vic
                    self._recent_vic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                # Make a plan to rescue the mildly injured victim alone if the human decides so, and communicate this to the human
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue alone' and 'mild' in self._recent_vic:
                    self._send_message('Picking up ' + self._recent_vic + ' in ' + self._door['room_name'] + '.',
                                      'RescueBot')
                    self._rescue = 'alone'
                    self._answered = True
                    self._waiting = False
                    self._goal_vic = self._recent_vic
                    self._goal_loc = self._remaining[self._goal_vic]
                    self._recent_vic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                # Continue searching other areas if the human decides so
                if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                    self._answered = True
                    self._waiting = False
                    self._todo.append(self._recent_vic)
                    self._recent_vic = None
                    self._phase = Phase.FIND_NEXT_GOAL
                    #DiffToDo: 这里也可以考虑减少trust如果当前已经对这个人不信任的情况下
                    self._trustBelief(self._team_members, trustBeliefs, self._folder, 2)
                    print("走到了 --------------  5")
                    print("这里也可以考虑减少trust如果当前已经对这个人不信任的情况下")

                # Remain idle untill the human communicates to the agent what to do with the found victim
                if self.received_messages_content and self._waiting and self.received_messages_content[
                    -1] != 'Rescue' and self.received_messages_content[-1] != 'Continue':
                    return None, {}
                # Find the next area to search when the agent is not waiting for an answer from the human or occupied with rescuing a victim
                if not self._waiting and not self._rescue:
                    self._recent_vic = None
                    self._phase = Phase.FIND_NEXT_GOAL
                return Idle.__name__, {'duration_in_ticks': 25}

            if Phase.PLAN_PATH_TO_VICTIM == self._phase:
                # Plan the path to a found victim using its location
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._found_victim_logs[self._goal_vic]['location']])
                # Follow the path to the found victim
                self._phase = Phase.FOLLOW_PATH_TO_VICTIM

            if Phase.FOLLOW_PATH_TO_VICTIM == self._phase:
                # Start searching for other victims if the human already rescued the target victim
                if self._goal_vic and self._goal_vic in self._collected_victims:
                    self._phase = Phase.FIND_NEXT_GOAL

                # Move towards the location of the found victim
                else:
                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)
                    # If there is a valid action, return it; otherwise, switch to taking the victim
                    if action is not None:
                        return action, {}
                    self._phase = Phase.TAKE_VICTIM

            if Phase.TAKE_VICTIM == self._phase:
                # Store all area tiles in a list
                room_tiles = [info['location'] for info in state.values()
                             if 'class_inheritance' in info
                             and 'AreaTile' in info['class_inheritance']
                             and 'room_name' in info
                             and info['room_name'] == self._found_victim_logs[self._goal_vic]['room']]
                self._roomtiles = room_tiles
                objects = []
                # When the victim has to be carried by human and agent together, check whether human has arrived at the victim's location
                for info in state.values():
                    # When the victim has to be carried by human and agent together, check whether human has arrived at the victim's location
                    if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'critical' in \
                            info['obj_id'] and info['location'] in self._roomtiles or \
                            'class_inheritance' in info and 'CollectableBlock' in info[
                        'class_inheritance'] and 'mild' in info['obj_id'] and info[
                        'location'] in self._roomtiles and self._rescue == 'together' or \
                            self._goal_vic in self._found_victims and self._goal_vic in self._todo and len(
                        self._searched_rooms) == 0 and 'class_inheritance' in info and 'CollectableBlock' in info[
                        'class_inheritance'] and 'critical' in info['obj_id'] and info['location'] in self._roomtiles or \
                            self._goal_vic in self._found_victims and self._goal_vic in self._todo and len(
                        self._searched_rooms) == 0 and 'class_inheritance' in info and 'CollectableBlock' in info[
                        'class_inheritance'] and 'mild' in info['obj_id'] and info['location'] in self._roomtiles:
                        objects.append(info)
                        # Remain idle when the human has not arrived at the location
                        if not self._human_name in info['name']:
                            self._waiting = True
                            self._moving = False
                            return None, {}
                # Add the victim to the list of rescued victims when it has been picked up
                if len(objects) == 0 and 'critical' in self._goal_vic or len(
                        objects) == 0 and 'mild' in self._goal_vic and self._rescue == 'together':
                    self._waiting = False
                    if self._goal_vic not in self._collected_victims:
                        self._collected_victims.append(self._goal_vic)
                    self._carrying_together = True
                    # Determine the next victim to rescue or search
                    self._phase = Phase.FIND_NEXT_GOAL

                    #DiffToDo: 人已经救了，从hunman_collected里移除，在一开始检测是否carrying_together时候已经处理，无需更多操作，此comment只作为解释说明

                # When rescuing mildly injured victims alone, pick the victim up and plan the path to the drop zone
                if 'mild' in self._goal_vic and self._rescue == 'alone':
                    self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                    if self._goal_vic not in self._collected_victims:
                        self._collected_victims.append(self._goal_vic)
                    self._carrying = True
                    return CarryObject.__name__, {'object_id': self._found_victim_logs[self._goal_vic]['obj_id'],
                                                  'human_name': self._human_name}

            if Phase.PLAN_PATH_TO_DROPPOINT == self._phase:
                self._navigator.reset_full()
                # Plan the path to the drop zone
                self._navigator.add_waypoints([self._goal_loc])
                # Follow the path to the drop zone
                self._phase = Phase.FOLLOW_PATH_TO_DROPPOINT

            if Phase.FOLLOW_PATH_TO_DROPPOINT == self._phase:
                # Communicate that the agent is transporting a mildly injured victim alone to the drop zone
                if 'mild' in self._goal_vic and self._rescue == 'alone':
                    self._send_message('Transporting ' + self._goal_vic + ' to the drop zone.', 'RescueBot')
                self._state_tracker.update(state)
                # Follow the path to the drop zone
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                # Drop the victim at the drop zone
                self._phase = Phase.DROP_VICTIM

            if Phase.DROP_VICTIM == self._phase:
                # Communicate that the agent delivered a mildly injured victim alone to the drop zone
                if 'mild' in self._goal_vic and self._rescue == 'alone':
                    self._send_message('Delivered ' + self._goal_vic + ' at the drop zone.', 'RescueBot')
                # Identify the next target victim to rescue
                self._phase = Phase.FIND_NEXT_GOAL
                self._rescue = None
                self._current_door = None
                self._tick = state['World']['nr_ticks']
                self._carrying = False
                # Drop the victim on the correct location on the drop zone
                return Drop.__name__, {'human_name': self._human_name}

    def _get_drop_zones(self, state):
        '''
        @return list of drop zones (their full dict), in order (the first one is the
        place that requires the first drop)
        '''
        places = state[{'is_goal_block': True}]
        places.sort(key=lambda info: info['location'][1])
        zones = []
        for place in places:
            if place['drop_zone_nr'] == 0:
                zones.append(place)
        return zones

    #TODO
    #我觉得应该是在这个地方改，就是说收到消息以后机器人怎么操作是取决于我的competence和willingness的
    #我随便定义了两个初始值用来这里做一个sample

    def _process_messages(self, state, teamMembers, condition, trustBeliefs, human_name):
        '''
        process incoming messages received from the team members
        '''

        receivedMessages = {}
        # Create a dictionary with a list of received messages from each team member
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        # Check the content of the received messages
        for mssgs in receivedMessages.values():
            for msg in mssgs:
                # If a received message involves team members searching areas, add these areas to the memory of areas that have been explored
                if msg.startswith("Search:"):

                    #Diff: 不接受人的信息如果trust很低
                    if(trustBeliefs[human_name]['competence'] > 0.0 and trustBeliefs[human_name]['willingness'] > 0.0):
                        area = 'area ' + msg.split()[-1]

                        #Diff: store areas that searched by human
                        if area not in self.human_searched_rooms:
                            self.human_searched_rooms.append(area)

                        # if self.initial_willingness > 0.5 and self.initial_competence > 0.5:
                            #这里是原始的行为
                            if area not in self._searched_rooms:
                                self._searched_rooms.append(area)
                    # else:
                    #     #这里是修改的行为
                    #     #应该什么都不做，因为我不信你去了
                    #     pass
                # If a received message involves team members finding victims, add these victims and their locations to memory
                if msg.startswith("Found:"):
                    # if self.initial_willingness > 0.5 and self.initial_competence > 0.5:
                    #     #可信就做原始行为
                        # Identify which victim and area it concerns
                    # Diff: 不接受人的信息如果trust很低
                    if (trustBeliefs[human_name]['competence'] > 0.0 and trustBeliefs[human_name]['willingness'] > 0.0):
                        if len(msg.split()) == 6:
                            foundVic = ' '.join(msg.split()[1:4])
                        else:
                            foundVic = ' '.join(msg.split()[1:5])
                        loc = 'area ' + msg.split()[-1]
                        # Add the area to the memory of searched areas
                        if loc not in self._searched_rooms:
                            self._searched_rooms.append(loc)
                        # Add the victim and its location to memory
                        if foundVic not in self._found_victims:
                            self._found_victims.append(foundVic)
                            self._found_victim_logs[foundVic] = {'room': loc}
                        if foundVic in self._found_victims and self._found_victim_logs[foundVic]['room'] != loc:
                            self._found_victim_logs[foundVic] = {'room': loc}
                        # Decide to help the human carry a found victim when the human's condition is 'weak'
                        if condition == 'weak':
                            self._rescue = 'together'
                        # Add the found victim to the to do list when the human's condition is not 'weak'
                        if 'mild' in foundVic and condition != 'weak':
                            self._todo.append(foundVic)
                            #Diff: store the human found victims and areas
                            if loc not in self.human_searched_rooms:
                                self.human_searched_rooms.append(loc)
                            if foundVic not in self.human_found_victims:
                                self.human_found_victims.append(foundVic)
                                self.human_found_victim_logs[foundVic] = {'room': loc}
                            if foundVic in self.human_found_victims and self.human_found_victim_logs[foundVic]['room'] != loc:
                                self.human_found_victim_logs[foundVic] = {'room': loc}
                    else:
                        #不可信的人说found实际上应该是没找到的，然后就应该自己去找
                        print('不采用你的消息')
                # If a received message involves team members rescuing victims, add these victims and their locations to memory
                if msg.startswith('Collect:'):
                    # if self.initial_willingness > 0.5 and self.initial_competence > 0.5:
                        #如果可信就真去帮忙collect受害者
                        # Identify which victim and area it concerns
                    # Diff: 不接受人的信息如果trust很低
                    if (trustBeliefs[human_name]['competence'] > 0.0 and trustBeliefs[human_name]['willingness'] > 0.0):
                        if len(msg.split()) == 6:
                            collectVic = ' '.join(msg.split()[1:4])
                        else:
                            collectVic = ' '.join(msg.split()[1:5])
                        loc = 'area ' + msg.split()[-1]
                        # Add the area to the memory of searched areas
                        if loc not in self._searched_rooms:
                            self._searched_rooms.append(loc)
                        # Add the victim and location to the memory of found victims
                        if collectVic not in self._found_victims:
                            self._found_victims.append(collectVic)
                            self._found_victim_logs[collectVic] = {'room': loc}
                        if collectVic in self._found_victims and self._found_victim_logs[collectVic]['room'] != loc:
                            self._found_victim_logs[collectVic] = {'room': loc}
                        # Add the victim to the memory of rescued victims when the human's condition is not weak
                        if condition != 'weak' and collectVic not in self._collected_victims:
                            self._collected_victims.append(collectVic)

                            #Diff: store the human collect victims
                            if loc not in self.human_searched_rooms:
                                self.human_searched_rooms.append(loc)
                            if collectVic not in self.human_collected_victims:
                                self.human_collected_victims.append(collectVic)
                                self.human_collected_victim_logs[collectVic] = {'room': loc}
                            if collectVic in self.human_collected_victims and self.human_collected_victim_logs[collectVic]['room'] != loc:
                                self.human_collected_victim_logs[collectVic] = {'room': loc}

                        # Decide to help the human carry the victim together when the human's condition is weak
                        if condition == 'weak':
                            self._rescue = 'together'
                    else:
                        # 不可信的人说found实际上应该是没找到的，然后就应该自己去找
                        print('不采用你的消息')
                # If a received message involves team members asking for help with removing obstacles, add their location to memory and come over
                if msg.startswith('Remove:'):
                    # Diff: 不接受人的信息如果trust很低
                    if (trustBeliefs[human_name]['competence'] > 0.0 and trustBeliefs[human_name]['willingness'] > 0.0):

                        # Come over immediately when the agent is not carrying a victim
                        if not self._carrying:
                            # Identify at which location the human needs help
                            area = 'area ' + msg.split()[-1]
                            self._door = state.get_room_doors(area)[0]
                            self._doormat = state.get_room(area)[-1]['doormat']
                            if area in self._searched_rooms:
                                self._searched_rooms.remove(area)
                            # Clear received messages (bug fix)
                            self.received_messages = []
                            self.received_messages_content = []
                            self._moving = True
                            self._remove = True
                            if self._waiting and self._recent_vic:
                                self._todo.append(self._recent_vic)
                            self._waiting = False
                            # Let the human know that the agent is coming over to help
                            self._send_message(
                                'Moving to ' + str(self._door['room_name']) + ' to help you remove an obstacle.',
                                'RescueBot')
                            # Plan the path to the relevant area
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                        # Come over to help after dropping a victim that is currently being carried by the agent
                        else:
                            area = 'area ' + msg.split()[-1]
                            self._send_message('Will come to ' + area + ' after dropping ' + self._goal_vic + '.',
                                              'RescueBot')
                    else:
                        print('不采用你的消息')
            # Store the current location of the human in memory
            if mssgs and mssgs[-1].split()[-1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                                                   '14']:
                self._human_loc = int(mssgs[-1].split()[-1])

    def _loadBelief(self, members, folder):
        '''
        Loads trust belief values if agent already collaborated with human before, otherwise trust belief values are initialized using default values.
        '''
        # Create a dictionary with trust values for all team members
        trustBeliefs = {}
        # Set a default starting trust value
        #在这个地方我把初值改为1了，这样的话就会人一开始是非常信任的
        default = 1
        trustfile_header = []
        trustfile_contents = []
        # Check if agent already collaborated with this human before, if yes: load the corresponding trust values, if no: initialize using default trust values
        with open(folder + '/beliefs/currentTrustBelief.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            for row in reader:
                if trustfile_header == []:
                    trustfile_header = row
                    continue
                # Retrieve trust values 
                if row and row[0] == self._human_name:
                    name = row[0]
                    competence = float(row[1])
                    willingness = float(row[2])
                    trustBeliefs[name] = {'competence': competence, 'willingness': willingness}
                # Initialize default trust values
                if row and row[0] != self._human_name:
                    competence = default
                    willingness = default
                    trustBeliefs[self._human_name] = {'competence': competence, 'willingness': willingness}
        return trustBeliefs

    def _trustBelief(self, members, trustBeliefs, folder, score):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        if (score == 1):
            trustBeliefs[self._human_name]['competence'] += 0.20
            trustBeliefs[self._human_name]['willingness'] += 0.20
        elif (score == 2):
            trustBeliefs[self._human_name]['competence'] -= 0.10
            trustBeliefs[self._human_name]['willingness'] -= 0.10
        elif (score == 3):
            trustBeliefs[self._human_name]['competence'] -= 0.00
            trustBeliefs[self._human_name]['willingness'] -= 0.20
        # elif (score == 4):
        #     trustBeliefs[self._human_name]['competence'] -= 0.20
        #     trustBeliefs[self._human_name]['willingness'] -= 0.00

            #我做的简单对应：competence == 能力，willingness == 懒逼
            #这里的逻辑是：2是撒谎，能力不行还懒   3是偷懒，我不知道能力行不行但是肯定是懒人，  4是机器人认为人能力不行，就是在人让机器人帮忙的地方
            #1是加分
        # data = []
        # with open(folder + '/beliefs/allTrustBeliefs.csv', mode='r') as csv_file:
        #     csv_reader = csv.reader(csv_file, delimiter=';', quotechar='"')
        #     for row in csv_reader:
        #         data.append(row)
        # # print(data)
        # data[-1] = [self._human_name, trustBeliefs[self._human_name]['competence'],
        #                      trustBeliefs[self._human_name]['willingness']]
        # data = data[1:]
        # print(data)
        # Save current trust belief values so we can later use and retrieve them to add to a csv file with all the logged trust belief values
        with open(folder + '/beliefs/currentTrustBeliefs.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['name', 'competence', 'willingness'])
            csv_writer.writerow([self._human_name, trustBeliefs[self._human_name]['competence'],
                              trustBeliefs[self._human_name]['willingness']])
        return trustBeliefs

    def _send_message(self, mssg, sender):
        '''
        send messages from agent to other team members
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages_content and 'Our score is' not in msg.content:
            self.send_message(msg)
            self._send_messages.append(msg.content)
        # Sending the hidden score message (DO NOT REMOVE)
        if 'Our score is' in msg.content:
            self.send_message(msg)

    def _getClosestRoom(self, state, objs, currentDoor):
        '''
        calculate which area is closest to the agent's location
        '''
        agent_location = state[self.agent_id]['location']
        locs = {}
        for obj in objs:
            locs[obj] = state.get_room_doors(obj)[0]['location']
        dists = {}
        for room, loc in locs.items():
            if currentDoor != None:
                dists[room] = utils.get_distance(currentDoor, loc)
            if currentDoor == None:
                dists[room] = utils.get_distance(agent_location, loc)

        return min(dists, key=dists.get)

    def _efficientSearch(self, tiles):
        '''
        efficiently transverse areas instead of moving over every single area tile
        '''
        x = []
        y = []
        for i in tiles:
            if i[0] not in x:
                x.append(i[0])
            if i[1] not in y:
                y.append(i[1])
        locs = []
        for i in range(len(x)):
            if i % 2 == 0:
                locs.append((x[i], min(y)))
            else:
                locs.append((x[i], max(y)))
        return locs