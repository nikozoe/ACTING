#-------------------------------------------------------------------------------
# Many of the notations used in this model are adapted from David Heise's 2010
# book "Expressive Order - Confirming Sentiments in Social Actions".
# If you want to understand the details of this implementation it helps to look
# at the book part2 "Mathematics of Affect Control Theory"
# ------------------------------------------------------------------------------

import numpy as np
from mesa.space import MultiGrid
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from ACTING_parameters import *

# this is the maximum length a sampled vector is allowed to have
# it is determined by the maximum length of an EPA vector that could
# be obtained from a survey, i.e. sqrt(4.3^2 + 4.3^2 + 4.3^2), EPA=[4.3, 4.3, 4.3]
MAX_RANDOM_VECTOR_LENGTH= 7.5

# Helper functions
def random_epa(epa_center, sd=1.0):
    """ Return EPA array sampled from a normal distribution clipped at
        MAX_RANDOM_VECTOR_LENGTH.

    Args:
      epa_center: EPA-array, center of distribution from which is sampled.
      sd: standard deviation of the distribution
    """
    profile = np.random.normal(epa_center,sd)
    while np.linalg.norm(profile) > MAX_RANDOM_VECTOR_LENGTH:
        profile = np.random.normal(epa_center,sd)
    return profile


def construct_i_beta(a_f, o_f, a_t, o_t):
    """ returns I_beta (see Heise's book "Expressive Order")

    Arg:
      a_f: EPA-array for fundamentals of actor
      o_f: EPA-array for fundamentals of object
      a_t: EPA-array for transients of actor
      o_f: EPA-array for transients of object
    """
    part1 = np.hstack((a_f, 1,1,1, o_f))
    part2 = make_act_products(a_t, np.ones(3), o_t)
    return np.hstack((part1, part2))


def make_act_products(a, b, o):
    """returns 64-dimensional vector that linearly enters the ACT equations
       for the transients.

    Args:
      a: EPA-array for actor
      b: EPA-array for behavior
      o: EPA-array for object
    """
    part1 = np.hstack((1, a, b, o))
    #(10)AeBe AeBp AeBa AeOe AeOp AeOa (16)ApBe ApBp ApBa ApOe ApOp ApOa
    # (22)AaBe AaBp AaBa AaOe AaOp AaOa
    part2 = np.hstack((np.outer(a,b), np.outer(a,o))).flatten()
    #(28)BeOe BeOp BeOa BpOe BpOp BpOa BaOe BaOp BaOa
    part3 = np.outer(b,o).flatten()
    # (37)AeBeOe AeBeOp AeBeOa AeBpOe AeBpOp AeBpOa AeBaOe AeBaOp AeBaOa
    #(46)ApBeOe ApBeOp ApBeOa ApBpOe ApBpOp ApBpOa ApBaOe ApBaOp ApBaOa
    #(55)AaBeOe AaBeOp AaBeOa AaBpOe AaBpOp AaBpOa AaBaOe AaBaOp
    part4 = np.outer(np.outer(a,b).flatten(),o).flatten()

    return np.hstack((part1, part2, part3, part4))


def compute_transients(a,b,o, abo_coefficients):
    """ returns transients of an agent according to ACT, given EPA values
        for a,b,o and the abo_coefficients

    Args:
      a: EPA-array for actor
      b: EPA-array for behavior
      o: EPA-array for object
    """
    t = make_act_products(a,b,o)

    return np.round(np.dot(abo_coefficients[:,9:], t),2)


def compute_opt_behavior(actor, obj):
    """Returns optimal behavior (according to ACT) given that actor and object
       have been chosen

    Args:
      actor: Agent object
      obj: Agent object
    """
    actions = actor.model.discrete_actions
    if not (actions is None):
        # discrete list of actions
        min_deflection = np.inf
        beh = None
        for action in actions:
            if len(action) == 2 and obj.unique_id != -1:
                # action has network restriction:
                if action[1][actor.unique_id, obj.unique_id] == 0:
                    continue
                else:
                    action_epa = action[0]
            else:
                #no network restriction
                # the if clause is necessary for action on group
                if len(action) == 2:
                    action_epa = action[0]
                else:
                    action_epa = action
            transients = compute_transients(actor.current_transients, action_epa,
                                            obj.current_transients,
                                            actor.model.abo_coefficients)
            fundamentals = np.concatenate([actor.fundamentals, action_epa,
                                           obj.fundamentals])
            deflection = np.linalg.norm(fundamentals - transients)**2
            if deflection < min_deflection:
                min_deflection = deflection
                beh = action_epa
    else:
        i_beta = construct_i_beta(actor.fundamentals, obj.fundamentals,
                                  actor.current_transients,
                                  obj.current_transients)

        sim = np.dot(np.dot(s_beta, np.diag(i_beta)),
                     np.transpose(actor.model.abo_coefficients))
        gim = np.dot(actor.model.abo_coefficients,g*i_beta)
        simsim = np.dot(sim, np.transpose(sim))
        simgim = np.dot(sim, gim)
        beh = - np.dot(np.linalg.inv(simsim), simgim)
    return np.round(beh,2)


def compute_bales(beh_epa, IPA_EPAs = IPA_EPAs_1978):
    """returns Bales category for EPA value based on minimum distance.
       Note: IPA_EPAs are set to 1978 data set (IPA_EPAs_1978).
       Set IPA_EPAs = IPA_EPAs_2004
       for more recent dataset or pass custom numpy array of shape (12,3).

    Args:
      beh_epa: EPA array of behavior for which we want the closes Bales cat.
      IPA_EPAs: array of shape (12,3) representing the EPA values of Bales cat.
    """
    return ( np.argmin([np.linalg.norm(beh_epa - bales_cat)
                        for bales_cat in IPA_EPAs]) + 1 )


class GroupModel(Model):
    """
    The Group model

    Attributes:
        agents (dict or list of dicts):
            if passing a list of dictionaries -> [ {'epa' : [e,p,a],
                                                  'initial_tension': i_t},...]
            if passing  a single dictionary -> {'N': N, 'epa' : [e,p,a],
                                                'initial_tension': i_t,
                                                'individuality': ind}
        data_model (string):
            Determines which set of ACT equations to use for the simulation.
            Current options are: 'us_unisex', 'us_male', 'us_female',
            'canada_unisex', 'canada_male', 'canada_female', 'china_unisex',
            'ger_unisex'
        reciprocity_rate (float, optional):
            Probabilty for the an action to be reciprocal. Default 0.2.
        actor_choice (string, optional):
            sets the criterion on which the next actor is chosen.
            The default is 'max self-tension' which is currently also the only
            one implemented. In the future we might implement other options.
        object_choice (string, optional):
            sets the criterion on which the next object is chosen.
            The default is 'min event-tension' which selects object and behavior
            so that the sum of deflections for actor, behavior and object after
            the event is minimized relative to their fundamentals.
            'max deflection-reduction' selects object and behavior so that
            as much deflection as possible is reduced relative to before the event.
            'random' selects randomly among object candidates.
        action_on_group (Bool, optional):
            determines whether actions on the whole group are possible
        group_action_rate (float, optional):
            if passed, sets the propability for the next action to be on the
            whole group only makes sense if action_on_group is True.
        network_structure (tuple, optional):
            if interactions are restricted to a certain network structure,
            pass adjacency matrix in the form tuple of tuples ((),...())
        discrete_actions (list, optional):
            list of allowed actions of format [[e,p,a],...]
            if there is a network restriction on the actions the format is
            [[[e,p,a], network],...]
        seed (int, optional):
            seed passed to numpy.random to make simulation reproducible
    """

    def __init__(self, agents, data_model,
                 reciprocity_rate = 0.0,
                 actor_choice = "max self-tension",
                 object_choice = "min event-tension",
                 action_on_group = False,
                 group_action_rate = 0.0,
                 network_structure = None,
                 discrete_actions = None,
                 seed = None,
                 IPAs = IPA_EPAs_1978):
        self.running = True
        #set random seed, if given, to make simulations reproducible
        np.random.seed(seed=seed)
        #create agent list
        if isinstance(agents,list) or isinstance(agents, tuple):
            self.initial_agents = agents
            for i,ag in enumerate(self.initial_agents):
                if 'individuality' not in ag:
                    ag['individuality'] = 0.0
            self.num_agents = len(agents)
        elif isinstance(agents, dict):
            self.num_agents = agents['N']
            self.initial_agents = [{"epa": agents["epa"],
                                   "initial_tension": agents['initial_tension'],
                                   "individuality": agents["individuality"]}
                                   for i in range(agents["N"])]
        else:
            print("""as agents pass either a list of dictionaries
                     [ {'epa' : [e,p,a], 'initial_tension': i_t},...]
                     or a single dictionary
                     {'N': N, 'epa' : [e,p,a], 'initial_tension': i_t,
                      'individuality': ind}
                  """)
        self.schedule = RandomActivation(self)
        self.reciprocity_rate= reciprocity_rate
        self.reciprocal = False
        self.abo_coefficients = abo_coefficients_dict[data_model]
        self.network_matrix = np.zeros((self.num_agents,self.num_agents))
        self.actor_choice = actor_choice
        self.object_choice = object_choice
        self.action_on_group = action_on_group
        self.group_action_rate = group_action_rate
        # initialize network structure
        if network_structure is None:
            self.network_structure = None
        else:
            self.network_structure = np.array(network_structure)
        # set of allowed actions, if all (continous) actions are allowed
        # discrete_actions is set to None (default)
        self.discrete_actions = discrete_actions

        # Create agents
        for i, agent in enumerate(self.initial_agents):
            a = GroupMember(i, self, agent['epa'], agent['individuality'],
                            agent['initial_tension'])
            self.schedule.add(a)

        # if actions on the whole group are allowed, initialize the group
        if action_on_group:
            fundamentals = np.mean(
                [ag.fundamentals for ag in self.schedule.agents],
                axis=0)
            self.group = Group(fundamentals, fundamentals)

        # initial values
        self.actor = np.random.choice(self.schedule.agents)
        if self.action_on_group:
            # this prevents reciprocal action as first action
            self.object=self.group
        else:
            # random object
            self.object = np.random.choice(self.schedule.agents)
        self.action = np.zeros(3)

        # collect data
        self.datacollector = DataCollector(
            model_reporters={"actor": lambda x: x.actor.unique_id,
                             "action_E": lambda x: x.action[0],
                             "action_P": lambda x: x.action[1],
                             "action_A": lambda x: x.action[2],
                             "bales_category": lambda x: compute_bales(x.action, IPAs),
                             "object": lambda x: x.object.unique_id,
                            "reciprocal": "reciprocal"},
            agent_reporters={"Deflection": "personal_deflection",
                             "E": lambda x: x.current_transients[0],
                             "P": lambda x: x.current_transients[1],
                             "A": lambda x: x.current_transients[2]}
        )

        # record initial data of agents
        agent_records = self.datacollector._record_agents(self)
        self.datacollector._agent_records[self.schedule.steps] = list(agent_records)

    def select_actor(self):
        """ select next actor according to actor selection criterion,
            reciprocity probability and network structure.
            Next actor is determined by setting self.actor and subsequently
            self.actor.action = True"""
        # check actor choice exists and use default if not
        possible_actor_choices = ['max self-tension']
        if self.actor_choice not in possible_actor_choices:
            print('actor choice ', self.actor_choice,
                  'does not exist. fall back to max self-tension' )
            self.actor_choice = 'max self-tension'
        self.reciprocal = False
        if self.actor_choice == 'max self-tension':
            # check that last action was not on group
            # check that network structure permits reciprocal action
            if (self.action_on_group and self.object.unique_id != -1
                    and not (self.network_structure is None)
                    and ( self.network_structure[self.object.unique_id,
                                                 self.actor.unique_id] == 0) ):
                reciprocal_ok = False
            else:
                reciprocal_ok = True
            if (self.object.unique_id != -1
                    and reciprocal_ok
                    and np.random.random_sample() < self.reciprocity_rate):
                #reciprocal action
                self.actor, self.object  =  self.object, self.actor
                self.actor.acting = True
                self.reciprocal = True
            else:
                #non-reciprocal action
                self.actor = max(self.schedule.agents,
                                 key=lambda ag: ag.personal_deflection)
                self.actor.acting = True

    def step(self):
        self.select_actor()
        self.schedule.step()
        self.datacollector.collect(self)


class GroupMember(Agent):
    """
    Group member class.

    Attributes:
        unique_id (int): unique id to recognize group member
        model (GroupModel obj): mesa model according to which agents act
        fundamentals (list): [e, p, a] values of agent's fundamental identity
        individuality (float): If >0 the fundamentals are drawn from a clipped
                               normal distribution centered at fundamentals
                               parameter above with individuality as the
                               standard deviation
        initial_tension (float): agent's initial transients are drawn from a
                                 (clipped) normal distribution centered around
                                 the fundamental with intial_tension as the
                                 standard deviation
    """

    def __init__(self, unique_id, model, fundamentals, individuality, initial_tension):
        super().__init__(unique_id, model)

        #set fundamentals
        if individuality > 0:
            self.fundamentals = random_epa(fundamentals, individuality)
        else:
            self.fundamentals = np.array(fundamentals)

        # set transients
        if initial_tension > 0:
            self.current_transients = random_epa(self.fundamentals, initial_tension)
        else:
            self.current_transients = self.fundamentals

        self.acting = False
        self.model = model

        self.update_deflection()

    def update_deflection(self):
        self.personal_deflection = np.linalg.norm(self.fundamentals
             - self.current_transients)**2

    def find_best_object_and_behavior(self):
        """Finds best object and behavior for a given actor(self) according to ACT

           Returns:
              tuple: (object, action, transients)
        """
        potential_event_deflection = np.inf

        if not (self.model.network_structure is None):
            # check if network structure was passed and only consider
            # candidate objects to which actor has link
            possible_ids = [i for i,link
                            in enumerate(self.model.network_structure[self.unique_id])
                            if link == 1]
            potential_objects = [ag for ag in self.model.schedule.agents
                                 if ag.unique_id in possible_ids]
            if self.model.action_on_group:
                potential_objects += [self.model.group]
        else:
            potential_objects = [ag for ag in self.model.schedule.agents
                                 if ag.unique_id != self.unique_id]
            if self.model.action_on_group:
                potential_objects += [self.model.group]

        if self.model.object_choice == "random":
            potential_object = np.random.choice(potential_objects)
            potential_action = compute_opt_behavior(self, potential_object)
            potential_transients = compute_transients(
                                    self.current_transients,
                                    potential_action,
                                    potential_object.current_transients,
                                    self.model.abo_coefficients)
        else:
            np.random.shuffle(potential_objects)

            for obj in potential_objects:
                opt_beh = compute_opt_behavior(self, obj)
                transients = compute_transients(self.current_transients,
                                                opt_beh, obj.current_transients,
                                                self.model.abo_coefficients)

                if self.model.object_choice == "min event-tension":
                    fundamentals_stack = np.hstack((self.fundamentals, opt_beh,
                                                    obj.fundamentals))
                    event_deflection = np.linalg.norm(fundamentals_stack
                                                      - transients)**2

                elif self.model.object_choice == "max deflection-reduction":
                    actor_deflection_diff =  (
                        np.linalg.norm(self.fundamentals - transients[:3])**2
                        - self.personal_deflection
                        )
                    beh_deflection = np.linalg.norm(opt_beh - transients[3:6])**2
                    # we include the actual deflection reduction of all agents
                    # in the object choice, not just the group object
                    if obj.unique_id == -1:
                        object_deflection_diff = (
                            np.linalg.norm(obj.fundamentals - transients[6:])**2
                            - obj.personal_deflection
                            )
                        for ag in [agent for agent in self.model.schedule.agents
                                   if agent.unique_id != self.unique_id]:
                            transients_g = compute_transients(
                                self.current_transients, opt_beh,
                                ag.current_transients,
                                self.model.abo_coefficients
                                )
                            object_deflection_diff+= (
                                np.linalg.norm(ag.fundamentals - transients_g[6:])**2
                                - ag.personal_deflection
                                )
                    else:
                        object_deflection_diff = (
                            np.linalg.norm(obj.fundamentals - transients[6:])**2
                            - obj.personal_deflection
                            )

                    event_deflection = ( actor_deflection_diff
                                        + beh_deflection
                                        + object_deflection_diff )
                else:
                    print("unknown or no object selection criterion specified")

                # check if deflection for this behavior is smaller
                # than for other behaviors
                if event_deflection < potential_event_deflection:
                    potential_event_deflection = event_deflection
                    potential_transients = transients
                    potential_object = obj
                    potential_action = opt_beh

        return (potential_object, potential_action, potential_transients)

    def act(self):
        """if agent gets to act, this is what they do"""
        #reciprocal action
        if self.model.reciprocal:
            action = compute_opt_behavior(self, self.model.object)
            if not (action is None):
                transients = compute_transients(
                    self.current_transients, action,
                    self.model.object.current_transients,
                    self.model.abo_coefficients
                    )

                # "carry out" the optimal action and update model,
                # transients and deflection
                self.model.action = action
                self.current_transients = transients[:3]
                self.update_deflection()

                #set new transients and deflection on behavior object
                self.model.object.current_transients = transients[6:]
                self.model.object.update_deflection()

        # action on whole group
        elif (self.model.action_on_group and np.random.random_sample()
                  < self.model.group_action_rate):
            action = compute_opt_behavior(self, self.model.group)
            if not (action is None):

                #update transients and deflection for all agents
                for ag in self.model.schedule.agents:
                    if ag.unique_id != self.unique_id:
                        transients = compute_transients(
                            self.current_transients, action,
                            ag.current_transients, self.model.abo_coefficients
                            )
                        ag.current_transients = transients[6:]
                        ag.update_deflection()

                #action on whole group pseudoagent
                transients = compute_transients(
                    self.current_transients, action,
                    self.model.group.current_transients,
                    self.model.abo_coefficients
                    )
                self.model.action = action
                self.current_transients = transients[:3]
                self.update_deflection()

                #set new transients on group object
                self.model.group.current_transients = transients[6:]
                self.model.group.update_deflection()
                self.model.object = self.model.group
        else:
            #find best action and object
            obj, action, transients = self.find_best_object_and_behavior()

            # if object is group object,
            # apply action to all group members individually
            if obj.unique_id == -1:
                for ag in self.model.schedule.agents:
                    if ag.unique_id != self.unique_id:
                        transients_g = compute_transients(
                            self.current_transients, action,
                            ag.current_transients, self.model.abo_coefficients
                            )
                        ag.current_transients = transients_g[6:]
                        ag.update_deflection()

            #"carry out" the optimal action and update model
            self.model.action = action
            self.current_transients = transients[:3]
            self.update_deflection()

            #set new transients and deflection on behavior object
            self.model.object = obj
            self.model.object.current_transients = transients[6:]
            self.model.object.update_deflection()

        self.acting = False

    def step(self):
        #check if agent is actor this round and if so -> act
        if self.acting == True:
            self.act()
        else:
            pass


class Group(Agent):
    """Group object class. The group has id -1."""
    def __init__(self,fundamentals, transients):
        self.unique_id = -1
        self.fundamentals = fundamentals
        self.current_transients = transients
        self.personal_deflection = np.linalg.norm(fundamentals - transients)**2
    def update_deflection(self):
        self.personal_deflection = np.linalg.norm(self.fundamentals
                                                  - self.current_transients)**2


class ProtocolGroupModel(GroupModel):
    """
    A protocol can be defined that allows to customize the group model to
    specific task groups. Additionally all other parameters from GroupModel can
    be initialized.

    protocol (list of dicts, mandatory):
        Each dicts represents a part of the protocol and holds a mandatory
        parameter n_iterations, that specifies  for how many steps the part is run.
        if n_iterations holds an array of 2 numbers specifying a range, then
        n_iterations will randomly be set to a number in that range. The rest of
        the parameters are then used to overwrite the original group model
        settings for the duration of this protocol part.
        The protocol runs in a loop. There are two optional special parameters:
        1. select_actor (str): If set, the string is interpreted as python code
        and overwrites the select_actor function from the GroupModel.
        2. code (tuple of str):  will execute strings as code, code[0] before
        step and code[1] after step."""

    def __init__(self,
                 agents,
                 data_model,
                 reciprocity_rate = 0.0,
                 actor_choice = "max self-tension",
                 object_choice = "min event-tension",
                 action_on_group = False,
                 group_action_rate = 0.0,
                 network_structure = None,
                 discrete_actions = None,
                 seed = None,
                 IPAs = IPA_EPAs_1978,
                 protocol = None):
        super().__init__(agents,
                         data_model,
                         reciprocity_rate,
                         actor_choice,
                         object_choice,
                         action_on_group,
                         group_action_rate,
                         network_structure,
                         discrete_actions,
                         seed,
                         IPAs)
        self.protocol = protocol
        self.protocol_part = 0
        self.protocol_step = 0
        self.protocol_max_step = 0

    def step(self):
        # at the beginning of protocol part, set parsed arguments for group model
        if self.protocol_step == 0:
            for param, val in self.protocol[self.protocol_part].items():
                if param not in ['n_iterations', 'code', "select_actor"]:
                    setattr(self, param, val)
            # check if n_iterations is number or range, if range pick a number
            if type(self.protocol[self.protocol_part]["n_iterations"])==int:
                self.n_iterations = self.protocol[self.protocol_part]["n_iterations"]
            elif len(self.protocol[self.protocol_part]["n_iterations"])==2:
                self.n_iterations = np.random.randint(
                    self.protocol[self.protocol_part]["n_iterations"][0],
                    self.protocol[self.protocol_part]["n_iterations"][1]
                    )

            # if parsed, execute additional code at beginning of protocol part
            if "code" in self.protocol[self.protocol_part]:
                exec(self.protocol[self.protocol_part]['code'][0])

        # step counter
        self.protocol_step+=1

        #if custom select_actor exists, use
        if "select_actor" in self.protocol[self.protocol_part]:
            exec(self.protocol[self.protocol_part]["select_actor"])
        else:
            self.select_actor()

        self.schedule.step()
        self.datacollector.collect(self)

        #at end of protocol part
        if self.protocol_step == self.n_iterations:
            self.protocol_step = 0

            #additional code at end of protocol part
            if "code" in self.protocol[self.protocol_part]:
                exec(self.protocol[self.protocol_part]['code'][1])

            if self.protocol_part == len(self.protocol)-1:
                self.protocol_part = 0
            else:
                self.protocol_part+=1
