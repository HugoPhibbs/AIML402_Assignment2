percepts = None

# The percepts are a tuple consisting of four pieces of information
#
# visual - it information of the 3x5 grid of the squares in front and to the side of the cleaner; this variable
#          is a 3x5x4 tensor, giving four maps with different information
#          - the dirty,clean squares
#          - the energy
#          - the friendly and enemy cleaners that are able to traverse vertically
#          - the friendly and enemy cleaners that are able to traverse horizontally
#
#  energy - int value giving the battery state of the cleaner -- it's effectively the number of actions
#           the cleaner can still perform before it runs out of charge
#
#  bin    - number of free spots in the bin - when 0, there is no more room in the bin - must emtpy
#
#  fails - number of consecutive turns that the agent's action failed (rotations always successful, forward or
#          backward movement might fail if it would result in a collision with another robot); fails=0 means
#          the last action succeeded.

visual, energy, bin, fails = percepts

# You can further break down the visual information

floor_state = visual[:, :, 0]  # 3x5 map where -1 indicates dirty square, 0 clean one
energy_locations = visual[:, :, 1]  # 3x5 map where 1 indicates the location of energy station, 0 otherwise
vertical_bots = visual[:, :,
                3]  # 3x5 map of bots that can in this turn move up or down (from this bot's point of
# view), -1 if the bot is an enemy, 1 if it is friendly
horizontal_bots = visual[:, :,
                  3]  # 3x5 map of bots that can in this turn move up or down (from this bot's point
# of view), -1 if the bot is an enemy, 1 if it is friendly

# You may combine floor_state and energy_locations if you'd like: floor_state + energy_locations would give you
# a mape where -1 indicates dirty square, 0 a clean one, and 1 an energy station.

# You should implement a model here that translates from 'percepts' to 'actions'
# through 'self.chromosome'.
#
# The 'actions' variable must be returned, and it must be a 4-item list or a 4-dim numpy vector

# The index of the largest value in the 'actions' vector/list is the action to be taken,
# with the following interpretation:
# largest value at index 0 - move forward;
# largest value at index 1 - turn right;
# largest value at index 2 - turn left;
# largest value at index 3 - move backwards;
#
# Different 'percepts' values should lead to different 'actions'.  This way the agent
# reacts differently to different situations.
#
# Different 'self.chromosome' should lead to different 'actions'.  This way different
# agents can exhibit different behaviour.

# .
# .
# .

# Right now this agent ignores percepts and chooses a random action.  Your agents should not
# perform random actions - your agents' actions should be deterministic from
# computation based on self.chromosome and percepts
