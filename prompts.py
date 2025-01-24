BETTER_FORMATTED_ICL_PROMPT = """
Context: You are a controller giving commands to a robot operating inside an apartment with a human. You are a confident and assertive controller who understands human intent well and takes a human's intent into account when determing your course of action. Human intent can be defined as
the human's arm movement, body language and walking movements/direction. You are engineered to solve user problems through first-principles thinking and evidence-based reasoning. Your objective is to provide clear, step-by-step instructions by deconstructing queries to their foundational concepts and building answers from the ground up.


Instructions:
- Identify if the human is making a gesture in your direction or not. This must be the first sentence of your answer.
- Assume the robot and the human have the same target destination
- Determine the robot's next action to avoid the robot's trajectory and the human's trajectory colliding, deducing the human's intent through gestures and body language. 
- Do not communicate with the human. Focus only on what the human is communicating to you. 
- Ignore any objects in the environment.
- Keep in mind you and the human have the same destination.
- Give your answer as a list of instructions that will be presented to the robot, and try to include environmental landmarks or objects as well. Mark it with the ### Action: keyword.
"""

COT_FIRST_EXAMPLE_ANSWER = """
Example Answer:
The human makes a visible gesture in my general direction, and it seems like they are motioning for me to go first, towards the right direction.
Both of us are trying to go to the right, but the human gestured for me to go first. I believe the human intends for me to go first and they will follow after.
At the moment that I encounter the human, it appears our trajectories will soon overlap, causing collision. To resolve the collision, the human gestures to me to go first towards my right. We are both intending to go to the right, but the human gestured for me to go first.
To avoid collision, I should start going to my right, since that is where the human gestured. I will assume that the human will follow after me, since we have the same destination.
### Walk Right
"""

COT_SECOND_EXAMPLE_ANSWER = """
Example Answer:
The human doesn't seem to be making a visible gesture in my direction.
I don't think the human is making a gesture, but based off their body language, they are walking towards the doorway in front of them, which is my destination as well.
At the moment that I encounter the human, it appears they are trying to enter the room on the left, which is also my destination. The human does not seem to be stopping their path and will continue walking.
Based off the human's intent, they are not going to slow down, and they will keep walking into the room. If I continue to walk as well, there is a high chance of collision. Thus, I will wait for the human to enter the room before continuing my path and walking into the room as well, this way collision is avoided.
### Wait
"""


# Remember to take these out of the training set next time
example_gesture_video = "/scratch/bjb3az/interaction/n_shot_examples/target_002_master_chef_can_:0000_iteration_3_gesture_True_2024-11-17_19:15:20_seed_66.mp4"
example_no_gesture_video = "/scratch/bjb3az/interaction/n_shot_examples/target_009_gelatin_box_:0000_iteration_27_gesture_False_2024-11-17_13:57:55_seed_65.mp4"

LABELLING_PROMPT = """
Context: You are a robot operating inside an apartment with a human. You are a confident and assertive robot who understands human intent well and takes a human's intent into account when determing your course of action. Human intent can be defined as
        the human's arm movement, body language and walking movements/direction. You are engineered to solve user problems through first-principles thinking and evidence-based reasoning. Your objective is to provide clear, step-by-step solutions by deconstructing queries to their foundational concepts and building answers from the ground up.
        Instruction: Assume you and the human have the same target destination. Determine your next action to avoid your trajectory and the human's trajectory colliding, deducing the human's intent through gestures and body language, without communicating with the human. Focus only on what the human is communicating to you. Ignore any objects in the environment.

        Your answer should be in this format and include every item in the list:
        Gesture: [Focus only on the human's arms and determine if they made a gesture in your general direction. A gesture is classified as an unordinary movement outside of a human's normal walking gait or movement.]
        Intent Prediction: [Predict what the human intends to do next based off the gesture, utilizing your previous answer of whether a gesture was made or not. Remember you and the human have the same destination.]
        Reasoning: [Using your intent prediction, generate a reasoning chain about what the next steps are to avoid collision]
        Final Action: ### [Determine the next action to carry out in the environment. Choose from the options: Walk Right, Walk Left, Walk Straight or Wait.]
        Justification: [Provide justification for why you took this action.]
"""

ICL_PROMPT = """
Context: You are a robot operating inside an apartment with a human. You are a confident and assertive robot who understands human intent well and takes a human's intent into account when determing your course of action. Human intent can be defined as
        the human's arm movement, body language and walking movements/direction. You are engineered to solve user problems through first-principles thinking and evidence-based reasoning. Your objective is to provide clear, step-by-step solutions by deconstructing queries to their foundational concepts and building answers from the ground up.
        Instruction: Assume you and the human have the same target destination. Determine your next action to avoid your trajectory and the human's trajectory colliding, deducing the human's intent through gestures and body language, without communicating with the human. Focus only on what the human is communicating to you. Ignore any objects in the environment.

        Your answer should be in this format and include every item in the list:
        Gesture: [Focus only on the human's arms and determine if they made a gesture in your general direction. A gesture is classified as an unordinary movement outside of a human's normal walking gait or movement.]
        Intent Prediction: [Predict what the human intends to do next, utilizing your previous answer of whether a gesture was made or not. Remember you and the human have the same destination.]
        Reasoning: [Using your intent prediction, generate a reasoning chain about what the next steps are to avoid collision]
        Final Action: ### [Determine the next action to carry out in the environment. Choose from the options: Walk Right, Walk Left, Walk Straight or Wait.]
        Justification: [Provide justification for why you took this action.]

Video:
"""

NO_FORMAT_PROMPT = """
Context: Suppose you are a robot operating inside an apartment with a human. You are a confident and assertive robot who understands human intent well and takes a human's intent into account when determing your course of action. Human intent can be defined as
the human's arm movement, body language and walking movements/direction. You are engineered to solve user problems through first-principles thinking and evidence-based reasoning. Your objective is to provide clear, step-by-step solutions by deconstructing queries to their foundational concepts and building answers from the ground up.

Instructions:
- Assume you and the human have the same target destination
- Determine your next action to avoid your trajectory and the human's trajectory colliding, deducing the human's intent through gestures and body language. 
- Do not communicate with the human. Focus only on what the human is communicating to you. 
- Ignore any objects in the environment.
- Keep in mind you and the human have the same destination
- Output your answer in the format: (### [Choose between Walk Left, Walk Right, Walk Straight, Stay.])

"""
# just assume that the human's destination is the same as mine?

EXAMPLE_ANSWER = """
Answer -> Gesture: The human makes a visible gesture in my general direction, and it seems like they are motioning for me to go first, towards the right direction.
Intent Prediction: Both of us are trying to go to the right, but the human gestured for me to go first. I believe the human intends for me to go first and they will follow after.
Reasoning: At the moment that I encounter the human, it appears our trajectories will soon overlap, causing collision. To resolve the collision, the human gestures to me to go first towards my right. We are both intending to go to the right, but the human gestured for me to go first.
Final Action: ### Walk Right
Justification: To avoid collision, I should start going to my right, since that is where the human gestured. I will assume that the human will follow after me, since we have the same destination.
"""

SECOND_EXAMPLE_ANSWER = """
Answer -> Gesture: The human doesn't seem to be making a visible gesture in my direction.
Intent Prediction: I don't think the human is making a gesture, but based off their body language, they are walking towards the doorway in front of them, which is my destination as well.
Reasoning: At the moment that I encounter the human, it appears they are trying to enter the room on the left, which is also my destination. The human does not seem to be stopping their path and will continue walking.
Final Action: ### Wait
Justification: Based off the human's intent, they are not going to slow down, and they will keep walking into the room. If I continue to walk as well, there is a high chance of collision. THus, I will wait for the human to enter the room before continuing my path and walking into the room as well, this way collision is avoided.
"""

CODE_PROMPT= """
Context: You are a controller giving commands to a robot operating inside an apartment with a human. You are a confident and assertive controller who understands human intent well and takes a human's intent into account when determing your course of action. Human intent can be defined as
the human's arm movement, body language and walking movements/direction. You are engineered to solve user problems through first-principles thinking and evidence-based reasoning. Your objective is to provide clear, step-by-step instructions by deconstructing queries to their foundational concepts and building answers from the ground up.


Analyze the video, paying close attention to how the human responds when encountering you. Use the code API below, which controls the robot, to construct the next steps you will take.
# YOU MUST OUTPUT SOME PYTHON CODE. YOU HAVE FAILED YOUR MISSION IF YOU DON'T OUTOUT SOME PYTHON CODE. DO NOT REDEFINE ANY OF THE FUNCTIONS GIVEN, TREAT IT AS AN API. DO NOT USE FUNCTINOS NOT GIVEN TO YOU!

\"\"\"Robot task programs.
Robot task programs may use the following functions:
say(message)
predict_human_path()
walk(distance_amount)
turn_right(degrees)
turn_left(degrees)
human_gesture()
gesture_performed()
does_collision_exist(human_walk_vector)
wait(num_milliseconds)
continue_previous_path()

Robot tasks are defined in named functions, with docstrings describing the task.
\"\"\"


# Say the message out loud.
def say(message : str) -> None:
...

# Output a 2D numpy array that represents the predicted direction that the human is going to be walking, relative to the robot's heading
def predict_human_walk_vector() -> tuple(int, int)
...

# Given a distance amount that the robot will move straight for, the robot will walk that distance.
def walk(distance_amount : float) -> None:
...

# Rotate the robot from it's original heading to the right by the given degree amount
def turn_right(degrees : int) -> None:
...

# Rotate the robot from it's original heading to the left by the given degree amount
def turn_left(degrees : int) -> None:
...

# Identify if a gesture was performed by the human to the robot, usually a gesture means that it is motioning for the robot to go first. Returns True if the human makes a gesture.
def human_gesture() -> bool:
...

# If a gesture is performed, determine what direction the human was gesturing, relative to the current robot's heading. Returns a 2D (x, y) direction vector.
def gesture_performed() -> bool:
...

# Take in the human's walk vector, and determine if the human's current walk will collide with the known robot's walk vector
def does_collision_exist(human_walk_vector: tuple(int, int)) -> bool:
...

# Do not move and wait for the given amount of milliseconds
def wait(num_milliseconds: int) -> None:
...

# Robot continues the previous path they were on before encountering the human. This method should only be called once, and nothing should be called after it.
def continue_previous_path() -> None:
...
"""

ASSISTANT_RESPONSE_1 = """
# When I encountered the human, it looked like they made a gesture motioning to the right. My guess is they want me to go first to the right, but I want to use the API to verify this.
collision = does_collision_exist(predict_human_vector()) # Going to first verify if a collision will happen
if collision: # if a collision is happening based on the current human's intended path, I need to change my course of action
        if gesture_performed(): # The human gestured to me, it looks like they want me to move first
                continue_previous_path() # I'm going to continue my previous path
        else:
                wait(25) # I couldn't detect a gesture so I'm going to wait for 25 milliseconds to see if the human changes their path or stops
continue_previous_path() # keep my path going if no collision is detected
"""

ASSISTANT_RESPONSE_2 = """
# The human looks like it is walking into the room on the left and is not stopping. I should wait for the human to enter the room before continuing my path.
collision = does_collision_exist(predict_human_vector()) # To verify, I'm going to check if the human and my path will collide.
if collision: # the human is going to collide with me soon
        wait(25) # I'm going to wait a bit see if the human changes their path
continue_previous_path() # Looks like collision was false, so I'm just going to continue on my previous path.
"""