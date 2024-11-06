class Config:
    """Just a huge class to put all the editable config stuff for scenarios"""
    def __init__(self,
                human_first=False,
                gesture=True,
                language=False,
                num_goals=1,
                iterations=20,
                ratio = 0.85,
                pixel_threshold = 1250,
                human_id=100,
                save_path="scenario_data/",
                seed=0) :
        
        self.human_first = human_first
        self.gesture = gesture
        self.language = language
        self.num_goals = num_goals
        self.iterations = iterations
        self.ratio = ratio
        self.pixel_threshold = pixel_threshold
        self.human_id = human_id
        self.save_path = save_path
        self.seed = seed
    def to_dict(self):
        return {
            "human_first": self.human_first,
            "gesture": self.gesture,
            "language": self.language,
            "num_goals": self.num_goals,
            "iterations": self.iterations,
            "ratio": self.ratio,
            "pixel_threshold": self.pixel_threshold,
            "human_id": self.human_id,
            "save_path": self.save_path,
            "seed": self.seed
        }
    