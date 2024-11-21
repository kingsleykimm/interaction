from config import Config
from scenarios import SocialNavScenario
import argparse

parser = argparse.ArgumentParser(description='Run the social navigation scenario')
parser.add_argument("--human_first", help="Whether the human goes first", action="store_true")
parser.add_argument("--gesture", help="Whether the human gestures", action="store_true")
parser.add_argument("--language", help="Whether the human speaks", action="store_true")
parser.add_argument("--num_goals", type=int, help="Number of targets", default=1)
parser.add_argument("--iterations", type=int, help="Number of iterations for each target", default=1)
parser.add_argument("--ratio", type=float, help="Ratio of area of robot view to be considered for human detection", default=0.85)
parser.add_argument("--pixel_threshold", type=int, help="Threshold for human detection", default=1000)
parser.add_argument("--human_id", type=int, help="Sematnic ID of the human in the scene", default=100)
parser.add_argument("--save_path", type=str, help="Path to save the data", default="scenario_data/")
parser.add_argument("--seed", type=int, help="Random seed", default=0)
parser.add_argument("--env", type=int, help="Which config to use from list of configs in env_setup", default=0)
args = parser.parse_args()

def main():
    scenario_config = Config(args.human_first,
                             args.gesture,
                            args.language,
                            args.num_goals,
                            args.iterations,
                            args.ratio,
                            args.pixel_threshold,
                            args.human_id,
                            args.save_path,
                            args.seed,
                            args.env)
    scenario = SocialNavScenario(scenario_config)
    scenario.record_data()



if __name__ == '__main__':
    main()