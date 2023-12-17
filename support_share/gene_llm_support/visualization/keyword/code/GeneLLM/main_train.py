from wrapper import analyze
import argparse


def main(args):
    analyze(args.input_data_path, args.task_type, args.task_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualize keywords.')
    parser.add_argument("--input-data-path",help="Supply data path to train", type=str,default='../../data/cellular_locations.csv')
    parser.add_argument("--task-type",help="Type of GeneLLM task", type=str, default='classification')
    parser.add_argument("--task-name",help="Name of GeneLLM task",type=str, default='subcellular_localization')
    args = parser.parse_args()
    main(args)