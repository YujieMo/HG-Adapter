from ruamel.yaml import YAML
import os
import argparse
from models import train
from utils import process

def get_args(model_name, dataset, custom_key="", yaml_path=None) -> argparse.Namespace:
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    # input arguments
    parser = argparse.ArgumentParser()
    if not custom_key == None:
        custom_key = custom_key.split("+")[0]
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--custom-key", default=custom_key)
    parser.add_argument("--dataset", default=dataset)
    parser.add_argument('--gpu_num', nargs='?', default='0')
    parser.add_argument('--lr', type = float, default = 0.005)
    parser.add_argument('--hid_units', type=int, default=256)
    parser.add_argument('--out_ft', type=int, default=512)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--prompt_num', type=int, default=1)
    parser.add_argument('--bottleneck_dim', type=int, default=4)
    parser.add_argument('--bottleneck_dim_2', type=int, default=64)
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--lammda', type=float, default=0.005)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--margin', type=float, default=1)
    parser.add_argument('--upload_pa', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--dropout_cla', type=float, default=0.2)
    parser.add_argument('--dropout_adj', type=float, default=0.2)

    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")
    args = parser.parse_args()
    return args



def main():
    process.setup_seed(0)
    args = get_args(
        model_name="SHGL",
        dataset="ACM",  #Heterogeneous graph: ACM, Yelp, DBLP, Aminer
        custom_key="Node",  # Node: node cilassfication 
    )
    if args.dataset in ["ACM", "Yelp", "DBLP", "Aminer"]:
        embedder = train(args)
    mac, mic = embedder.training()

if __name__ == '__main__':
    main()
