import os
import sys
import argparse
   
  
def main(args):
    counter = 1
    name = args.locexp
    for s in [1, 2, 3 ,4]:
        print("Round {}".format(counter))
        os.system(f'python3 ./main.py \
                --locexp {name} \
                --seed {s} ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter search')
    parser.add_argument('--locexp', type=str) 
    main(parser.parse_args())

