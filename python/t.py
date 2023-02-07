import argparse
import par

args = par.default_argument_parser().parse_args()

if args.test:
    train_flag = False
else:
    train_flag = True 

if args.bs:
    batch_size = args.bs
else:
    batch_size = args.default

if args.nn:
    nn = args.nn
else:
    nn = args.default

print(f'train_flag:{train_flag}')
print(f'batch_size:{batch_size}')
print(f'nn:{nn}')
print(f'type(nn):{type(nn)}')