import os
import argparse
import torch

from generators.generator import Seq2SeqGeneratorAllUser
from generators.generator import GeneratorAllUser
from trainers.sequence_trainer import SeqTrainer
from utils.utils import set_seed
from utils.logger import Logger


parser = argparse.ArgumentParser()


parser.add_argument("--model_name", 
                    default='fcrllm_sasrec',
                    choices=["fcrllm_gru4rec", "fcrllm_sasrec"],
                    type=str, 
                    required=False,
                    help="model name")
parser.add_argument("--dataset", 
                    default="yelp", 
                    choices=["yelp", "fashion", "beauty",], 
                    help="Choose the dataset")
parser.add_argument("--inter_file",
                    default="inter",
                    type=str,
                    help="the name of interaction file")
parser.add_argument("--demo", 
                    default=False, 
                    action='store_true', 
                    help='whether run demo')
parser.add_argument("--pretrain_dir",
                    type=str,
                    default="sasrec_seq",
                    help="the path that pretrained model saved in")
parser.add_argument("--output_dir",
                    default='./saved/',
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--check_path",
                    default='',
                    type=str,
                    help="the save path of checkpoints for different running")
parser.add_argument("--do_test",
                    default=False,
                    action="store_true",
                    help="whehther run the test on the well-trained model")
parser.add_argument("--do_emb",
                    default=False,
                    action="store_true",
                    help="save the user embedding derived from the SRS model")
parser.add_argument("--do_group",
                    default=False,
                    action="store_true",
                    help="conduct the group test")
parser.add_argument("--keepon",
                    default=False,
                    action="store_true",
                    help="whether keep on training based on a trained model")
parser.add_argument("--keepon_path",
                    type=str,
                    default="normal",
                    help="the path of trained model for keep on training")
parser.add_argument("--clip_path",
                    type=str,
                    default="",
                    help="the path to save the CLIP-pretrained embedding and adapter")
parser.add_argument("--ts_user",
                    type=int,
                    default=10,
                    help="the threshold to split the short and long seq")
parser.add_argument("--ts_item",
                    type=int,
                    default=20,
                    help="the threshold to split the long-tail and popular items")

# Model parameters
parser.add_argument("--hidden_size",
                    default=64,
                    type=int,
                    help="the hidden size of embedding")
parser.add_argument("--trm_num",
                    default=2,
                    type=int,
                    help="the number of transformer layer")
parser.add_argument("--num_heads",
                    default=1,
                    type=int,
                    help="the number of heads in Trm layer")
parser.add_argument("--num_layers",
                    default=1,
                    type=int,
                    help="the number of GRU layers")
parser.add_argument("--dropout_rate",
                    default=0.5,
                    type=float,
                    help="the dropout rate")
parser.add_argument("--max_len",
                    default=200,
                    type=int,
                    help="the max length of input sequence")
parser.add_argument("--aug",
                    default=False,
                    action="store_true",
                    help="whether augment the sequence data")
parser.add_argument("--aug_seq",
                    default=False,
                    action="store_true",
                    help="whether use the augmented data")
parser.add_argument("--aug_seq_len",
                    default=0,
                    type=int,
                    help="the augmented length for each sequence")
parser.add_argument("--aug_file",
                    default="inter",
                    type=str,
                    help="the augmentation file name")
parser.add_argument("--train_neg",
                    default=1,
                    type=int,
                    help="the number of negative samples for training")
parser.add_argument("--test_neg",
                    default=100,
                    type=int,
                    help="the number of negative samples for test")
parser.add_argument("--prompt_num",
                    default=2,
                    type=int,
                    help="the number of prompts")
parser.add_argument("--freeze",
                    default=False,
                    action="store_true",
                    help="whether freeze the pretrained architecture when finetuning")
parser.add_argument("--pg",
                    default="length",
                    choices=['length', 'attention'],
                    type=str,
                    help="choose the prompt generator")
parser.add_argument("--num_hopblock",
                    default=1,
                    type=int,
                    help="the number of hopblock")
parser.add_argument("--gamma_fc",
                    default=0.1,
                    type=float,
                    help="the weight of alignment loss")
parser.add_argument("--item_reg",
                    default=False,
                    action="store_true",
                    help="whether regularize the item embedding by CL")
parser.add_argument("--beta",
                    default=0.1,
                    type=float,
                    help="the weight of regulation loss")
parser.add_argument("--sim_user_num",
                    default=10,
                    type=int,
                    help="the number of similar users for enhancement")
parser.add_argument("--beta_hopfield",
                    default=0.1,
                    type=float,
                    help="softmax temperature")
parser.add_argument("--alpha_hopfield",
                    default=0.1,
                    type=float,
                    help="the strength of penalty")                    
parser.add_argument("--gamma_update",
                    default=0.1,
                    type=float,
                    help="the weight of teacher update")
parser.add_argument("--hyb_alpha",
                    default=0.5,
                    type=float,
                    help="trade-off weight ST and CT")


# Other parameters
parser.add_argument("--train_batch_size",
                    default=512,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--lr",
                    default=0.001,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2",
                    default=0,
                    type=float,
                    help='The L2 regularization')
parser.add_argument("--num_train_epochs",
                    default=100,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--lr_dc_step",
                    default=1000,
                    type=int,
                    help='every n step, decrease the lr')
parser.add_argument("--lr_dc",
                    default=0,
                    type=float,
                    help='how many learning rate to decrease')
parser.add_argument("--patience",
                    type=int,
                    default=10,
                    help='How many steps to tolerate the performance decrease while training')
parser.add_argument("--watch_metric",
                    type=str,
                    default='NDCG@10',
                    help="which metric is used to select model.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for different data split")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--gpu_id',
                    default=0,
                    type=int,
                    help='The device id.')
parser.add_argument('--num_workers',
                    default=0,
                    type=int,
                    help='The number of workers in dataloader')
parser.add_argument("--log", 
                    default=False,
                    action="store_true",
                    help="whether create a new log file")

torch.autograd.set_detect_anomaly(True)

args = parser.parse_args()
set_seed(args.seed) 
args.output_dir = os.path.join(args.output_dir, args.dataset)
args.pretrain_dir = os.path.join(args.output_dir, args.pretrain_dir)
args.output_dir = os.path.join(args.output_dir, args.model_name)
args.keepon_path = os.path.join(args.output_dir, args.keepon_path)
args.output_dir = os.path.join(args.output_dir, args.check_path)    


def main():

    log_manager = Logger(args) 
    logger, writer = log_manager.get_logger() 
    args.now_str = log_manager.get_now_str()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")


    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_name in ['fcrllm_gru4rec']:
        generator = GeneratorAllUser(args, logger, device)
    elif args.model_name in ["fcrllm_sasrec"]:
        generator = Seq2SeqGeneratorAllUser(args, logger, device)
    else:
        raise ValueError

    trainer = SeqTrainer(args, logger, writer, device, generator)
    
    model = trainer.get_model()

    if args.do_test:
        trainer.test()
    elif args.do_emb:
        trainer.save_user_emb()
    elif args.do_group:
        trainer.test_group()
    else:
        trainer.train()

    log_manager.end_log()  



if __name__ == "__main__":

    main()



