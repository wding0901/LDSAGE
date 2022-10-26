import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed

from model import LightSAGE
from trainer import MyTrainer
from recbole.quick_start import run_recbole


def run_baseline(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    run_recbole(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict, saved=saved)


def run_light_sage(args):
    # configurations initialization
    config = Config(model=LightSAGE, dataset=args.dataset, config_file_list=args.config_file_list)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = LightSAGE(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = MyTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gowalla',
                        help='The datasets can be: yelp, amazon-books, gowalla.')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    parser.add_argument('--model', type=str, default='LightSAGE',
                        help='The models can be: LightSAGE, NGCF, BPR, NeuMF...')
    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = ['args/default.yaml']
    if args.config is not '':
        args.config_file_list.append(args.config)

    if args.dataset in ['gowalla', 'yelp', 'amazon-books']:
        args.config_file_list.append(f'args/{args.dataset}.yaml')

    if args.model in ['LightSAGE']:
        run_light_sage(args)
    else:
        run_baseline(model=args.model, dataset=args.dataset, config_file_list=args.config_file_list)
