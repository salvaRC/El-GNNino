import argparse


data_dir = "data/"

def get_argparser(experiment="ersstv5"):
    """

    :param experiment: ersstv5, or cnn_data
    :return:
    """

    parser = argparse.ArgumentParser(description='PyTorch ENSO Time series forecasting')
    parser.add_argument("--data_dir", type=str, default=r'data/')
    parser.add_argument('--load_data', type=bool, default=False)  # whether to load processed data from data_path
    parser.add_argument('--store_data', type=bool, default=False)  # whether to store processed data to data_path
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--L1Loss', type=bool, default=False)  # L2 loss default
    parser.add_argument('--normalize', type=int, default=0)  # 0 means: as is
    # parser.add_argument('--drop_imfs', type=int, default=3, help='How many EEMD IMFs to drop')
    parser.add_argument('--shuffle', type=bool, default=True)  # shuffle training batches?
    parser.add_argument('--device', type=str, default='cuda', help='')
    parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--adaptive_edges', type=bool, default=True,
                        help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--subgraph_size', type=int, default=20, help='k')
    parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
    parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension/#features per node')
    parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
    parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
    parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
    parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
    parser.add_argument('--end_channels', type=int, default=64, help='end channels')
    parser.add_argument('--window', type=int, default=3,
                        help='input sequence length')  # how many time steps used for prediction?...
    parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length')
    parser.add_argument('--horizon', type=int, default=6)  # predict horizon months in advance...
    parser.add_argument('--layers', type=int, default=2, help='number of layers')

    parser.add_argument('--prelu', type=bool, default=True,
                        help='whether to use PReLU instead of ReLU for final layers')
    parser.add_argument('--graph_prelu', type=bool, default=False,
                        help='whether to  use PReLU in the graph learning module')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=6e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')

    parser.add_argument('--clip', type=int, default=5, help='clip')

    parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
    parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')

    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--transfer_epochs', type=int, default=100, help='Only used if transfer learning is done.')
    parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
    parser.add_argument('--step_size', type=int, default=100, help='step_size')
    parser.add_argument("--resolution", type=int, default=5, help="Which grid resolution to use")
    if experiment.lower() == "ersstv5":
        parser.add_argument("--use_heat_content", type=bool, default=False,
                            help="Whether to use heat content anomalies")
        parser.add_argument("--drop_imfs", type=int, default=3, help="EEMD")

        parser.add_argument("--train_all_nodes", type=bool, default=False,
                            help="Whether to train on all nodes or only ONI region ones")

        parser.add_argument('--save', type=str, default='models/exp1/', help='path to save the final model')
    else:
        raise ValueError()
    parser.add_argument('--validation_frac', type=float, default=0.15, help='Validation set fraction')

    parser.add_argument('--lon_min', type=int, default=190, help='Longitude min. (Eastern)')
    parser.add_argument('--lon_max', type=int, default=240, help='Longitude max. (Eastern)')
    parser.add_argument('--lat_min', type=int, default=-5, help='Latitude min. (Southern)')
    parser.add_argument('--lat_max', type=int, default=5, help='Latitude max. (Southern)')
    parser.add_argument('--index', type=str, default="ONI", help='Which index to predict')
    return parser
