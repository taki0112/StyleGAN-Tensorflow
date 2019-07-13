from StyleGAN import StyleGAN
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of StyleGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train, test, draw]')
    parser.add_argument('--draw', type=str, default='uncurated', help='[uncurated, style_mix, truncation_trick]')
    parser.add_argument('--dataset', type=str, default='FFHQ', help='The dataset name what you want to generate')

    parser.add_argument('--iteration', type=int, default=120, help='The number of images used in the train phase')
    parser.add_argument('--max_iteration', type=int, default=2500, help='The total number of images')

    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch in the test phase')
    parser.add_argument('--gpu_num', type=int, default=1, help='The number of gpu')

    parser.add_argument('--progressive', type=str2bool, default=True, help='use progressive training')
    parser.add_argument('--sn', type=str2bool, default=False, help='use spectral normalization')

    parser.add_argument('--start_res', type=int, default=8, help='The number of starting resolution')
    parser.add_argument('--img_size', type=int, default=1024, help='The target size of image')
    parser.add_argument('--test_num', type=int, default=100, help='The number of generating images in the test phase')

    parser.add_argument('--seed', type=str2bool, default=True, help='seed in the draw phase')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        gan = StyleGAN(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

        if args.phase == 'draw' :
            if args.draw == 'style_mix' :
                gan.draw_style_mixing_figure()
                print(" [*] Style mix finished!")

            elif args.draw == 'truncation_trick' :
                gan.draw_truncation_trick_figure()
                print(" [*] Truncation_trick finished!")

            else :
                gan.draw_uncurated_result_figure()
                print(" [*] Un-curated finished!")

if __name__ == '__main__':
    main()
