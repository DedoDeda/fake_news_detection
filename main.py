from os import path

from log import log
import dataset as ds
import network as net


LOAD_NETWORK = True
TRAINING = False

NETWORK_OUT_SIZE = 2
NETWORK_EMBEDDING_SIZE = 128

NETWORK_STATE_PATH = path.join('network', 'network_state.pt')

BATCH_SIZE = 128
SHUFFLE_DATASET = True

LEARNING_RATE = 1e-3
NUM_EPOCHS = 10


def main():
    log(f'PyTorch device: {str(net.DEVICE).upper()}.')
    log(f'Mode: {'training' if TRAINING else 'testing'}.')

    log(f'Starting to load datasets...')
    train_dataset, test_dataset, vocab = ds.load_datasets_and_vocab()
    log(f'Finished loading datasets.')

    if LOAD_NETWORK:
        log(f'Starting to load network...')
        network = net.load_network(len(vocab), NETWORK_OUT_SIZE, NETWORK_EMBEDDING_SIZE, NETWORK_STATE_PATH)
        log(f'Finished loading network.')
    else:
        network = net.create_network(len(vocab), NETWORK_OUT_SIZE, NETWORK_EMBEDDING_SIZE)
        log(f'Created brand new network.')

    if TRAINING:
        net.train(network, train_dataset, BATCH_SIZE, SHUFFLE_DATASET, LEARNING_RATE, NUM_EPOCHS)
        log('Starting to save network...')
        net.save_network(network, NETWORK_STATE_PATH)
        log('Finished saving network.')
    else:
        net.test(network, test_dataset, BATCH_SIZE, SHUFFLE_DATASET)


if __name__ == '__main__':
    main()
