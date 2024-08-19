import argparse
from pathlib import Path
import os, sys
import h5py  # HDF5 파일을 다루기 위한 라이브러리
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.test_part import forward
import time
    
def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_Varnet', help='Name of network')
    parser.add_argument('-p', '--path_data', type=Path, default='/home/Data/', help='Directory of test data')
    
    parser.add_argument('--cascade', type=int, default=2, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=5, help='Number of channels for sensitivity map U-Net')
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')

    args = parser.parse_args()
    return args


def save_reconstructions_and_targets(reconstructions, targets, save_path):
    """
    reconstruction 데이터와 target 데이터를 함께 HDF5 파일에 저장하는 함수
    """
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('reconstruction', data=reconstructions)
        f.create_dataset('target', data=targets)
    print(f"Saved reconstruction and target to {save_path}")


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    
    public_acc, private_acc = None, None

    # assert(len(os.listdir(args.path_data)) == 2)

    for acc in os.listdir(args.path_data):
        if acc in ['train']:
            public_acc = acc
        if acc in ['val']:
            private_acc = acc
        
    assert(None not in [public_acc, private_acc])
    
    start_time = time.time()
    
    # train Acceleration 경로 설정
    args.data_path = args.path_data / public_acc
    args.forward_dir = '../result' / args.net_name / 'reconstructions_train'
    print(f'Saving reconstructions to {args.forward_dir}')
    
    # forward 함수 호출하여 reconstruction 결과 생성
    reconstructions = forward(args)
    if reconstructions is None or len(reconstructions) == 0:
        raise ValueError("Reconstruction 데이터가 비어 있습니다. forward 함수의 출력을 확인하세요.")


    # target 데이터를 train 디렉토리에서 읽어옴
    image_dir = args.path_data / 'train/image'
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.h5'):
            image_path = image_dir / file_name
            with h5py.File(image_path, 'r') as f:
                target = f['image_label'][:]  # image_label 데이터 가져오기
                print(f"Loaded target from {image_path}")
            
            # reconstruction 파일 이름과 동일한 형식으로 저장
            save_path = args.forward_dir / file_name
            save_reconstructions_and_targets(reconstructions, target, save_path)
    
    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')
