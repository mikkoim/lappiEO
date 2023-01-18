import wandb

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

import src.utils as ut
import src.classificationutils as cu
import dataset_stats

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", type=str, required=True,
                    help="Location of the csv containing filenames and labels")
parser.add_argument("--test_csv", type=str, required=True,
                    help="location of the test dataset csv")

parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset for statistics")

parser.add_argument("--external_data_csv", type=str, required=True,
                    help="location of the external dataset csv")

parser.add_argument("--img_size", type=int, default=49,
                    help="all tifs are resized to this size")
parser.add_argument("--batch_size", type=int, default=64,
                    help="training batch size")

parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--iterations", type=int, default=3)

parser.add_argument("--learning_rate", type=float, default=1e-4,
                    help="training learning rate")
parser.add_argument("--use_gpu", default=False, action="store_true")
parser.add_argument("--load_dataset_to_memory", default=False, action="store_true")
parser.add_argument("--channels", type=str, default="[0,1,2,3,4,5,6,7,8,9,10]",
                    help="input image channels that are used")
parser.add_argument("--output_model_name", type=str, required=True,
                    help="name of the output model")

parser.add_argument("--freeze_base", default=False, action="store_true")
parser.add_argument("--random_crop", default=False, action="store_true")

parser.add_argument("--pretrained_model", type=str, default=None,
                    help="uses a pretrained model given as string")
parser.add_argument("--N_channels_source", type=int, default=None)
parser.add_argument("--N_classes_source", type=int, default=None)
parser.add_argument("--wandb_project", type=str, default='tests')

args = parser.parse_args()

def main(args):

    DATASET = args.dataset
    STATS0 = dataset_stats.stats[DATASET]
    STATS = {}

    EXTRA_CSV = args.external_data_csv
    TRAIN_CSV = args.train_csv
    TEST_CSV = args.test_csv

    OUTPUT_SIZE=(args.img_size, args.img_size)
    BATCH_SIZE = args.batch_size

    EPOCHS = args.epochs
    ITERATIONS = args.iterations
    
    LEARNING_RATE = args.learning_rate
    GPU = args.use_gpu
    LOAD_TO_MEMORY = args.load_dataset_to_memory
    RANDOM_CROP = args.random_crop

    CHANNELS = list(map(int, args.channels.strip('[]').split(',')))
    print("Channels: ", CHANNELS)

    MODEL_NAME = args.output_model_name
    FREEZE_BASE = args.freeze_base

    PRETRAINED_MODEL = args.pretrained_model
    N_CHANNELS_SOURCE = args.N_channels_source
    N_CLASSES_SOURCE = args.N_classes_source
    PROJECT = args.wandb_project

    STATS['mean'] = STATS0['mean'][CHANNELS]
    STATS['std'] = STATS0['std'][CHANNELS]
    
    TF_PRELOAD = transforms.Normalize(STATS['mean'], STATS['std'])
    
    TF_TRAIN = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    ])

    TF_NOISE_0 = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomApply([transforms.GaussianBlur((3,3))],
                                        p=0.5)
                    ])
    cropmax = (args.img_size+1)//2
    if RANDOM_CROP:
        TF_NOISE = lambda x: transforms.CenterCrop(np.random.randint(1,cropmax)*2+1)(TF_NOISE_0(x))
    else:
        TF_NOISE = TF_NOISE_0

    print("Nodata value: ", STATS0['nodata'])
    def remove_nodata(img):
        img[img==STATS0['nodata']] = 0
        return img
    array_transform = remove_nodata

    fnames_extra, _ = ut.read_fname_csv(EXTRA_CSV)
    fnames_train, labels_train = ut.read_fname_csv(TRAIN_CSV)
    fnames_test, labels_test = ut.read_fname_csv(TEST_CSV)

    y, le = ut.encode_labels(labels_train+labels_test)
    ut.print_label_counts(y, le)
    N_classes = len(le.classes_)
    print("N filenames: ", len(fnames_train+fnames_test))
    print("N classes: ", N_classes)
    y_train = le.transform(labels_train)
    y_test = le.transform(labels_test)


    # Datasets and dataloaders
    print("Create dataloaders...")

    extraset = cu.ImagePathDataset(fnames_extra,
                                None, 
                                output_size=OUTPUT_SIZE,
                                channels=CHANNELS,
                                array_transform=array_transform,
                                preload_tensor_transform=TF_PRELOAD,
                                load_to_memory=LOAD_TO_MEMORY)

    trainset = cu.ImagePathDataset(fnames_train, 
                                y_train, 
                                output_size=OUTPUT_SIZE,
                                channels=CHANNELS,
                                array_transform=array_transform,
                                preload_tensor_transform=TF_PRELOAD,
                                tensor_transform=TF_TRAIN,
                                load_to_memory=LOAD_TO_MEMORY)

    testset = cu.ImagePathDataset(fnames_test, 
                                y_test, 
                                output_size=OUTPUT_SIZE,
                                channels=CHANNELS,
                                array_transform=array_transform,
                                preload_tensor_transform=TF_PRELOAD,
                                load_to_memory=LOAD_TO_MEMORY)
    
    subset_ind = torch.randperm(len(trainset))[:len(testset)]
    train_subset = torch.utils.data.Subset(trainset, subset_ind)

    extraloader = torch.utils.data.DataLoader(extraset, batch_size=BATCH_SIZE, shuffle=True)
    trainloader = torch.utils.data.DataLoader(extraset, batch_size=BATCH_SIZE, shuffle=True)
    
    eval_loader_test = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
    eval_loader_train = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE)


    print('done')

    def softCrossEntropy(input, target):
        logprobs = F.log_softmax(input, dim=1)
        return  -(F.softmax(target, dim=1) * logprobs).sum(dim=1).mean()


    hparams = {'model': 'distillation',
                'out_name': MODEL_NAME,
                'output_size': OUTPUT_SIZE,
                'batch_size': BATCH_SIZE,
                'lr': LEARNING_RATE,
                'N_classes': N_classes,
                'N_channels': len(CHANNELS),
                'epochs': EPOCHS,
                'iterations': ITERATIONS,
                'train_aug': TF_TRAIN}
    
    print(hparams)

    wandb.init(project=PROJECT, 
       config=hparams)
    
    def validate(model, dataloader, n_guess=None, tf=None):
        y_true, y_pred, _ = cu.torch_predict(model, dataloader, gpu=GPU, n_guess=n_guess, tf=tf)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return acc, f1
    
    step = 1
    for iteration in range(ITERATIONS):
        
        print(f"Iteration: {iteration+1}")
        
        print("Using teacher model {}".format(PRETRAINED_MODEL))
        teacher = cu.Sentinel2_ResNet50(N_CHANNELS_SOURCE,
                                        N_CLASSES_SOURCE, freeze_base=FREEZE_BASE)

        student = cu.Sentinel2_ResNet50(N_CHANNELS_SOURCE, 
                                        N_CLASSES_SOURCE, freeze_base=FREEZE_BASE)
        wandb.watch(student)

        teacher.load_state_dict(torch.load(PRETRAINED_MODEL))
        #student.load_state_dict(torch.load(PRETRAINED_MODEL))

        if GPU:
            print("Using GPU")
            student.to('cuda')
            teacher.to('cuda')
        else:
            print("Using CPU")

        optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)

        student.train()
        teacher.eval()

        for epoch in range(EPOCHS):
            print(f"Epoch: {epoch+1}")

            # Train
            #extra_iter = iter(extraloader)
            for batch in tqdm(trainloader):
                #x_e, _ = next(extra_iter)
                x, y = batch
                if GPU:
                    #x_e = x_e.to('cuda')
                    x = x.to('cuda')
                    y = y.to('cuda')

                #t_output_e = cu.multi_guess(x, TF_NOISE, teacher, guesses=5)
                t_output_e = teacher(x)
                s_output_e = student(TF_NOISE(x))
                #s_output = student(TF_NOISE(x))

                soft_loss = softCrossEntropy(s_output_e, t_output_e)
                #hard_loss = F.cross_entropy(s_output, y)

                loss = soft_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    wandb.log({"loss":loss.item(), "epoch":epoch, "iteration":iteration})
                step += 1

            # Validation
            acc_test, f1_test = validate(student, eval_loader_test)
            acc_test_multi, f1_test_multi = validate(student, eval_loader_test, n_guess=5, tf=TF_NOISE)
            acc_train, f1_train = validate(student, eval_loader_train)

            wandb.log({"val acc":acc_test, 
                        "val f1":f1_test, 
                        "train acc":acc_train, 
                        "train f1":f1_train,
                        "val acc multi":acc_test_multi,
                        "val f1 multi": f1_test_multi,
                        "epoch":epoch})

        PRETRAINED_MODEL = f'{MODEL_NAME}_{iteration:02}.pt'
        torch.save(student.state_dict(), PRETRAINED_MODEL)

    wandb.finish()
if __name__=='__main__':
    main(args)
    exit()
