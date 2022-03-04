import sys
import os
import shutil
import numpy as np



if __name__ == '__main__':

    path = 'data/Brats_2020_slices/MICCAI_BraTS2020_TrainingData_slices'
    #d,counts = np.unique(np.array(["_".join(x.split("_")[:-1]) for x in os.listdir(path)]), return_counts=True)
    d = ["_".join(x.split("_")) for x in os.listdir(path)]

    data_ids = [s for s in d if s.startswith("Subject")]

    nr_of_clients = int(sys.argv[1])
    training_subjects_per_client = int(sys.argv[2])
    validation_subjects_per_client = int(sys.argv[3])
    partition_sets = {i: [] for i in range(nr_of_clients)}

    subjects_per_client = training_subjects_per_client + validation_subjects_per_client

    split_points = np.arange(subjects_per_client, subjects_per_client * (nr_of_clients + 1), subjects_per_client)

    if os.path.exists('data/datapartitions'):
        shutil.rmtree('data/datapartitions')
    else:
        os.makedirs('data/datapartitions')


    for i in range(nr_of_clients):
        os.makedirs(os.path.join('data/datapartitions',str(i)))
        os.makedirs(os.path.join('data/datapartitions', str(i), 'train_set'))
        os.makedirs(os.path.join('data/datapartitions', str(i), 'validation_set'))
    for p in data_ids:

        print("p: ", p)
        subject = int(p.split("/")[-1].split("_")[1])
        print("subject: ", subject)

        if subject <= subjects_per_client * nr_of_clients:
            dp = np.where(subject <= split_points)[0][0]

            if (subject - 1) % subjects_per_client == training_subjects_per_client:

                shutil.copy2(os.path.join(path, p), os.path.join('data/datapartitions', str(dp), 'validation_set') )

            else:
                shutil.copy2(os.path.join(path, p), os.path.join('data/datapartitions', str(dp), 'train_set'))

