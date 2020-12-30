import glob
import os

# Defence 1
# Eiffel 2
# Invalides 3
# Louvre 4
# Moulin Rouge 5
# Musée d'Orsay 6
# Notre Dame 7
# Pantheon 8
# Pompidou 9
# Sacre Coeur 10

i = 0
for files in os.listdir('C:/Users/inesp/Documents/3A Inès/Deep Learning/Projet/Paris'):
    if files != 'Landmarks.py' and files != 'general' and files != 'AlexNet.ipynb':
        os.chdir('C:/Users/inesp/Documents/3A Inès/Deep Learning/Projet/Paris/')
        if i == 0:
            f = open("Id_landmark_test.txt", "a")
        elif i == 1:
            f = open("Id_landmark_train.txt", "a")
        elif i == 2:
            f = open("Id_landmark_train_2%.txt", "a")
        i += 1
        for filename in os.listdir('C:/Users/inesp/Documents/3A Inès/Deep Learning/Projet/Paris/' + files):
            os.chdir('C:/Users/inesp/Documents/3A Inès/Deep Learning/Projet/Paris/' + files + '/' + filename)
            for image in list(glob.glob('paris_defense*')):
                f.write(image + ' 1\n')
            for image in list(glob.glob('paris_eiffel*')):
                f.write(image + ' 2\n')
            for image in list(glob.glob('paris_invalides*')):
                f.write(image + ' 3\n')
            for image in list(glob.glob('paris_louvre*')):
                f.write(image + ' 4\n')
            for image in list(glob.glob('paris_moulinrouge*')):
                f.write(image + ' 5\n')
            for image in list(glob.glob('paris_museedorsay*')):
                f.write(image + ' 6\n')
            for image in list(glob.glob('paris_notredame*')):
                f.write(image + ' 7\n')
            for image in list(glob.glob('paris_pantheon*')):
                f.write(image + ' 8\n')
            for image in list(glob.glob('paris_pompidou*')):
                f.write(image + ' 9\n')
            for image in list(glob.glob('paris_sacrecoeur*')):
                f.write(image + ' 10\n')
        f.close()