# importing os module
import os
path = '/Users/ismaillakhani/Desktop/Facial Expression/Sai Priya/practice/Neutral/'
files = os.listdir(path)

# os.rename(src, dst) : src is source address of file to be renamed and
# dst is destination with the new name.

for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join(['Nu_', str(index), '.jpg'])))
    