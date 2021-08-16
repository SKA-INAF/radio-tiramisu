import os
import random
import shutil
import math

galaxies = []
sidelobes = []
sources = []

for filename in os.listdir("E:\INAF\pytorch_tiramisu-master\data\img"):
    if "galaxy" in filename:
        galaxies.append(filename)
    if "sidelobe" in filename:
        sidelobes.append(filename)
    if "source" in filename:
        sources.append(filename)

random.shuffle(galaxies)
random.shuffle(sidelobes)
random.shuffle(sources)

galaxies_train = galaxies[:math.floor(len(galaxies) * 0.7)]
sidelobes_train = sidelobes[:math.floor(len(sidelobes) * 0.7)]
sources_train = sources[:math.floor(len(sources) * 0.7)]

galaxies_test = galaxies[math.floor(len(galaxies) * 0.7):math.floor(len(galaxies) * 0.8)]
sidelobes_test = sidelobes[math.floor(len(sidelobes) * 0.7):math.floor(len(sidelobes) * 0.8)]
sources_test = sources[math.floor(len(sources) * 0.7):math.floor(len(sources) * 0.8)]

galaxies_val = galaxies[math.floor(len(galaxies) * 0.8):]
sidelobes_val = sidelobes[math.floor(len(sidelobes) * 0.8):]
sources_val = sources[math.floor(len(sources) * 0.8):]


for galaxy in galaxies_train:
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\img\\" + galaxy, "E:\INAF\pytorch_tiramisu-master\data\\train\\" + galaxy)
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\imgannot\\" + galaxy, "E:\INAF\pytorch_tiramisu-master\data\\trainannot\\" + galaxy)

for galaxy in galaxies_test:
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\img\\" + galaxy, "E:\INAF\pytorch_tiramisu-master\data\\test\\" + galaxy)
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\imgannot\\" + galaxy, "E:\INAF\pytorch_tiramisu-master\data\\testannot\\" + galaxy)

for galaxy in galaxies_val:
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\img\\" + galaxy, "E:\INAF\pytorch_tiramisu-master\data\\val\\" + galaxy)
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\imgannot\\" + galaxy, "E:\INAF\pytorch_tiramisu-master\data\\valannot\\" + galaxy)

for sidelobe in sidelobes_train:
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\img\\" + sidelobe, "E:\INAF\pytorch_tiramisu-master\data\\train\\" + sidelobe)
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\imgannot\\" + sidelobe, "E:\INAF\pytorch_tiramisu-master\data\\trainannot\\" + sidelobe)

for sidelobe in sidelobes_test:
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\img\\" + sidelobe, "E:\INAF\pytorch_tiramisu-master\data\\test\\" + sidelobe)
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\imgannot\\" + sidelobe, "E:\INAF\pytorch_tiramisu-master\data\\testannot\\" + sidelobe)

for sidelobe in sidelobes_val:
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\img\\" + sidelobe, "E:\INAF\pytorch_tiramisu-master\data\\val\\" + sidelobe)
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\imgannot\\" + sidelobe, "E:\INAF\pytorch_tiramisu-master\data\\valannot\\" + sidelobe)

for source in sources_train:
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\img\\" + source, "E:\INAF\pytorch_tiramisu-master\data\\train\\" + source)
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\imgannot\\" + source, "E:\INAF\pytorch_tiramisu-master\data\\trainannot\\" + source)

for source in sources_test:
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\img\\" + source, "E:\INAF\pytorch_tiramisu-master\data\\test\\" + source)
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\imgannot\\" + source, "E:\INAF\pytorch_tiramisu-master\data\\testannot\\" + source)

for source in sources_val:
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\img\\" + source, "E:\INAF\pytorch_tiramisu-master\data\\val\\" + source)
    shutil.copy("E:\INAF\pytorch_tiramisu-master\data\imgannot\\" + source, "E:\INAF\pytorch_tiramisu-master\data\\valannot\\" + source)