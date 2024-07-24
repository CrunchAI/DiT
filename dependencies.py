import os
os.system('pip install -r requirements.txt')
os.system('git config --global user.name ' + input("your_username: "))
os.system('git config --global user.email ' + input("your_email_address@example.com: "))

import os
os.system('git clone https://github.com/facebookresearch/detectron2.git')
os.system('pip install -e detectron2')
os.system("git clone https://github.com/microsoft/unilm.git")
os.system("sed -i 's/from collections import Iterable/from collections.abc import Iterable/' unilm/dit/object_detection/ditod/table_evaluation/data_structure.py")
os.system("curl -LJ -o publaynet_dit-b_cascade.pth 'https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_cascade.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D'")

import sys
sys.path.append("unilm")
sys.path.append("detectron2")